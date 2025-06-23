#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cute/tensor.hpp"

#include "reference/transpose.h"
#include "reference/initializer.h"

#define PRINT(STR)                                                             \
  do {                                                                         \
    printf("threadIdx: %d, %s:\n", threadIdx.x, #STR);                         \
    cute::print(STR);                                                          \
    printf("\n\n");                                                            \
  } while (0)

#define PRINT_TENSOR(STR)                                                      \
  do {                                                                         \
    printf("threadIdx: %d, %s:\n", threadIdx.x, #STR);                         \
    cute::print_tensor(STR);                                                   \
    printf("\n\n");                                                            \
  } while (0)

template <typename ProblemShape, typename ElementA, typename StrideA,
          typename ElementB, typename StrideB>
struct Params {
  ProblemShape problem_size;
  ElementA const *ptr_A;
  StrideA dA;
  ElementB *ptr_B;
  StrideB dB;
};

template<
 typename ElementA,
 class SmemLayoutA
>
struct SharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
};

template<
 typename ProblemShape,
 typename CtaTiler,
 typename ElementA,
 typename StrideA,
 typename ElementB,
 typename StrideB,
 typename SmemLayout,
 typename G2STiledCopy,
 typename S2RTiledCopy,
 typename R2GTiledCopy
>
__global__ void TransposeDevice(Params<ProblemShape, ElementA, StrideA, ElementB, StrideB> params) {

  using namespace cute;
  static_assert(is_static<CtaTiler>::value, "");

  using SharedStorage = SharedStorage<ElementA, SmemLayout>;

  extern __shared__ char shared_memory[];

  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  CtaTiler cta_tiler;

  SmemLayout smem_layout;

  G2STiledCopy g2s_tiled_copy;
  S2RTiledCopy s2r_tiled_copy;
  R2GTiledCopy r2g_tiled_copy;

  ////<
  Tensor mA = make_tensor(make_gmem_ptr(params.ptr_A), params.problem_size, params.dA);
  Tensor mB = make_tensor(make_gmem_ptr(params.ptr_B), params.problem_size, params.dB);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y);

  ///< partional tensor
  Tensor gA = local_tile(mA, cta_tiler, cta_coord);
  Tensor gB = local_tile(mB, cta_tiler, cta_coord);

  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), smem_layout);

  ThrCopy g2s_thr_copy = g2s_tiled_copy.get_slice(threadIdx.x);
  Tensor tAgA = g2s_thr_copy.partition_S(gA);    ///< (CPY, CPY_M, CPY_N)
  Tensor tAsA = g2s_thr_copy.partition_D(sA);    ///< (CPY, CPY_M, CPY_N)

  copy(g2s_tiled_copy, tAgA, tAsA);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();


  ThrCopy s2r_thr_copy = s2r_tiled_copy.get_slice(threadIdx.x);
  Tensor tBsA = s2r_thr_copy.partition_S(sA);

  Tensor tBrB = make_fragment_like(tBsA);

  copy(s2r_tiled_copy, tBsA, tBrB);

  ThrCopy r2g_thr_copy = r2g_tiled_copy.get_slice(threadIdx.x);
  Tensor tBgB = r2g_thr_copy.partition_D(gB); ///< (CPY, CPY_M, CPY_N)

  auto N_iters = size<0, 0>(tBrB);

  Tensor tBrB_tranpsosed = make_fragment_like(tBgB);

  ///< we need register transpose
  for (int iter = 0; iter < N_iters; ++iter) {
    copy(tBrB(make_coord(iter, _), _, _), tBrB_tranpsosed(make_coord(_, iter), _, _));
  }

  copy(r2g_tiled_copy, tBrB_tranpsosed, tBgB);

}



template<
 typename ElementA,
 typename ElementB
>
void LaunchTransposeKernel(const ElementA *A, ElementB * B, int m, int n, int lda, int ldb) {

  using namespace cute;
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);

  auto prob_shape = make_shape(M, N);

  auto dA = make_stride(lda, _1{});
  auto dB = make_stride(_1{}, ldb);

  // Define CTA tile sizes (static)
  auto bM = Int<64>{};
  auto bN = Int<64>{};

  auto cta_tiler = make_shape(bM, bN);

  TiledCopy G2STiledCopy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                                     Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                                     Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major

  TiledCopy S2RTiledCopy = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, cute::half_t>{},
                                                     Layout<Shape<_8,_16>,Stride<_1,_8>>{},  // Thr layout 16x8 k-major
                                                     Layout<Shape<_8,_4>, Stride<_4,_1>>{});               // Val layout  1x8 k-major

  TiledCopy R2GTiledCopy = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, cute::half_t>{},
                                                     Layout<Shape<_8, _16>,Stride<_1,_8>>{},  // Thr layout 16x8 k-major
                                                     Layout<Shape<_8, _4>>{});               // Val layout  1x8 k-major



  auto sA = make_layout(make_shape(bM, bN), make_stride(bN , _1{}));

  // print_layout(sA);

  int smem_size = int(sizeof(SharedStorage<cute::half_t, decltype(sA)>));

  dim3 dimBlock(size(G2STiledCopy));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  auto kernel_fptr = TransposeDevice<
    decltype(prob_shape), decltype(cta_tiler),
    cute::half_t, decltype(dA),
    cute::half_t, decltype(dB),
    decltype(sA),
    decltype(G2STiledCopy),
    decltype(S2RTiledCopy),
    decltype(R2GTiledCopy)
  >;

  //  Set L1 to be SMEM only
  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size>>>
      ({prob_shape, A, dA, B, dB});
}

using ElementA = cute::half_t;
using ElementB = cute::half_t;

using HostKernel = reference::Transpose<ElementA, ElementB>;

void host_gemm(const ElementA *ptr_A, ElementB *ptr_B, int m, int n, int lda, int ldb) {
  HostKernel host_op;
  host_op(ptr_A, ptr_B, m, n, lda, ldb);
}

int main() {

  int M = 2048;
  int N = 10240;

  ElementA *h_A = new ElementA[M * N];
  ElementB *h_B = new ElementB[N * M];

  ElementB *result_B = new ElementB[M * N];

  // reference::sequence_initializer<ElementA>::init(h_A, M * N);
  reference::random_initializer<ElementA>::init(h_A, M * N);

  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < K; ++j) {
  //     std::cout << h_A[i * K + j] << ",";
  //   }
  //   std::cout << std::endl;
  // }


  ElementA *d_A;
  ElementB *d_B;


  cudaMalloc(&d_A, M * N * sizeof(ElementA));
  cudaMalloc(&d_B, N * M * sizeof(ElementB));

  cudaMemcpy(d_A, h_A, M * N * sizeof(ElementA), cudaMemcpyHostToDevice);

  LaunchTransposeKernel<ElementA, ElementB>(d_A, d_B, M, N, N, M);

  // for (int i = 0; i < 20; i++)

  cudaMemcpy(result_B, d_B, M * N * sizeof(ElementB), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

#ifdef HOST_CHECK
  host_gemm(h_A, h_B, M, N, N, M);

  for (int i = 0; i < M * N; i++) {
    float abs_err = fabs(float(h_B[i]) - float(result_B[i]));
    if (abs_err > 1e-5) {
      std::cout <<"i: " << i << " cpu: " << float(h_B[i]) << "\tgpu: " << float(result_B[i]) << "\tdiff: " << abs_err << std::endl;
    }
  }
#endif

  cudaEvent_t start, stop;
  float elapsed_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int copy_iter = 50;


  cudaEventRecord(start);

  for (int i = 0; i < copy_iter; ++i)
    cudaMemcpy(d_B, d_A, M * N * sizeof(ElementA), cudaMemcpyDeviceToDevice);

  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Device to Device copy time: %.4f us\n", (elapsed_time / copy_iter) * 1000);

  delete[] h_A;
  delete[] h_B;
  delete[] result_B;

  cudaFree(d_A);
  cudaFree(d_B);



  return 0;
}