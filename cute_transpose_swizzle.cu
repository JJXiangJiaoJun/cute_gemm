#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
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

static const bool kIsSwizzle = true;

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
 typename G2SSmemLayout,
 typename S2GSmemLayout,
 typename G2STiledCopy,
 typename S2RTiledCopy,
 typename R2GTiledCopy
>
__global__ void TransposeDevice(Params<ProblemShape, ElementA, StrideA, ElementB, StrideB> params) {

  using namespace cute;
  static_assert(is_static<CtaTiler>::value, "");
  static_assert(cute::cosize_v<G2SSmemLayout> == cute::cosize_v<S2GSmemLayout>, "");

  using SharedStorage = SharedStorage<ElementA, G2SSmemLayout>;

  extern __shared__ char shared_memory[];

  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  CtaTiler cta_tiler;

  G2SSmemLayout g2s_smem_layout;
  S2GSmemLayout s2g_smem_layout;

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

  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), g2s_smem_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.A.begin()), s2g_smem_layout);

  ThrCopy g2s_thr_copy = g2s_tiled_copy.get_slice(threadIdx.x);
  Tensor tAgA = g2s_thr_copy.partition_S(gA);    ///< (CPY, CPY_M, CPY_N)
  Tensor tAsA = g2s_thr_copy.partition_D(sA);    ///< (CPY, CPY_M, CPY_N)

  /// we need predicated here
  /// predicate for m, n
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)));  ///

  Tensor cA = make_identity_tensor(shape(gA));
  Tensor tAcA = g2s_thr_copy.partition_S(cA);

  auto TileM = size<0>(gA);
  auto TileN = size<1>(gA);

  auto M = size<0>(mA);
  auto N = size<1>(mA);

  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<1>(tApA); ++n) {
      tApA(m, n) = ((blockIdx.x * TileM + get<0>(tAcA(0, m, n))) < M) && ((blockIdx.y * TileN + get<1>(tAcA(0, m, n))) < N);
    }
  }

  // copy(g2s_tiled_copy, tAgA, tAsA);
  copy_if(g2s_tiled_copy, tApA, tAgA, tAsA);


  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  ThrCopy s2r_thr_copy = s2r_tiled_copy.get_slice(threadIdx.x);
  Tensor tBsB = s2r_thr_copy.partition_S(sB);

  Tensor tBrB = make_fragment_like(tBsB);

  copy(s2r_tiled_copy, tBsB, tBrB);

  ThrCopy r2g_thr_copy = r2g_tiled_copy.get_slice(threadIdx.x);
  Tensor  tBgB = r2g_thr_copy.partition_D(gB);  ///< (CPY, CPY_M, CPY_N)

  auto N_iters = size<0, 0>(tBrB);

  Tensor tBrB_tranpsosed = make_fragment_like(tBgB);

  ///< we need register transpose
  for (int iter = 0; iter < N_iters; ++iter) {
    copy(tBrB(make_coord(iter, _), _, _), tBrB_tranpsosed(make_coord(_, iter), _, _));
  }

  Tensor tBpB = make_tensor<bool>(make_shape(size<0, 1>(tBgB), size<1>(tBgB), size<2>(tBgB)));  ///
  Tensor tBcB = r2g_thr_copy.partition_D(cA);  ///< (CPY, CPY_M, CPY_N)

  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < size<0>(tBpB); ++k) {
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(tBpB); ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<2>(tBpB); ++n) {
        tBpB(k, m, n) = ((blockIdx.x * TileM + get<0>(tBcB(make_coord(0, k), m, n))) < M) && ((blockIdx.y * TileN + get<1>(tBcB(make_coord(0, k), m, n))) < N);
      }
    }
  }

  // copy(r2g_tiled_copy, tBrB_tranpsosed, tBgB);
  copy_if(r2g_tiled_copy, tBpB, tBrB_tranpsosed, tBgB);

}

using ElementA = cute::half_t;
using ElementB = cute::half_t;

static const int kAlignment = 128 / cutlass::sizeof_bits<ElementA>::value;
using ThreadBlockShape = cutlass::MatrixShape<64, 64>;

static const int kSharedLoadAlignment = kAlignment / 2;

using G2SAccessType = cutlass::Array<ElementA, kAlignment>;
using SharedLoadAccessType = cutlass::Array<ElementA, kSharedLoadAlignment>;
using GlobalWriteAccessType = G2SAccessType;

template<
 typename ElementA,
 typename ElementB
>
float LaunchTransposeKernel(const ElementA *A, ElementB * B, int m, int n, int lda, int ldb, int iterations = 1) {

  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);

  auto prob_shape = make_shape(M, N);

  auto dA = make_stride(lda, _1{});
  auto dB = make_stride(_1{}, ldb);

  // Define CTA tile sizes (static)
  auto bM = Int<ThreadBlockShape::kRow>{};
  auto bN = Int<ThreadBlockShape::kColumn>{};

  auto cta_tiler = make_shape(bM, bN);

  TiledCopy G2STiledCopy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<G2SAccessType>, ElementA>{},
                                                     Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                                     Layout<Shape< _1, Int<kAlignment>>>{});

  TiledCopy S2RTiledCopy = make_tiled_copy(Copy_Atom<UniversalCopy<SharedLoadAccessType>, ElementA>{},
                                                     Layout<Shape<_8,_16>,Stride<_1,_8>>{},
                                                     Layout<Shape<_1,Int<kSharedLoadAlignment>>>{});

  TiledCopy R2GTiledCopy = make_tiled_copy(Copy_Atom<UniversalCopy<GlobalWriteAccessType>, ElementA>{},
                                                     Layout<Shape<_8, _16>,Stride<_1,_8>>{},
                                                     Layout<Shape<Int<kAlignment>, Int<kSharedLoadAlignment>>>{});

  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_64, Stride<_1, _8>>>{});

  auto g2s_A = make_layout(make_shape(make_shape(_8{}, _8{}), _64{}), make_stride(make_stride(_512{}, _64{}), _1{}));
  auto g2s_A_swizzle = raked_product(swizzle_atom, Layout<_8, _1>{});

  ///< shared->global shared-memory view
  auto s2g_B = make_layout(make_shape(bM, bN), make_stride(bN, _1{}));
  auto s2g_B_swizzle = tile_to_shape(swizzle_atom, make_shape(bM,bN));


  int smem_size = int(sizeof(SharedStorage<cute::half_t, decltype(g2s_A)>));

  dim3 dimBlock(size(G2STiledCopy));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  cudaEvent_t start, stop;
  float elapsed_time_ms;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  if constexpr (kIsSwizzle) {
    auto kernel_fptr = TransposeDevice<
      decltype(prob_shape), decltype(cta_tiler),
      cute::half_t, decltype(dA),
      cute::half_t, decltype(dB),
      decltype(g2s_A_swizzle),
      decltype(s2g_B_swizzle),
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

    for (int i = 0; i < iterations; ++i)
      kernel_fptr<<<dimGrid, dimBlock, smem_size>>>({prob_shape, A, dA, B, dB});

  } else {

    auto kernel_fptr = TransposeDevice<
      decltype(prob_shape), decltype(cta_tiler),
      cute::half_t, decltype(dA),
      cute::half_t, decltype(dB),
      decltype(g2s_A),
      decltype(s2g_B),
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

    for (int i = 0; i < iterations; ++i)
      kernel_fptr<<<dimGrid, dimBlock, smem_size>>>({prob_shape, A, dA, B, dB});
  }

  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed_time_ms / iterations;
}


using HostKernel = reference::Transpose<ElementA, ElementB>;

void host_transpose(const ElementA *ptr_A, ElementB *ptr_B, int m, int n, int lda, int ldb) {
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

  ElementA *d_A;
  ElementB *d_B;


  cudaMalloc(&d_A, M * N * sizeof(ElementA));
  cudaMalloc(&d_B, N * M * sizeof(ElementB));

  cudaMemcpy(d_A, h_A, M * N * sizeof(ElementA), cudaMemcpyHostToDevice);

  float kernel_execute_time = 0;

  kernel_execute_time = LaunchTransposeKernel<ElementA, ElementB>(d_A, d_B, M, N, N, M, 50);

  printf("Trasnpose kernel elapsed time: %.4f us\n", kernel_execute_time * 1000);

  // for (int i = 0; i < 20; i++)

  cudaMemcpy(result_B, d_B, M * N * sizeof(ElementB), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

#ifdef HOST_CHECK
  host_transpose(h_A, h_B, M, N, N, M);

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

  int copy_iter = 100;

  cudaEventRecord(start, 0);

  for (int i = 0; i < copy_iter; ++i)
    cudaMemcpy(d_B, d_A, M * N * sizeof(ElementA), cudaMemcpyDeviceToDevice);

  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&elapsed_time, start, stop);

  float avg_time_ms = (elapsed_time / copy_iter);

  printf("Device to Device copy avg time: %.4f us\n", avg_time_ms * 1000);

  double transaction_bytes = M * N * sizeof(ElementA) * copy_iter;

  double bandwidthInGBs = (2.0f * transaction_bytes) / (float)1e9;
  float time_s = elapsed_time / (float)1e3;
  bandwidthInGBs = bandwidthInGBs / time_s;

  printf("Device to Device band-width : %.4lf GBs\n", bandwidthInGBs);

  delete[] h_A;
  delete[] h_B;
  delete[] result_B;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}