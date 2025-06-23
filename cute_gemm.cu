#include <cstdio>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cute/tensor.hpp"

#include "reference/gemm.h"
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


template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <typename ProblemShape,
          typename ElementA, typename StrideA, typename ElementB,
          typename StrideB, typename ElementC, typename StrideC>
struct Params {
  ProblemShape problem_size;
  ElementA const *ptr_A;
  StrideA dA;
  ElementB const *ptr_B;
  StrideB dB;
  ElementC *ptr_C;
  StrideC dC;
};

template<
 typename ProblemShape,
 typename TileShape,
 typename ElementA,
 typename StrideA,
 typename ASmemLayout,
 typename TiledCopyA,
 typename S2RAtomA,
 typename ElementB,
 typename StrideB,
 typename BSmemLayout,
 typename TiledCopyB,
 typename S2RAtomB,
 typename ElementC,
 typename StrideC,
 typename Mma
>
__global__ void GemmDevice(Params<ProblemShape, ElementA, StrideA, ElementB, StrideB, ElementC, StrideC> params) {

  using namespace cute;

  using SharedStorage = SharedStorage<ElementA, ElementB, ASmemLayout, BSmemLayout>;

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

  TileShape cta_tiler;

  ASmemLayout sA_layout;
  TiledCopyA copy_a;

  BSmemLayout sB_layout;
  TiledCopyB copy_b;

  S2RAtomA s2r_copy_atom_a;
  S2RAtomB s2r_copy_atom_b;

  Mma tiled_mma;

  //// Step 1. make global tensor
  static_assert(is_static_v<TileShape>, "");

  Tensor mA = make_tensor(make_gmem_ptr(params.ptr_A), select<0, 2>(params.problem_size), params.dA);
  Tensor mB = make_tensor(make_gmem_ptr(params.ptr_B), select<1, 2>(params.problem_size), params.dB);
  Tensor mC = make_tensor(make_gmem_ptr(params.ptr_C), select<0, 1>(params.problem_size), params.dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

  ///< tiler tensor for cta
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  ///< make tensor for shared memory
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)


  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)


  ///< S2R copy
  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_copy_atom_a, tiled_mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_copy_atom_b, tiled_mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

  // Size of the register pipeline
  auto k_tile_count = size<3>(tAgA);
  auto K_PIPE_MAX = size<3>(tAsA);

  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = 0;

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);

  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
    ++smem_pipe_write;
  }

  // Clear the accumulators
  clear(tCrC);

  // Pipe slice
  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_copy_a, tXsA(_,_,Int<0>{}, smem_pipe_read), tXrA(_,_,Int<0>{}));
    copy(s2r_copy_b, tXsB(_,_,Int<0>{}, smem_pipe_read), tXrB(_,_,Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
  //           and explicit pipelines in shared memory.
  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
  //   Data is computed on registers(b_block).
  //
  //   This allows all copies and compute to overlap:
  //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
  //     Copy from smem->rmem can overlap with compute on rmem.
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        // tXsA_p = tXsA(_,_,_,smem_pipe_read);
        // tXsB_p = tXsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();

        ++smem_pipe_write;
        ++smem_pipe_read;

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_write == K_PIPE_MAX ? 0 : smem_pipe_write;
        smem_pipe_read = smem_pipe_read == K_PIPE_MAX ? 0 : smem_pipe_read;
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
      copy(s2r_copy_a, tXsA(_,_,k_block_next, smem_pipe_read), tXrA(_,_,k_block_next));
      copy(s2r_copy_b, tXsB(_,_,k_block_next, smem_pipe_read), tXrB(_,_,k_block_next));

      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // // Advance the smem pipe
        // smem_pipe_write = smem_pipe_read;
        // smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }

      // Thread-level register gemm for k_block
      gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

  }

  // if (thread0()) {

  //   for (int i = 0; i < 128; ++i) {
  //     for (int j = 0; j < 64; ++j) {
  //       print("%10.2f ", float(*(smem.A.begin() + i * 64 + j)));
  //     }
  //     print("\n");
  //   }

  // }

  axpby(1.0, tCrC, 0.0, tCgC);

}


template<
 typename ElementA,
 typename ElementB,
 typename ElementC
>
void LaunchGemmKernel(const ElementA *A, const ElementB * B, ElementC * C, int m, int n, int k, int lda, int ldb, int ldc) {

  using namespace cute;
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);

  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(lda, _1{});
  auto dB = make_stride(ldb, _1{});
  auto dC = make_stride(ldc, _1{});

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int< 64>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,Shape <_8, _8>>,
                                         Stride<_8,Stride<_1,_64>>>{});

  // auto swizzle_atom = composition(Swizzle<3,3,3>{},
  //                                 Layout<Shape <_8,Shape <_8, _8>>,
  //                                        Stride<_64,Stride<_1,_8>>>{});

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));

  print_layout(swizzle_atom);

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 n-major

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{},
                                 Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM

  //Copy_Atom<DefaultCopy, half_t> s2r_atom_A;
  //Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_A;
  Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_A;
  //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_A;
  // Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;

  //Copy_Atom<DefaultCopy, half_t> s2r_atom_B;
  //Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_B;
  Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_B;
  //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_B;
  // Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;


#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  auto kernel_fptr = GemmDevice<
    decltype(prob_shape), decltype(cta_tiler),
    cute::half_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
    cute::half_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
    cute::half_t, decltype(dC), decltype(mmaC)>;

  // Set L1 to be SMEM only
  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size>>>
      ({prob_shape, A, dA, B, dB, C, dC});


}

using ElementA = cute::half_t;
using ElementB = cute::half_t;
using ElementC = cute::half_t;
using ElementAccumulator = cute::half_t;

using HostKernel = reference::Gemm<ElementA, cutlass::layout::RowMajor,
                                   ElementB, cutlass::layout::ColumnMajor,
                                   ElementC, cutlass::layout::RowMajor,
                                   ElementAccumulator, ElementAccumulator>;

void host_gemm(const ElementA *ptr_A, const ElementB *ptr_B,
               const ElementC *ptr_C, ElementC *ptr_D, int m, int n, int k) {
  HostKernel host_op;
  host_op(ptr_A, ptr_B, ptr_C, ptr_D, m, n, k);
}

int main() {

  int M = 128;
  int N =  64;
  int K = 64;

  ElementA *h_A = new ElementA[M * K];
  ElementB *h_B = new ElementB[N * K];
  ElementC *h_D = new ElementC[M * N];

  ElementC *result_D = new ElementC[M * N];

  reference::sequence_initializer<ElementB>::init(h_A, M * K);
  // reference::random_initializer<ElementA>::init(h_A, M * K);
  // reference::random_initializer<ElementB>::init(h_B, N * K);
  reference::diagonal_initializer<ElementB>::init(h_B, N);

  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < K; ++j) {
  //     std::cout << h_A[i * K + j] << ",";
  //   }
  //   std::cout << std::endl;
  // }


  ElementA *d_A;
  ElementB *d_B;
  ElementC *d_D;


  cudaMalloc(&d_A, M * K * sizeof(ElementA));
  cudaMalloc(&d_B, N * K * sizeof(ElementB));
  cudaMalloc(&d_D, M * N * sizeof(ElementC));

  cudaMemcpy(d_A, h_A, M * K * sizeof(ElementA), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * K * sizeof(ElementB), cudaMemcpyHostToDevice);

  LaunchGemmKernel<ElementA, ElementB, ElementC>(d_A, d_B, d_D, M, N, K, K, K, N);

  // for (int i = 0; i < 20; i++)

  cudaMemcpy(result_D, d_D, M * N * sizeof(ElementC), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

#ifdef HOST_CHECK
  host_gemm(h_A, h_B, nullptr, h_D, M, N, K);

  for (int i = 0; i < M * N; i++) {
    float abs_err = fabs(float(h_D[i]) - float(result_D[i]));
    if (abs_err > 1e-2) {
      std::cout <<"i: " << i << " cpu: " << float(h_D[i]) << "\tgpu: " << float(result_D[i]) << "\tdiff: " << abs_err << std::endl;
    }
  }
#endif

  delete[] h_A;
  delete[] h_B;
  delete[] h_D;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);


  return 0;
}