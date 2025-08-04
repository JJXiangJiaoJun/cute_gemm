#include <cstdio>
#include <iostream>

#include "cutlass/kernel_launch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cute/tensor.hpp"

#include "reference/gemm.h"
#include "reference/initializer.h"

#define PRINT_HOST_DEVICE(STR)                                                 \
  do {                                                                         \
    printf("threadIdx: %d, %s:\n", threadIdx.x, #STR);                         \
    cute::print(STR);                                                          \
    printf("\n\n");                                                            \
  } while (0)


#define PRINT_DEVICE(STR)                                                      \
  do {                                                                         \
    printf("threadIdx: %d, %s:\n", threadIdx.x, #STR);                         \
    cute::print(STR);                                                          \
    printf("\n\n");                                                            \
  } while (0)

#define PRINT_TENSOR_DEVICE(STR)                                               \
  do {                                                                         \
    printf("threadIdx: %d, %s:\n", threadIdx.x, #STR);                         \
    cute::print_tensor(STR);                                                   \
    printf("\n\n");                                                            \
  } while (0)

using namespace cute;

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  typename ElementSource_ = ElementOutput_
>
class LinearCombination {
public:

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;

  using InputConverter = cutlass::NumericConverter<ElementCompute, ElementAccumulator>;
  using SourceConverter = cutlass::NumericConverter<ElementSource, ElementAccumulator>;
  using OutputConverter = cutlass::NumericConverter<ElementOutput, ElementCompute>;

  CUTLASS_HOST_DEVICE
  LinearCombination() {}

  CUTLASS_HOST_DEVICE
  void operator()(ElementOutput &output, ElementAccumulator const &acc, ElementSource const &source) {

    InputConverter  input_converter;
    SourceConverter source_converter;
    OutputConverter output_converter;

    output = output_converter(input_converter(acc) + source_converter(source));

  }


  CUTLASS_HOST_DEVICE
  void operator()(ElementOutput &output, ElementAccumulator const &acc) {

    OutputConverter output_converter;
    output = output_converter(acc);

  }

};




namespace utils {

template<typename Operator>
void __global__ Kernel(typename Operator::Params params) {
  extern __shared__ char smem_buf[];
  Operator op;
  op(params, smem_buf);
}

}

template<
 typename ProblemShape_,
 typename TileShape_,
 typename ElementA_,
 typename StrideA_,
 typename ElementB_,
 typename StrideB_,
 typename ElementC_,
 typename StrideC_,
 typename ElementD_,
 typename StrideD_,
 typename TiledMma_,
 typename GmemTiledCopyA_,
 typename SmemLayoutAtomA_,
 typename SmemCopyAtomA_,
 typename GmemTiledCopyB_,
 typename SmemLayoutAtomB_,
 typename SmemCopyAtomB_,
 int Stages,
 typename EpilogueOutputOp_,
 typename CopyAtomR2S_,
 typename SmemLayoutEpilogue_,
 typename TiledCopyS2R_,
 typename GmemTiledCopyC_,
 typename GmemTiledCopyD_
>
class Gemm {

public:
  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static const int kStages = Stages;

  using TileShape = TileShape_;
  using ElementA  = ElementA_;
  using StrideA   = StrideA_;
  using ElementB  = ElementB_;
  using StrideB   = StrideB_;
  using ElementC  = ElementC_;
  using StrideC   = StrideC_;
  using ElementD  = ElementD_;
  using StrideD   = StrideD_;

  using TiledMma = TiledMma_;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemCopyAtomA = SmemCopyAtomA_;

  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomB = SmemCopyAtomB_;

  ///< Epilogue
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;
  using CopyAtomR2S = CopyAtomR2S_;
  using SmemLayoutEpilogue = SmemLayoutEpilogue_;
  using TiledCopyS2R = TiledCopyS2R_;
  using GmemTiledCopyC = GmemTiledCopyC_;
  using GmemTiledCopyD = GmemTiledCopyD_;

  /// MxKxkStage
  using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{})));


  /// NxKxkStage
  using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{})));

  struct MainloopSharedStorage {
    cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  struct EpilogueSharedStorage {
    cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutEpilogue>> s_acc;
  };

  union SharedStorage {
    MainloopSharedStorage mainloop_smem;
    EpilogueSharedStorage epilogue_smem;
  };

  // Host side kernel arguments
  struct MainloopArguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  struct EpilogueArguments {
    ElementC const *ptr_C;
    StrideC dC;
    ElementD * ptr_D;
    StrideD dD;
  };

  struct Arguments {
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
  };

  using Params = Arguments;

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(cute::size(TiledMma{}));


  static Params
  to_underlying_params(Arguments const& args) {
    return Params{args.problem_shape, args.mainloop, args.epilogue};
  }

  static dim3
  get_grid_shape(Params const& params) {
    int batch_count = 1;
    if constexpr (cute::rank(ProblemShape{}) == 4) {
      batch_count = cute::size<3>(params.problem_shape);
    }

    return dim3(
      cute::size(cute::ceil_div(cute::shape<0>(params.problem_shape), cute::shape<0>(TileShape{}))),
      cute::size(cute::ceil_div(cute::shape<1>(params.problem_shape), cute::shape<1>(TileShape{}))),
      batch_count
    );
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }


  CUTLASS_DEVICE
  void operator()(Params &params, char *smem_buf) {

    SharedStorage &smem = *reinterpret_cast<SharedStorage *>(smem_buf);

    int thread_idx = int(threadIdx.x);
    auto block_coord_mnkl = make_coord(blockIdx.x, blockIdx.y, _, blockIdx.z);

    auto block_shape = TileShape{};

    auto M = size<0>(params.problem_shape);
    auto N = size<1>(params.problem_shape);
    auto K = size<2>(params.problem_shape);
    auto L = size<3>(params.problem_shape);

    ///<  Create global tensor
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M, K, L), params.mainloop.dA);
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N, K, L), params.mainloop.dB);

    ///< Batch slice
    Tensor mA_mk = mA_mkl(_, _, get<3>(block_coord_mnkl));
    Tensor mB_nk = mB_nkl(_, _, get<3>(block_coord_mnkl));

    ///< Slice to get the tiles this thread block is responsible for
    Tensor gA = local_tile(mA_mk, block_shape, take<0,3>(block_coord_mnkl), Step<_1, X,_1>{});           // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_nk, block_shape, take<0,3>(block_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)

    // if (thread0()) {
    //   PRINT_TENSOR_DEVICE(gA);
    //   PRINT_TENSOR_DEVICE(gB);
    // }

    ///< Create smem tensor
    Tensor sA = make_tensor(make_smem_ptr(smem.mainloop_smem.smem_a.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.mainloop_smem.smem_b.begin()), SmemLayoutB{});

    ///< Global to Shared
    GmemTiledCopyA gmem_tiled_copy_A;
    ThrCopy gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);     // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);     // (CPY,CPY_M,CPY_K,PIPE)

    GmemTiledCopyB gmem_tiled_copy_B;
    ThrCopy gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

    ///< TiledMma
    TiledMma tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);

    ///< compute thread partition and allocate accumulators
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); ///< (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

    Tensor accumulator = partition_fragment_C(tiled_mma, take<0, 2>(block_shape));

    ///< Shared to register
    ///< CopyAtom retiling
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A  = smem_tiled_copy_A.get_slice(thread_idx);
    auto tCsA = smem_thr_copy_A.partition_S(sA);        /// (CPY, CPY_M, CPY_K, PIPE)
    auto tCrA_view = smem_thr_copy_A.retile_D(tCrA);    /// (CPY, CPY_M, CPY_K)

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B  = smem_tiled_copy_B.get_slice(thread_idx);
    auto tCsB = smem_thr_copy_B.partition_S(sB);        /// (CPY, CPY_N, CPY_K, PIPE)
    auto tCrB_view = smem_thr_copy_B.retile_D(tCrB);    /// (CPY, CPY_N, CPY_K)


    int k_tile_count = size<2>(gA);
    int k_tile_iter  = 0;

    // Size of the register pipeline
    int K_BLOCK_MAX = size<2>(tCrA);

    int smem_pipe_read  = 0;
    int smem_pipe_write = 0;

    ///< prologue
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < kStages -1; ++stage) {
      copy(gmem_tiled_copy_A, tAgA(_, _, _, k_tile_iter), tAsA(_, _, _, smem_pipe_write));
      copy(gmem_tiled_copy_B, tBgB(_, _, _, k_tile_iter), tBsB(_, _, _, smem_pipe_write));
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) {++k_tile_iter;}
      ++smem_pipe_write;
    }

    // Clear the accumulators
    clear(accumulator);

    if (K_BLOCK_MAX > 1) {
      cp_async_wait<kStages - 2>();
      __syncthreads();
      copy(smem_tiled_copy_A, tCsA(_, _, 0, smem_pipe_read), tCrA_view(_, _, 0));
      copy(smem_tiled_copy_B, tCsB(_, _, 0, smem_pipe_read), tCrB_view(_, _, 0));
    }

    ///< mainloop k iter
    while (k_tile_count > -(kStages -1)) {

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {

        if (k_block == K_BLOCK_MAX - 1) {
          ///< wait next global load stage complete and advance read/write stage
          cp_async_wait<kStages - 2>();
          __syncthreads();

          ++smem_pipe_write;
          ++smem_pipe_read;

          // Advance the smem pipe
          smem_pipe_write = smem_pipe_write == kStages ? 0 : smem_pipe_write;
          smem_pipe_read = smem_pipe_read == kStages ? 0 : smem_pipe_read;
        }

        ///< s2r pipline
        int k_block_next = (k_block + 1) % K_BLOCK_MAX;
        copy(smem_tiled_copy_A, tCsA(_, _, k_block_next, smem_pipe_read), tCrA_view(_, _, k_block_next));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block_next, smem_pipe_read), tCrB_view(_, _, k_block_next));

        if (k_block == 0) {
          ///< emit next stage global load
          copy(gmem_tiled_copy_A, tAgA(_, _, _, k_tile_iter), tAsA(_, _, _, smem_pipe_write));
          copy(gmem_tiled_copy_B, tBgB(_, _, _, k_tile_iter), tBsB(_, _, _, smem_pipe_write));
          cp_async_fence();

          --k_tile_count;
          if (k_tile_count > 0) {++k_tile_iter;}

        }

        ///< mma
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accumulator);

      }
    }


    cp_async_wait<0>();
    __syncthreads();

    ///< Epilogue perform activation (acc + bias)
    Tensor sAcc = make_tensor(make_smem_ptr(smem.epilogue_smem.s_acc.begin()), SmemLayoutEpilogue{});

    auto r2s_tiled_copy_Acc = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto r2s_thr_copy_Acc   = r2s_tiled_copy_Acc.get_slice(thread_idx);
    Tensor tRS_rAcc = r2s_thr_copy_Acc.retile_S(accumulator);
    Tensor tRS_sAcc = r2s_thr_copy_Acc.partition_D(sAcc);

    copy(r2s_tiled_copy_Acc, tRS_rAcc, tRS_sAcc);
    __syncthreads();

    TiledCopyS2R s2r_tiled_copy_Acc;
    auto s2r_thr_copy_acc = s2r_tiled_copy_Acc.get_slice(thread_idx);
    Tensor tSR_sAcc = s2r_thr_copy_acc.partition_S(sAcc);
    Tensor tSR_rAcc = make_fragment_like(tSR_sAcc);

    copy(s2r_tiled_copy_Acc, tSR_sAcc, tSR_rAcc);


    ///< Load source
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.epilogue.ptr_C), make_shape(M, N, L), params.epilogue.dC);
    Tensor mC_mn  = mC_mnl(_, _, get<3>(block_coord_mnkl));

    Tensor gC = local_tile(mC_mn, block_shape, take<0,3>(block_coord_mnkl), Step<_1, _1, X>{}); ///< (BLK_M, BLK_N)

    GmemTiledCopyC gmem_tiled_copy_C;
    auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(thread_idx);
    Tensor tCgC = gmem_thr_copy_C.partition_S(gC); ///< (CPY, CPY_M, CPY_N)
    Tensor tCrC = make_fragment_like(tCgC);

    copy(gmem_tiled_copy_C, tCgC, tCrC);

    ///< Partition output
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.epilogue.ptr_D), make_shape(M, N, L), params.epilogue.dD);
    Tensor mD_mn  = mD_mnl(_, _, get<3>(block_coord_mnkl));

    Tensor gD = local_tile(mD_mn, block_shape, take<0, 3>(block_coord_mnkl), Step<_1, _1, X>{}); ///< (BLK_M, BLK_N)

    GmemTiledCopyD gmem_tiled_copy_D;
    auto gmem_thr_copy_D = gmem_tiled_copy_D.get_slice(thread_idx);
    Tensor tDgD = gmem_thr_copy_D.partition_D(gD);
    Tensor tDrD = make_fragment_like(tDgD);

    EpilogueOutputOp output_op;

    ///< Elementwise operation with conversion
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tSR_rAcc); ++i) {
      output_op(tDrD(i), tSR_rAcc(i), tCrC(i));
    }

    copy(gmem_tiled_copy_D, tDrD, tDgD);
  }

};


template<typename GemmKernel_>
struct GemmUniversalAdapter {
public:
  using GemmKernel = GemmKernel_;
  using Arguments  = typename GemmKernel::Arguments;
  using Params     = typename GemmKernel::Params;

public:

  static dim3
  get_grid_shape(Params const& params) {
    return GemmKernel::get_grid_shape(params);
  }

  static dim3
  get_block_shape() {
    return GemmKernel::get_block_shape();
  }

private:
  Params params_;

public:
  void initialize(Arguments const &args) {
    params_ = GemmKernel::to_underlying_params(args);
  }

  void run(cudaStream_t stream = nullptr) {
    dim3 const block = get_block_shape();
    dim3 const grid  =  get_grid_shape(params_);

    int smem_size = sizeof(typename GemmKernel::SharedStorage);

    // first, account for dynamic smem capacity if needed
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      std::cout << "  Setting smem size to " << smem_size << std::endl;
      result = cudaFuncSetAttribute(
          utils::Kernel<GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        std::cout <<
          "  cudaFuncSetAttribute() returned error: "
          << cudaGetErrorString(result) << std::endl;
        exit(-1);
      }
    }

    utils::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
      std::cout << "Cuda Error: "
        << "grid: " << "("  << grid.x << ", " << grid.y << ", " << grid.z << ")\n"
        << "block: " << "("  << block.x << ", " << block.y << ", " << block.z << ")\n"
        << "smem-size: " <<  smem_size << "\n"
        << cudaGetErrorString(result) << std::endl;
      exit(-1);
    }
  }

};



template <class IntT>
using KMajorStride = cute::Stride<IntT, cute::Int<1>, int64_t>;

template <class IntT>
using RowMajorStrideCD = KMajorStride<IntT>;

template <class IntT>
using MNMajorStride = cute::Stride<cute::Int<1>, IntT, int64_t>;

template <class IntT>
CUTLASS_HOST_DEVICE
KMajorStride<IntT>
make_cute_stride(KMajorStride<IntT> s, int leading_dimension, int batch_stride) {
  auto s_copy = s;
  cute::get<0>(s_copy) = leading_dimension;
  cute::get<2>(s_copy) = batch_stride;
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE
MNMajorStride<IntT>
make_cute_stride(MNMajorStride<IntT> s, int leading_dimension, int batch_stride) {
  auto s_copy = s;
  cute::get<1>(s_copy) = leading_dimension;
  cute::get<2>(s_copy) = batch_stride;
  return s_copy;
}



using ProblemShape = cute::tuple<int, int, int, int>;
using TileShape = cute::Shape<_128, _128, _64>;
using ElementA = cutlass::half_t;
using StrideA  = KMajorStride<int>;
using ElementB = cutlass::half_t;
using StrideB  = KMajorStride<int>;
using ElementC = cutlass::half_t;
using StrideC  = RowMajorStrideCD<int>;
using ElementD = cutlass::half_t;
using StrideD  = RowMajorStrideCD<int>;

using TiledMma = decltype(
 make_tiled_mma(
  SM80_16x8x16_F16F16F16F16_TN{},
  Layout<Shape<_2, _2>>{},     // 2x2x1 MMA Atoms
  Tile<_32, _32, _16>{}        // 32x32x16 Tiled MMA for LDSM
 )
);

using GmemTiledCopyA = decltype(
 make_tiled_copy(
  Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
  Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major,
  Layout<Shape< _1,_8>>{}
 )
);

using GmemTiledCopyB = decltype(
 make_tiled_copy(
  Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
  Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major,
  Layout<Shape< _1,_8>>{}
 )
);

using SmemLayoutAtomA = decltype(
composition(
  Swizzle<3,3,3>{},
  Layout<Shape <_8,Shape <_8, _8>>,
  Stride<_64,Stride<_1,_8>>>{}
 )
);

using SmemLayoutAtomB = decltype(
composition(
  Swizzle<3,3,3>{},
  Layout<Shape <_8,Shape <_8, _8>>,
  Stride<_64,Stride<_1,_8>>>{}
 )
);

using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;

using ElementAccumulator = cute::half_t;
using ElementCompute     = ElementAccumulator;

using EpilogueOutputOp = LinearCombination<ElementD, ElementAccumulator, ElementCompute, ElementC>;

using CopyAtomR2S = Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<cute::sizeof_bits_v<ElementAccumulator> * 2>, ElementAccumulator>;
using SmemLayoutEpilogue = decltype(
  make_layout(make_shape(size<0>(TileShape{}), size<1>(TileShape{})), make_stride(size<1>(TileShape{}), _1{}))
);

using TiledCopyS2R = decltype(
 make_tiled_copy(
  Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccumulator>{},
  Layout<Shape<_8,_16>,Stride<_16,_1>>{},  // Thr layout 16x8 k-major,
  Layout<Shape< _1,_8>>{}
 )
);

using GmemTiledCopyC = decltype(
 make_tiled_copy(
  Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccumulator>{},
  Layout<Shape<_8,_16>,Stride<_16,_1>>{},  // Thr layout 16x8 k-major,
  Layout<Shape< _1,_8>>{}
 )
);

using GmemTiledCopyD = decltype(
 make_tiled_copy(
  Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccumulator>{},
  Layout<Shape<_8,_16>,Stride<_16,_1>>{},  // Thr layout 16x8 k-major,
  Layout<Shape< _1,_8>>{}
 )
);

// using R2GmemTiledCopyAtomD = Copy_Atom<UniversalCopy<ElementC>, ElementC>;


static const int kStages = 3;

using GemmKernel =
  Gemm<
 ProblemShape,
 TileShape,
 ElementA,
 StrideA,
 ElementB,
 StrideB,
 ElementC,
 StrideC,
 ElementD,
 StrideD,
 TiledMma,
 GmemTiledCopyA,
 SmemLayoutAtomA,
 SmemCopyAtomA,
 GmemTiledCopyB,
 SmemLayoutAtomB,
 SmemCopyAtomB,
 kStages,
 EpilogueOutputOp,
 CopyAtomR2S,
 SmemLayoutEpilogue,
 TiledCopyS2R,
 GmemTiledCopyC,
 GmemTiledCopyD
>;

using DeviceKernel =
 GemmUniversalAdapter<GemmKernel>;


void device_kernel(
  const ElementA *A,
  const ElementB *B,
  const ElementC *C,
  ElementD *D,
  int b, int m, int n, int k,
  int lda, int ldb, int ldc, int ldd,
  int batch_stride_A,
  int batch_stride_B,
  int batch_stride_C,
  int batch_stride_D
) {
  StrideA dA = make_cute_stride(StrideA{}, lda, batch_stride_A);
  StrideA dB = make_cute_stride(StrideB{}, ldb, batch_stride_B);
  StrideA dC = make_cute_stride(StrideC{}, ldc, batch_stride_C);
  StrideA dD = make_cute_stride(StrideD{}, ldd, batch_stride_D);

  DeviceKernel::Arguments arguments = {
   {m, n, k, b},
   {A, dA, B, dB},
   {C, dC, D, dD}
  };

  DeviceKernel op;
  op.initialize(arguments);
  op.run();
}


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

  int B = 1;
  int M = 1024;
  int N = 512;
  int K = 11008;

  int lda = K;
  int ldb = K;
  int ldc = 0;
  int ldd = N;

  int batch_stride_A = B > 1 ? M * K : 0;
  int batch_stride_B = B > 1 ? N * K : 0;
  int batch_stride_C = 0;
  int batch_stride_D = B > 1 ? M * N : 0;

  ElementA *h_A = new ElementA[M * K];
  ElementB *h_B = new ElementB[N * K];
  ElementC *h_C = new ElementC[N];
  ElementD *h_D = new ElementD[M * N];

  ElementD *result_D = new ElementD[M * N];

  // reference::sequence_initializer<ElementA>::init(h_A, M * K);
  // reference::diagonal_initializer<ElementB>::init(h_B, N);

  reference::random_initializer<ElementA>::init(h_A, M * K);
  reference::random_initializer<ElementB>::init(h_B, N * K);
  reference::random_initializer<ElementC>::init(h_C, N);

  ElementA *d_A;
  ElementB *d_B;
  ElementC *d_C;
  ElementD *d_D;


  cudaMalloc(&d_A, M * K * sizeof(ElementA));
  cudaMalloc(&d_B, N * K * sizeof(ElementB));
  cudaMalloc(&d_C, N * sizeof(ElementC));
  cudaMalloc(&d_D, M * N * sizeof(ElementD));

  cudaMemcpy(d_A, h_A, M * K * sizeof(ElementA), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * K * sizeof(ElementB), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, N * sizeof(ElementC), cudaMemcpyHostToDevice);

  device_kernel(d_A, d_B, d_C, d_D, B, M, N, K, lda, ldb, ldc, ldd, batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D);

  // for (int i = 0; i < 20; i++)

  cudaMemcpy(result_D, d_D, M * N * sizeof(ElementD), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

#ifdef HOST_CHECK
  host_gemm(h_A, h_B, h_C, h_D, M, N, K);

  for (int i = 0; i < M * N; i++) {
    float abs_err = fabs(float(h_D[i]) - float(result_D[i]));
    if (abs_err > 1e-2) {
      std::cout <<"i: " << i << " cpu: " << float(h_D[i]) << "\tgpu: " << float(result_D[i]) << "\tdiff: " << abs_err << std::endl;
    }
  }
#endif

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_D;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);


  return 0;
}