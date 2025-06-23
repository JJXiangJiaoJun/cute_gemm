#pragma once

#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/activation.h"

namespace reference {

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementOutput,
  typename LayoutOutput,
  typename ElementAccumulate = float,
  typename ElementCompute = ElementAccumulate
>
struct Gemm;

template <
  typename ElementA_,
  typename ElementB_,
  typename ElementOutput_,
  typename ElementAccumulate_,
  typename ElementCompute_
>
struct Gemm<
 ElementA_,
 cutlass::layout::RowMajor,
 ElementB_,
 cutlass::layout::ColumnMajor,
 ElementOutput_,
 cutlass::layout::RowMajor,
 ElementAccumulate_,
 ElementCompute_
> {
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementOutput = ElementOutput_;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulate = ElementAccumulate_;
  using ElementCompute = ElementCompute_;

  void operator()(const ElementA *A, const ElementB *B, const ElementOutput *C,
                  ElementOutput *D, int M, int N, int K) {
    for (int m_i = 0; m_i < M; ++m_i) {
      for (int n_i = 0; n_i < N; ++n_i) {

        ElementAccumulate tmp =
            (C == nullptr ? ElementAccumulate(0)
                          : (static_cast<ElementAccumulate>(C[n_i])));

        for (int k_i = 0; k_i < K; ++k_i) {
          tmp += ElementAccumulate(A[m_i * K + k_i]) *
                 ElementAccumulate(B[n_i * K + k_i]);
        }
        ElementCompute res = static_cast<ElementCompute>(tmp);
        D[m_i * N + n_i] = static_cast<ElementOutput>(res);
      }
    }
  }
};

} /// end of namespace reference