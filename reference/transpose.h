#pragma once

#include "cutlass/layout/matrix.h"

namespace reference {

template <
  typename ElementInput_,
  typename ElementOutput_ = ElementInput_
>
struct Transpose {
public:
 using ElementInput = ElementInput_;
 using ElementOutput = ElementOutput_;

 void operator()(const ElementInput *input, ElementOutput *output, int M, int N,
                 int ld_input, int ld_output) {
    for (int m_i  = 0; m_i < M; ++m_i) {
      for (int n_i = 0; n_i < N; ++n_i) {
        output[m_i + n_i * ld_output] = input[m_i * ld_input + n_i];
      }
    }
  }
};


} /// end of namespace reference