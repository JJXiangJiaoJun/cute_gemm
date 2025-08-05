# cute_gemm

# Environment

CUDA 11.4
CUTLASS v4.0.0

# Quick Start

```shell
git clone git@github.com:JJXiangJiaoJun/cute_gemm.git
cd cute_gemm
git clone git@github.com:NVIDIA/cutlass.git
cd cutlass && git checkout v4.0.0 && cd ..
```

## cute_transpose_swizzle
```shell
nvcc --std=c++17 -arch=sm_86 --expt-relaxed-constexpr -O2 -I ./ -I ./cutlass/include -DHOST_CHECK cute_transpose_swizzle.cu.cu -o cute_transpose_swizzle

```

