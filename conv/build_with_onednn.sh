#!/bin/sh
USE_HIP_NAIVE_CONV=1
GPU_ARCH="gfx908"

DNNLROOT=/opt/intel/inteloneapi/oneDNN/latest/cpu_iomp/
# DNNLROOT=/opt/intel/inteloneapi/oneDNN/latest/cpu_gomp/
CXXFLAGS=" -std=c++11 -I${DNNLROOT}/include -pthread"
if [ -n "$USE_HIP_NAIVE_CONV" ] ; then
CXXFLAGS="${CXXFLAGS} -DHIP_NAIVE_CONV -D__HIP_PLATFORM_HCC__= -I/opt/rocm/hip/include -I/opt/rocm/hcc/include -I/opt/rocm/hsa/include"
fi
LDFLAGS="-L${DNNLROOT}/lib -ldnnl -Wl,-rpath=${DNNLROOT}/lib"
if [ -n "$USE_HIP_NAIVE_CONV" ] ; then
LDFLAGS="${LDFLAGS} -L/opt/rocm/lib -L/opt/rocm/lib64 -Wl,-rpath=/opt/rocm/lib -ldl -lm -lpthread -Wl,--whole-archive -lamdhip64 -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"
fi
SRC=conv.cpp
TARGET=conv.exe

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET

if [ -n "$USE_HIP_NAIVE_CONV" ] ; then
/opt/rocm/llvm/bin/clang++ -x hip --hip-device-lib-path=/opt/rocm/lib \
        -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
        -D__HIP_ROCclr__=1 -isystem /opt/rocm/hip/../include -isystem /opt/rocm/llvm/lib/clang/11.0.0/include/.. \
        -D__HIP_PLATFORM_HCC__=1  -D__HIP_ROCclr__=1 -isystem /opt/rocm/hip/include -isystem /opt/rocm/include \
        --hip-device-lib-path=/opt/rocm/lib --hip-link  --std=c++11  --cuda-gpu-arch=$GPU_ARCH  --cuda-device-only -c -O3 \
        hip_naive_conv.cpp -o hip_naive_conv.hsaco
fi