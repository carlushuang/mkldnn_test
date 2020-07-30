#!/bin/sh
DNNLROOT=/opt/intel/inteloneapi/oneDNN/latest/cpu_iomp/
# DNNLROOT=/opt/intel/inteloneapi/oneDNN/latest/cpu_gomp/
CXXFLAGS=" -std=c++11 -I${DNNLROOT}/include  "
LDFLAGS="-L${DNNLROOT}/lib -ldnnl "
SRC=conv.cpp
TARGET=conv.exe

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET
