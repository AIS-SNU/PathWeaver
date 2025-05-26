#!/bin/bash

# Abort script on first error
set -e

#============================== CAGRA =======================================
BIN_DIR=bin/cagra
mkdir -p $BIN_DIR

# Move to source directory
cd ./csrc/cagra_graph
rm -rf build

# Compile the CAGRA C++ code
export CUDACXX=/usr/local/cuda-12.1/bin/nvcc

PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

BUILD_TYPE=Release
BUILD_DIR=build/

RAFT_REPO_REL=""
EXTRA_CMAKE_ARGS=""

if [[ ${RAFT_REPO_REL} != "" ]]; then
  RAFT_REPO_PATH="`readlink -f \"${RAFT_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_raft_SOURCE=${RAFT_REPO_PATH}"
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DRAFT_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 ${EXTRA_CMAKE_ARGS} \
 ../

cmake  --build . -j${PARALLEL_LEVEL}

cd ../

# Copy the compiled library to the parent directory
cp ./build/cagra_wrapper.* ../../$BIN_DIR



#============================== PATHWEAVER ==================================
BIN_DIR=bin/pathweaver
mkdir -p ./$BIN_DIR

python setup.py build_ext --inplace
mv ./*.so ./$BIN_DIR