cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(cagra)

project(cagra LANGUAGES CXX CUDA)

# include_directories(/nfs/home/daisy1212/anaconda3/envs/raft/include/raft/neighbors)
include_directories(/nfs/home/daisy1212/cagra-extended/local_build)

# ------------- configure raft -----------------#

rapids_cpm_init()

include(cmake/thirdparty/get_raft.cmake)

# -------------- compile tasks ----------------- #
find_package(pybind11 REQUIRED)

# pybind11 cagra_wrapper module
pybind11_add_module(cagra_wrapper cagra_wrapper.cu)
target_link_libraries(cagra_wrapper PRIVATE raft::raft raft::compiled)