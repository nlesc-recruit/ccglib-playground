cmake_minimum_required(VERSION 3.20)

project(example)

option(USE_HIP "Use HIP backend instead of CUDA" False)

if(${USE_HIP})
  enable_language(HIP)
  set(CUDAWRAPPERS_BACKEND "HIP")
  set(CCGLIB_BACKEND "HIP")
else()
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
endif()

include(FetchContent)

FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG "main")
FetchContent_MakeAvailable(cudawrappers)

if(NOT DEFINED CCGLIB_GIT_REVISION)
  set(CCGLIB_GIT_REVISION "origin/main")
endif()

FetchContent_Declare(
  ccglib
  GIT_REPOSITORY https://github.com/nlesc-recruit/ccglib
  GIT_TAG ${CCGLIB_GIT_REVISION})
FetchContent_MakeAvailable(ccglib)

add_executable(complex_interleaved src/complex_interleaved.cpp)
target_link_libraries(complex_interleaved PUBLIC ccglib cudawrappers::cu
                                                 cudawrappers::nvrtc)
target_embed_source(complex_interleaved kernels/kernel_complex_interleaved.cu)
target_include_directories(complex_interleaved
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/kernels)

if(${USE_HIP})
  set_source_files_properties(src/complex_interleaved.cpp PROPERTIES LANGUAGE
                                                                     HIP)
endif()
