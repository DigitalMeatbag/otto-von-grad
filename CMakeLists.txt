cmake_minimum_required(VERSION 3.20)
project(otto_von_grad C)

set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(OVG_CUDA "Enable CUDA acceleration (requires CUDA Toolkit)" OFF)

set(OVG_SOURCES
    src/tg_tensor.c
    src/tg_ops.c
    src/tg_train.c
    src/tg_mlp.c
    src/attention.c
    src/tg_block.c
    src/tg_transformer.c
    src/tg_gpt.c
)

if(OVG_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    # VS 18 (2026) is newer than CUDA 12.8's supported range; override the check.
    # The generated code is correct — MSVC ABI is stable across these versions.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    find_package(CUDAToolkit REQUIRED)
    list(APPEND OVG_SOURCES src/tg_cuda.cu src/cuda_ops.cu)
endif()

add_library(ottovongrad STATIC ${OVG_SOURCES})
target_include_directories(ottovongrad PUBLIC src/)

if(OVG_CUDA)
    target_compile_definitions(ottovongrad PUBLIC OVG_CUDA_ENABLED)
    target_link_libraries(ottovongrad PUBLIC CUDA::cudart)
endif()

if(NOT MSVC)
    target_link_libraries(ottovongrad PUBLIC m)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(otto_von_grad src/main.c src/smoke_tests.c)
    target_link_libraries(otto_von_grad PRIVATE ottovongrad)
endif()
