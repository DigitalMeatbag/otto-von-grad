cmake_minimum_required(VERSION 3.20)
project(otto_von_grad C)

set(CMAKE_C_STANDARD 11)

# Flatten all build outputs into the binary dir (no Release/ or Debug/ subdirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY         "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}")

option(OVG_CUDA "Enable CUDA acceleration (requires CUDA Toolkit)" OFF)

# ── Layered targets ──────────────────────────────────────────────────────────
# Dependencies point strictly downward: lm → nn → core. A source file belongs
# to the lowest layer that has everything it needs; nothing in core or nn may
# include an lm header. `ottovongrad` is the umbrella consumers link today.

set(OVG_CORE_SOURCES
    src/tg_tensor.c
    src/tg_ops.c
    src/tg_train.c
    src/tg_rng.c
    src/tg_checkpoint.c
    src/ovg_error.c
)

set(OVG_NN_SOURCES
    src/tg_mlp.c
    src/tg_attention.c
    src/tg_block.c
    src/tg_transformer.c
)

set(OVG_LM_SOURCES
    src/tg_gpt.c
    src/tg_tokenizer.c
    src/tg_sample.c
)

if(OVG_CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    # VS 18 (2026) is newer than CUDA 12.8's supported range; override the check.
    # The generated code is correct — MSVC ABI is stable across these versions.
    # These must be set before enable_language(CUDA) so they apply during
    # compiler identification, not just compilation.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND OVG_CORE_SOURCES src/tg_cuda.cu src/cuda_ops.cu)
endif()

# core: tensor, autograd ops, optimizers, RNG, checkpoint I/O, error handling, CUDA
add_library(ovg_core STATIC ${OVG_CORE_SOURCES})
target_include_directories(ovg_core
    PUBLIC  include/ovg/
    PRIVATE src/)

if(OVG_CUDA)
    target_compile_definitions(ovg_core PUBLIC OVG_CUDA_ENABLED)
    target_link_libraries(ovg_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(NOT MSVC)
    target_compile_options(ovg_core PRIVATE -O3 -march=native -ffast-math)
    target_link_libraries(ovg_core PUBLIC m)
endif()

# nn: model-agnostic building blocks (linear, attention, transformer block/stack)
add_library(ovg_nn STATIC ${OVG_NN_SOURCES})
target_link_libraries(ovg_nn PUBLIC ovg_core)
if(NOT MSVC)
    target_compile_options(ovg_nn PRIVATE -O3 -march=native -ffast-math)
endif()

# lm: language-model toolkit (GPT, byte tokenizer, sampling)
add_library(ovg_lm STATIC ${OVG_LM_SOURCES})
target_link_libraries(ovg_lm PUBLIC ovg_nn)
if(NOT MSVC)
    target_compile_options(ovg_lm PRIVATE -O3 -march=native -ffast-math)
endif()

# umbrella: everything, for consumers that link `ottovongrad`
add_library(ottovongrad INTERFACE)
target_link_libraries(ottovongrad INTERFACE ovg_lm)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(otto_von_grad src/main.c)
    target_link_libraries(otto_von_grad PRIVATE ovg_lm)

    add_executable(otto_von_grad_tests
        tests/test_main.c
        tests/test_ops.c
        tests/test_train.c
        tests/test_attention.c
        tests/test_gpt.c
        tests/test_tokenizer.c
        tests/test_checkpoint.c
        tests/test_sample.c
    )
    target_link_libraries(otto_von_grad_tests PRIVATE ovg_lm)
    target_include_directories(otto_von_grad_tests PRIVATE tests/)

    enable_testing()
    add_test(NAME ovg_tests COMMAND otto_von_grad_tests)
endif()
