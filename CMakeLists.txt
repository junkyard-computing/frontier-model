cmake_minimum_required(VERSION 3.20)
project(frontier-model CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Upstream llama.cpp checkout, built out-of-tree (never modified from here).
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp" CACHE PATH "llama.cpp source tree")

option(FM_LLAMA_TESTS "also build llama.cpp tests (test-backend-ops) for backend microbenchmarks" OFF)
set(LLAMA_BUILD_TESTS    ${FM_LLAMA_TESTS} CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
option(FM_LLAMA_TOOLS "also build llama.cpp tools (llama-bench) for baselines" OFF)
set(LLAMA_BUILD_TOOLS    ${FM_LLAMA_TOOLS} CACHE BOOL "" FORCE)
if (FM_LLAMA_TOOLS OR FM_LLAMA_TESTS OR FM_LLAMA_SERVER)
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)   # tools and tests need the common library
else()
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
endif()
option(FM_LLAMA_SERVER "also build llama-server (web UI + OpenAI API) on top of the estore patch" OFF)
set(LLAMA_BUILD_SERVER   ${FM_LLAMA_SERVER} CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)

add_executable(fm-trace tools/trace/trace.cpp)
target_link_libraries(fm-trace PRIVATE llama ggml)

# expert-streaming runtime (plain C, no llama.cpp dependency)
find_package(PkgConfig REQUIRED)
pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing)
add_library(estore STATIC runtime/estore.c)
target_include_directories(estore PUBLIC runtime)
target_link_libraries(estore PUBLIC PkgConfig::URING)
target_compile_options(estore PRIVATE -Wall -Wextra)

add_executable(fm-replay runtime/replay.c)
target_link_libraries(fm-replay PRIVATE estore)
target_compile_options(fm-replay PRIVATE -Wall -Wextra)

add_executable(fm-logits tools/logits/logits.cpp)
target_link_libraries(fm-logits PRIVATE llama)

add_executable(fm-gen tools/gen/gen.cpp)
target_link_libraries(fm-gen PRIVATE llama)
