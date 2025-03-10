cmake_minimum_required(VERSION 3.21)
project(neural_pruning)

set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CPM)
CPMAddPackage(
    NAME RTNeural
    GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural
    GIT_TAG main
    OPTIONS "RTNEURAL_XSIMD ON"
)
CPMAddPackage(
    NAME libsndfile
    GIT_REPOSITORY https://github.com/libsndfile/libsndfile
    GIT_TAG master
)

add_subdirectory(proto)
add_subdirectory(plugin)
