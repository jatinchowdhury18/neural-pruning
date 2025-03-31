cmake_minimum_required(VERSION 3.21)
project(neural_pruning)

set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CPM)
CPMAddPackage(
    NAME RTNeural
    GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural
    GIT_TAG 29e41da0b7cd2a5378669747b8d1c49b5d44d97e
    OPTIONS "RTNEURAL_EIGEN ON"
)
CPMAddPackage(
    NAME libsndfile
    GIT_REPOSITORY https://github.com/libsndfile/libsndfile
    GIT_TAG ea9ff560b4c2086c2f1cae3f02287768a0de4673
)

add_subdirectory(pruning_experiments)
add_subdirectory(plugin)
