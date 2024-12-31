cmake_minimum_required(VERSION 3.21)
project(neural_pruning)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(../RTNeural RTNeural)

add_executable(neural_pruning_test neural_pruning_test.cpp)
target_link_libraries(neural_pruning_test PRIVATE RTNeural)
target_compile_definitions(neural_pruning_test PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
