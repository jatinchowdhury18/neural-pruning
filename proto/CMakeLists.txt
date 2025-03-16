add_executable(lstm_pruning_test lstm_pruning_test.cpp)
target_link_libraries(lstm_pruning_test PRIVATE RTNeural sndfile)
target_compile_definitions(lstm_pruning_test PRIVATE TRAIN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../train")

add_executable(dense_pruning_test dense_pruning_test.cpp)
target_link_libraries(dense_pruning_test PRIVATE RTNeural sndfile)
target_compile_definitions(dense_pruning_test PRIVATE TRAIN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../train")

add_executable(conv_pruning_test conv_pruning_test.cpp)
target_link_libraries(conv_pruning_test PRIVATE RTNeural sndfile)
target_compile_definitions(conv_pruning_test PRIVATE TRAIN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../train")
