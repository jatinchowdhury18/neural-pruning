add_executable(lstm_pruning_test lstm_pruning_test.cpp)
target_link_libraries(lstm_pruning_test PRIVATE RTNeural)
target_compile_definitions(lstm_pruning_test PRIVATE MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models")

add_executable(dense_pruning_test dense_pruning_test.cpp)
target_link_libraries(dense_pruning_test PRIVATE RTNeural sndfile)
target_compile_definitions(dense_pruning_test PRIVATE TRAIN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../train")
