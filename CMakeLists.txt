cmake_minimum_required(VERSION 3.0.0)

project(gpt_test)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Werror)
add_compile_options(-Wpedantic)
add_compile_options(-Wall)
add_compile_options(-Wextra)

add_executable(gpt_test test.cpp)

add_executable(makemore makemore.cpp)

target_include_directories(gpt_test PRIVATE ../hex/include)
target_include_directories(makemore PRIVATE ../hex/include)