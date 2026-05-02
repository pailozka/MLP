cmake_minimum_required(VERSION 3.21)
project(pv021_project)

set(CMAKE_CXX_STANDARD 14)

add_executable(pv021_project
        src/NeuralNetwork.cpp
        src/activation.cpp
        src/data.cpp)
