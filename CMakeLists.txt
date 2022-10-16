cmake_minimum_required(VERSION 3.23)
project(NeuralNetworkLib)

set(CMAKE_CXX_STANDARD 23)

add_executable(NeuralNetworkLib main.cpp MatrixLib/MatrixLib.cpp BrainLib/BrainFart.cpp BrainLib/BrainFart.h MatrixLib/MatrixLib.h)
