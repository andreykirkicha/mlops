cmake_minimum_required(VERSION 3.26)
project(tensor_ops LANGUAGES CXX)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# TODO 3: build _core from cpp/bindings.cpp and cpp/kernel.cpp using C++17.
# Install the compiled target into tensor_ops inside the wheel.
# See ../example/CMakeLists.txt for a scalar example.
message(FATAL_ERROR "TODO 3: add the extension target and its install rule")
