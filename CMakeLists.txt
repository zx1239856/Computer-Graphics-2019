cmake_minimum_required(VERSION 3.08)
project(hw2)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(BUILD_CUDA)
    message("Build with cuda support")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE release)
set(EXEC_FILES main.cpp utils/vector3.hpp utils/bezier.hpp utils/object.hpp utils/ray.hpp utils/scene.hpp utils/camera.hpp utils/renderer.cpp)
add_executable(main ${EXEC_FILES})
