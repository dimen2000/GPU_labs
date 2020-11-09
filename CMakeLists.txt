cmake_minimum_required(VERSION 2.8) 
PROJECT (GPU_labs)
find_package(OpenCL REQUIRED )
find_package(OpenMP REQUIRED )

include_directories  (${OpenCL_INCLUDE_DIRS})

add_subdirectory(modules)