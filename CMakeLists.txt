cmake_minimum_required(VERSION 3.25)

project(hydraulic-erosion)
set(CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
add_compile_definitions(CL_HPP_ENABLE_EXCEPTIONS)
add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=300)


set(PROJECT_SOURCE_FILES
    src/kernel_loader.cpp
)

set(PROJECT_HEADER_FILES
    src/kernel_loader.h
)

set(PROJECT_TEMPLATE_FILES
    src/kernel_loader.ipp
)

message(STATUS "${CMAKE_SYSTEM_LIBRARY_PATH}")

add_executable(hydraulic_erosion src/hydraulic_erosion.cpp ${PROJECT_SOURCE_FILES} ${PROJECT_HEADER_FILES} ${PROJECT_TEMPLATE_FILES})
target_include_directories(hydraulic_erosion PRIVATE vendor)
target_link_libraries(hydraulic_erosion OpenCL::OpenCL)