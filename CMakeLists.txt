cmake_minimum_required(VERSION 3.25)

project(hydraulic-erosion)
set(CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

set(PROJECT_SOURCE_FILES
)

set(PROJECT_HEADER_FILES
)

message(STATUS "${CMAKE_SYSTEM_LIBRARY_PATH}")

add_executable(hydraulic_erosion src/hydraulic_erosion.cpp ${PROJECT_SOURCE_FILES} ${PROJECT_HEADER_FILES})
target_link_libraries(hydraulic_erosion OpenCL::OpenCL)