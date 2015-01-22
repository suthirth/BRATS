cmake_minimum_required(VERSION 2.8)

project(Normalization)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(Normalization Normalization.cxx)

target_link_libraries(Normalization ${ITK_LIBRARIES})
