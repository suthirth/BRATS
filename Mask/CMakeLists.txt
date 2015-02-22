cmake_minimum_required(VERSION 2.8)

project(Mask)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(Mask Mask.cxx)


target_link_libraries(Mask ${ITK_LIBRARIES})
