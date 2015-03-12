cmake_minimum_required(VERSION 2.8)

project(CSFMean)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(CSFMean csfmean.cxx)

target_link_libraries(CSFMean ${ITK_LIBRARIES})
