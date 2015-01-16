cmake_minimum_required(VERSION 2.8)

project(N4Bias)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(N4Bias N4Bias.cxx)

target_link_libraries(N4Bias ${ITK_LIBRARIES})
