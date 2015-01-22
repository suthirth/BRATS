cmake_minimum_required(VERSION 2.8)

project(ExtractFeatures)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(ExtractFeatures ExtractFeatures.cxx)

target_link_libraries(ExtractFeatures ${ITK_LIBRARIES})
