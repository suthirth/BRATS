cmake_minimum_required(VERSION 2.8)

project(ExtractFeatures)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(ExtractFeatures ExtractFeatures.cxx)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

target_link_libraries(ExtractFeatures ${ITK_LIBRARIES})
