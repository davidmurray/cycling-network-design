cmake_minimum_required(VERSION 3.18)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed.
Please create a directory and run cmake from there, passing the path to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'. Please delete them.")
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(osrm-batch-router C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(STATUS "Building on a 64 bit system")
else()
  message(STATUS "Building on a 32 bit system")
endif()

link_directories(${LibOSRM_LIBRARY_DIRS})
add_executable(osrm_batch_router_parallel osrm_batch_router_parallel.cpp)
add_executable(osrm_batch_router osrm_batch_router.cpp)

set(FMT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fmt-9.1.0/include")
add_compile_definitions(FMT_HEADER_ONLY)
include_directories(SYSTEM ${FMT_INCLUDE_DIR})

find_package(LibOSRM REQUIRED)

target_link_libraries(osrm_batch_router_parallel ${LibOSRM_LIBRARIES} ${LibOSRM_DEPENDENT_LIBRARIES})
target_link_libraries(osrm_batch_router ${LibOSRM_LIBRARIES} ${LibOSRM_DEPENDENT_LIBRARIES})
include_directories(SYSTEM ${LibOSRM_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LibOSRM_CXXFLAGS}")

install(TARGETS osrm_batch_router_parallel osrm_batch_router DESTINATION bin)
