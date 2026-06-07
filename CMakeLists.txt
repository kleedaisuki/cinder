cmake_minimum_required(VERSION 3.20)

project(
  cinder
  VERSION 0.1.0
  DESCRIPTION "Python bindings for C++ functions"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
