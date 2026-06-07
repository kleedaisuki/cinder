cmake_minimum_required(VERSION 3.20)

project(
  cinder
  VERSION 0.1.0
  DESCRIPTION "Python bindings for C++ functions"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CINDER_ENABLE_CUDA "Enable CUDA language support." ON)

if(CINDER_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CINDER_ENABLE_CUDA is ON, but no CUDA compiler was found.")
  endif()

  enable_language(CUDA)

  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CinderCompilerOptions)

include(CTest)

add_subdirectory(src/cinder/native)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
