find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE pybind11_cmake_dir_result
)

if(pybind11_cmake_dir_result EQUAL 0)
  list(APPEND CMAKE_PREFIX_PATH ${pybind11_cmake_dir})
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
  bindings.cpp
  add.cpp
)

set_target_properties(_core
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cinder
)

target_include_directories(_core
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(_core PRIVATE cinder_options)

file(COPY ${PROJECT_SOURCE_DIR}/src/cinder/__init__.py DESTINATION ${CMAKE_BINARY_DIR}/cinder)

install(TARGETS _core LIBRARY DESTINATION cinder)
