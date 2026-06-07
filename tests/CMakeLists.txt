find_package(Python COMPONENTS Interpreter REQUIRED)

add_executable(cinder_tensor_smoke
  tensor_smoke.cpp
)

target_include_directories(cinder_tensor_smoke
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(cinder_tensor_smoke PRIVATE cinder_options)

add_test(
  NAME tensor_smoke
  COMMAND cinder_tensor_smoke
)

add_test(
  NAME pytest
  COMMAND ${Python_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests
)

set_tests_properties(pytest
  PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}"
)
