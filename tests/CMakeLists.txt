find_package(Python COMPONENTS Interpreter REQUIRED)

add_test(
  NAME pytest
  COMMAND ${Python_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests
)

set_tests_properties(pytest
  PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}"
)
