find_package(Python COMPONENTS Interpreter REQUIRED)

set(cinder_tensor_smoke_source tensor_smoke.cu)

if(NOT CINDER_ENABLE_CUDA)
  set_source_files_properties(${cinder_tensor_smoke_source}
    PROPERTIES
      LANGUAGE CXX
  )
endif()

add_executable(cinder_tensor_smoke
  ${cinder_tensor_smoke_source}
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
  COMMAND
    ${Python_EXECUTABLE}
    -c
    "import sys, pytest; sys.meta_path = [finder for finder in sys.meta_path if finder.__class__.__module__ != '_cinder_editable']; sys.path.insert(0, '${CMAKE_BINARY_DIR}'); raise SystemExit(pytest.main(['${PROJECT_SOURCE_DIR}/tests']))"
)

set_tests_properties(pytest
  PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}"
)
