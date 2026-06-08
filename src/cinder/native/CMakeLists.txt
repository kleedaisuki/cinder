set(Python_FIND_VIRTUALENV FIRST)
set(Python_FIND_STRATEGY LOCATION)

option(CINDER_PREFER_PROJECT_VENV "Prefer the active or project-local Python virtual environment." ON)

set(cinder_python_venv_hints)

if(DEFINED ENV{VIRTUAL_ENV})
  list(APPEND cinder_python_venv_hints "$ENV{VIRTUAL_ENV}")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/.venv")
  list(APPEND cinder_python_venv_hints "${PROJECT_SOURCE_DIR}/.venv")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/venv")
  list(APPEND cinder_python_venv_hints "${PROJECT_SOURCE_DIR}/venv")
endif()

if(CINDER_PREFER_PROJECT_VENV OR NOT Python_EXECUTABLE)
  foreach(cinder_python_venv IN LISTS cinder_python_venv_hints)
    foreach(cinder_python_name python3 python)
      set(cinder_python_candidate "${cinder_python_venv}/bin/${cinder_python_name}")
      if(EXISTS "${cinder_python_candidate}")
        set(Python_EXECUTABLE "${cinder_python_candidate}" CACHE FILEPATH "Python interpreter used by Cinder." FORCE)
        break()
      endif()
    endforeach()

    if(Python_EXECUTABLE)
      break()
    endif()
  endforeach()
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE pybind11_cmake_dir_error
  RESULT_VARIABLE pybind11_cmake_dir_result
)

if(pybind11_cmake_dir_result EQUAL 0)
  set(pybind11_DIR "${pybind11_cmake_dir}" CACHE PATH "pybind11 CMake package directory." FORCE)
else()
  message(FATAL_ERROR
    "The selected Python interpreter could not import pybind11.\n"
    "  Python_EXECUTABLE: ${Python_EXECUTABLE}\n"
    "  Error: ${pybind11_cmake_dir_error}\n"
    "Install the project's Python development dependencies in the virtual environment."
  )
endif()

find_package(pybind11 CONFIG REQUIRED)

set(cinder_native_sources
  bindings.cpp
)

if(CINDER_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND cinder_native_sources tensor.cu)
endif()

pybind11_add_module(core
  ${cinder_native_sources}
)

set_target_properties(core
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cinder
)

target_include_directories(core
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(core PRIVATE cinder_options)

if(CINDER_ENABLE_CUDA)
  target_compile_definitions(core PRIVATE CINDER_HAS_CUDA=1)
  target_link_libraries(core PRIVATE CUDA::cudart)
  set_target_properties(core
    PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
  )
endif()

file(COPY ${PROJECT_SOURCE_DIR}/src/cinder/__init__.py DESTINATION ${CMAKE_BINARY_DIR}/cinder)

install(TARGETS core LIBRARY DESTINATION cinder)
