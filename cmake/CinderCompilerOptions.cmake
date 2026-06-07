include_guard(GLOBAL)

option(CINDER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors." OFF)

add_library(cinder_options INTERFACE)
add_library(cinder::options ALIAS cinder_options)

target_compile_features(cinder_options INTERFACE cxx_std_23)

target_compile_options(cinder_options
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wsign-conversion
      -Wshadow
      -Wold-style-cast
      -Wnon-virtual-dtor
      -Woverloaded-virtual
      -Wnull-dereference
      -Wdouble-promotion
      -Wformat=2
    >
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
      /W4
      /permissive-
      /Zc:__cplusplus
    >
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
      --expt-relaxed-constexpr
      --expt-extended-lambda
      --Wreorder
      -Wno-deprecated-gpu-targets
    >
    $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
      -Wall
      -Wextra
      -Wpedantic
    >
)

target_compile_options(cinder_options
  INTERFACE
    $<$<AND:$<BOOL:${CINDER_WARNINGS_AS_ERRORS}>,$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>>:
      -Werror
    >
    $<$<AND:$<BOOL:${CINDER_WARNINGS_AS_ERRORS}>,$<COMPILE_LANG_AND_ID:CXX,MSVC>>:
      /WX
    >
    $<$<AND:$<BOOL:${CINDER_WARNINGS_AS_ERRORS}>,$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>>:
      --Werror=all-warnings
    >
    $<$<AND:$<BOOL:${CINDER_WARNINGS_AS_ERRORS}>,$<COMPILE_LANG_AND_ID:CUDA,Clang>>:
      -Werror
    >
)
