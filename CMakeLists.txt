cmake_minimum_required(VERSION 3.18...3.31)

project(GlotNet LANGUAGES C CXX VERSION 0.1.2)

# Built standalone (pip install / direct CMake), or as a subproject of e.g.
# an audio plugin via add_subdirectory/FetchContent? (PROJECT_IS_TOP_LEVEL
# needs CMake 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GLOTNET_IS_TOP_LEVEL ON)
else()
    set(GLOTNET_IS_TOP_LEVEL OFF)
endif()

# Only set global C++ settings when standalone; as a subproject the parent
# decides (glotnet_core still requires C++17 via target_compile_features).
if(GLOTNET_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(GNUInstallDirs)

# Subprojects get only the torch-free core library by default
option(GLOTNET_BUILD_PYTHON "Build the Python extension module (requires torch)" ${GLOTNET_IS_TOP_LEVEL})
option(GLOTNET_BUILD_TESTS "Build C++ tests for the core library" OFF)

# Eigen is header-only: use the submodule headers directly rather than
# configuring Eigen's own CMake project (tests, docs, etc.).
# A parent project that already has Eigen can provide an Eigen3::Eigen
# target before including GlotNet, and that is used instead.
if(NOT TARGET glotnet_eigen)
    add_library(glotnet_eigen INTERFACE)
    if(TARGET Eigen3::Eigen)
        target_link_libraries(glotnet_eigen INTERFACE Eigen3::Eigen)
    else()
        target_include_directories(glotnet_eigen SYSTEM INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
    endif()
endif()

# Core DSP / inference library: plain C++ and Eigen, no torch or Python.
# Parameters are passed as glotnet::ConstArrayView (glotnet/ArrayView.h).
# Public headers: include/glotnet/*.h, used as #include "glotnet/WaveNet.h".
add_library(glotnet_core STATIC
    ext/src/Activations.cpp
    ext/src/Distributions.cpp
    ext/src/Convolution.cpp
    ext/src/ConvolutionLayer.cpp
    ext/src/ConvolutionStack.cpp
    ext/src/WaveNet.cpp
    ext/src/WaveNetAR.cpp
    ext/src/GlotNetAR.cpp
)
add_library(GlotNet::core ALIAS glotnet_core)
target_include_directories(glotnet_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(glotnet_core PUBLIC glotnet_eigen)
target_compile_features(glotnet_core PUBLIC cxx_std_17)
# Linked into the Python module (a shared object), so needs PIC
set_target_properties(glotnet_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(GLOTNET_BUILD_TESTS)
    enable_testing()
    add_executable(glotnet_test_core test/cpp/test_core.cpp)
    target_link_libraries(glotnet_test_core PRIVATE glotnet_core)
    add_test(NAME glotnet_test_core COMMAND glotnet_test_core)
endif()

if(NOT GLOTNET_BUILD_PYTHON)
    return()
endif()

# Python extension module. Built by scikit-build-core during `pip install`,
# or directly with CMake when a Python interpreter with torch is available.
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate torch from the torch Python package. The extension is CPU-only, so
# link libtorch_cpu/c10/torch_python directly instead of find_package(Torch):
# with a CUDA build of torch, TorchConfig requires a full CUDA toolkit (nvcc)
# matching torch's CUDA version, even though nothing here uses CUDA.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import os, torch; print(os.path.dirname(torch.__file__) + ';' + str(int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', True))))"
    OUTPUT_VARIABLE Torch_PY_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(GET Torch_PY_INFO 0 TORCH_INSTALL_PREFIX)
list(GET Torch_PY_INFO 1 TORCH_CXX11_ABI)
set(TORCH_INCLUDE_DIRS
    "${TORCH_INSTALL_PREFIX}/include"
    "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")
foreach(lib torch_cpu c10 torch_python)
    string(TOUPPER ${lib} LIB)
    find_library(${LIB}_LIBRARY ${lib} REQUIRED
        PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
endforeach()
set(TORCH_LIBRARIES ${TORCH_CPU_LIBRARY} ${C10_LIBRARY})

set(python_module_name cpp_extensions)

# pybind11 headers come from torch's include directory (TORCH_INCLUDE_DIRS),
# which guarantees they match the pybind11 version torch was built with.
python_add_library(${python_module_name} MODULE WITH_SOABI
    ext/bindings.cpp
    ext/bind/Activations.cpp
    ext/bind/Convolution.cpp
    ext/bind/ConvolutionLayer.cpp
    ext/bind/ConvolutionStack.cpp
    ext/bind/Distributions.cpp
    ext/bind/WaveNet.cpp
)
set_target_properties(${python_module_name} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# The installed module lives in site-packages/glotnet/ and finds libtorch in
# site-packages/torch/lib/ (torch is a runtime dependency, never bundled).
if(APPLE)
    set_target_properties(${python_module_name} PROPERTIES
        INSTALL_RPATH "@loader_path/../torch/lib")
elseif(UNIX)
    set_target_properties(${python_module_name} PROPERTIES
        INSTALL_RPATH "$ORIGIN/../torch/lib")
endif()
target_compile_definitions(${python_module_name} PRIVATE
    TORCH_EXTENSION_NAME=${python_module_name}
)
# torch headers require C++20 (the core library stays C++17)
target_compile_features(${python_module_name} PRIVATE cxx_std_20)
if(NOT APPLE AND NOT MSVC)
    target_compile_definitions(${python_module_name} PRIVATE
        _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
endif()

target_link_libraries(${python_module_name} PRIVATE glotnet_core)

target_include_directories(${python_module_name} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${python_module_name} PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

if(DEFINED ENV{DOCKER_LIB_PATH})
    target_link_libraries(${python_module_name} PRIVATE -L$ENV{DOCKER_LIB_PATH}/lib/)
endif()

install(TARGETS ${python_module_name} DESTINATION glotnet)
