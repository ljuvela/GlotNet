cmake_minimum_required(VERSION 3.18...3.31)

project(GlotNet LANGUAGES C CXX VERSION 0.1.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

option(GLOTNET_BUILD_PYTHON "Build the Python extension module (requires torch)" ON)

# Eigen is header-only: use the submodule headers directly rather than
# configuring Eigen's own CMake project (tests, docs, etc.)
if(NOT TARGET glotnet_eigen)
    add_library(glotnet_eigen INTERFACE)
    target_include_directories(glotnet_eigen SYSTEM INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
endif()

# Core DSP / inference library: plain C++ and Eigen, no torch or Python.
# Parameters are passed as glotnet::ConstArrayView (ext/src/ArrayView.h).
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
target_include_directories(glotnet_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ext/src)
target_link_libraries(glotnet_core PUBLIC glotnet_eigen)
target_compile_features(glotnet_core PUBLIC cxx_std_17)
# Linked into the Python module (a shared object), so needs PIC
set_target_properties(glotnet_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT GLOTNET_BUILD_PYTHON)
    return()
endif()

# Python extension module. Built by scikit-build-core during `pip install`,
# or directly with CMake when a Python interpreter with torch is available.
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate the Torch CMake config from the torch Python package
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE Torch_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${Torch_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED
    PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)

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
