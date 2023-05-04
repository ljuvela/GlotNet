cmake_minimum_required(VERSION 3.14...3.22)

project(GlotNet LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

include(GNUInstallDirs)

# Add Conda headers
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    include_directories(${CONDA_PREFIX}/include)
endif ()
list(APPEND CMAKE_PREFIX_PATH ${CONDA_PREFIX})

add_subdirectory(third_party/pybind11)
set(python_module_name cpp_extensions)
pybind11_add_module(${python_module_name} MODULE
    ext/bindings.cpp
    ext/bind/Activations.cpp
    ext/bind/Convolution.cpp
    ext/bind/ConvolutionLayer.cpp
    ext/bind/ConvolutionStack.cpp
    ext/bind/Distributions.cpp
    ext/bind/WaveNet.cpp
    ext/src/Activations.cpp
    ext/src/Distributions.cpp
    ext/src/Convolution.cpp
    ext/src/WaveNet.cpp
    ext/src/ConvolutionLayer.cpp
    ext/src/WaveNetAR.cpp
    ext/src/ConvolutionStack.cpp
    ext/src/GlotNetAR.cpp
)

add_subdirectory(third_party/eigen)
target_link_libraries(${python_module_name} PUBLIC Eigen3::Eigen)

if(DEFINED ENV{CONDA_PREFIX})
    target_link_libraries(${python_module_name} PUBLIC -L${CONDA_PREFIX}/lib/)
endif ()

if(DEFINED ENV{DOCKER_LIB_PATH})
    target_link_libraries(${python_module_name} PUBLIC -L${DOCKER_LIB_PATH}/lib/)
endif ()

set(CMDSTR "import torch; print(torch.utils.cmake_prefix_path)")
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${CMDSTR}" OUTPUT_VARIABLE Torch_DIR_RAW)
string(STRIP ${Torch_DIR_RAW} Torch_DIR)
set(Torch_DIR "${Torch_DIR}/Torch")
message(STATUS "Torch_DIR: ${Torch_DIR}")


# find_package(Python COMPONENTS Interpreter)
# set(Torch_DIR "${Python_SITELIB}/torch/share/cmake/Torch")
# if(DEFINED ENV{DOCKER_LIB_PATH})
#     set(Torch_DIR "/usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch")
# endif()

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${python_module_name}
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${TORCH_PYTHON_LIBRARY}
)
target_include_directories(${python_module_name} PRIVATE ${TORCH_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
target_link_libraries (${python_module_name} PRIVATE Eigen3::Eigen)

install(TARGETS ${python_module_name} DESTINATION .)

# The extension module must load the hello library as a dependency when the
# extension module is loaded. The easiest way to locate the hello library is
# via RPATH. Absolute RPATHs are possible, but they make the resulting
# binaries not redistributable to other Python installations (conda is broke,
# wheel reuse is broke, and more!).
#
# Placing the hello library in the package and using relative RPATHs that
# doesn't point outside of the package means that the built package is
# relocatable. This allows for safe binary redistribution.
# if(APPLE)
#     set_target_properties(${python_module_name} PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
# else()
#     set_target_properties(${python_module_name} PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
# endif()