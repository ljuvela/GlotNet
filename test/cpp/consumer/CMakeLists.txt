# Minimal downstream project (e.g. an audio plugin) using the GlotNet core
# library via FetchContent. Doubles as a usage example and a CI check that
# GlotNet works as a subproject without Python or torch.
#
# To build against a local checkout instead of fetching from GitHub:
#   cmake -S test/cpp/consumer -B build-consumer \
#         -DFETCHCONTENT_SOURCE_DIR_GLOTNET=$PWD
cmake_minimum_required(VERSION 3.18...3.31)
project(GlotNetConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20) # parent's choice; GlotNet needs at least C++17

include(FetchContent)
FetchContent_Declare(glotnet
    GIT_REPOSITORY https://github.com/ljuvela/GlotNet.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glotnet)

if(TARGET cpp_extensions)
    message(FATAL_ERROR "GlotNet Python module should not be built as a subproject")
endif()

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE GlotNet::core)

enable_testing()
add_test(NAME consumer COMMAND consumer)
