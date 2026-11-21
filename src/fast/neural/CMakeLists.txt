# New Dawn neural rendering: the DLSS-NR network as a Vulkan compute post-process beside the OpenGL renderer.
#
# Builds as part of libultraship (add_subdirectory) or on its own for the standalone runner:
#   cmake -S libultraship/src/fast/neural -B build-neural -G Ninja && cmake --build build-neural
cmake_minimum_required(VERSION 3.24)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(NewDawnNeural C CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE RelWithDebInfo)
    endif()
    set(NEWDAWN_NEURAL_TOOL_DEFAULT ON)
else()
    set(NEWDAWN_NEURAL_TOOL_DEFAULT OFF)
endif()
option(NEWDAWN_NEURAL_TOOL "Build nd-nr, the standalone neural rendering runner" ${NEWDAWN_NEURAL_TOOL_DEFAULT})

# Vulkan is opened at run time through volk: headers only at build time, no link dependency, and a machine without
# a Vulkan loader still starts the game. Neither repository's own CMake project is used.
include(FetchContent)
FetchContent_Declare(nd_vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG vulkan-sdk-1.4.363.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR not-a-directory
)
FetchContent_Declare(nd_volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.363.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR not-a-directory
)
FetchContent_MakeAvailable(nd_vulkan_headers nd_volk)

find_package(nlohmann_json REQUIRED)

add_library(newdawn_neural STATIC
    NrVk.cpp
    NrModel.cpp
    NrGraph.cpp
    NrRuntime.cpp
    ${nd_volk_SOURCE_DIR}/volk.c
)
target_include_directories(newdawn_neural PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${nd_volk_SOURCE_DIR}
    ${nd_vulkan_headers_SOURCE_DIR}/include
)
target_compile_definitions(newdawn_neural PUBLIC VK_NO_PROTOTYPES)
target_compile_features(newdawn_neural PUBLIC cxx_std_20)
target_link_libraries(newdawn_neural PUBLIC nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
set_target_properties(newdawn_neural PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NEWDAWN_NEURAL_TOOL)
    find_package(Threads REQUIRED)
    add_executable(nd-nr tool/NrTool.cpp tool/NrSynth.cpp NrReference.cpp)
    target_link_libraries(nd-nr PRIVATE newdawn_neural Threads::Threads)
endif()
