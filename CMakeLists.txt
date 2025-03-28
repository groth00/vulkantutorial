cmake_minimum_required(VERSION 3.31)
project(VulkanTutorial)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG")

find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

file(GLOB SRC_FILES src/*.cpp)
add_executable(VulkanApp src/main.cpp)

target_include_directories(VulkanApp PUBLIC include)
target_link_libraries(VulkanApp Vulkan::Vulkan glfw)

if(APPLE)
  target_compile_definitions(VulkanApp PRIVATE VK_USE_PLATFORM_MACOS_MVK)
  target_link_libraries(VulkanApp "-framework Cocoa" "-framework Metal")
endif()

message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan lib: ${Vulkan_LIBRARIES}")
message(STATUS "GLFW include: ${GLFW_INCLUDE_DIRS}")
message(STATUS "GLFW lib: ${GLFW_LIBRARIES}")
