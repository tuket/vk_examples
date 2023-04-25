cmake_minimum_required(VERSION 3.12.0)
project(vk_examples)
set (CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc REQUIRED)

file(GLOB_RECURSE glsl_files "${CMAKE_SOURCE_DIR}/shaders/*.glsl")
foreach(glsl_file ${glsl_files})
    string(REGEX REPLACE "[.]glsl$" ".spirv" spirv_file ${glsl_file})
    message(${spirv_file})
    add_custom_command(
        DEPENDS ${glsl_file}
        OUTPUT ${spirv_file}
        COMMAND ${GLSLC} ${glsl_file} -o ${spirv_file}
    )
    list(APPEND spirv_files ${spirv_file})
endforeach()
add_custom_target(shaders_target DEPENDS ${spirv_files})

add_subdirectory(libs/glm)
add_subdirectory(libs/vma)
add_subdirectory(libs/glfw)
set(COMMON_LIBS glm vma glfw)

add_subdirectory(examples/async_load_image)