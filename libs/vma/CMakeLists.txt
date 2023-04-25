project(vma)

find_package(Vulkan REQUIRED)

set(SRCS vma.h vma.cpp)
add_library(vma ${SRCS})
target_include_directories(vma PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(vma Vulkan::Vulkan)
source_group("" FILES ${SRCS})