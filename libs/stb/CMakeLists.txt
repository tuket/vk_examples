project(stb)

set(SRCS stb_image.h stb.c)
add_library(stb ${SRCS})
target_include_directories(stb PUBLIC ${PROJECT_SOURCE_DIR})
source_group("" FILES ${SRCS})