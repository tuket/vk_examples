set(SRCS
	../common.hpp
	main.cpp
)

add_executable(async_load_image ${SRCS})
add_dependencies(async_load_image shaders_target)
target_link_libraries(async_load_image ${COMMON_LIBS})

source_group("" FILES ${SRCS})
set_property(TARGET async_load_image PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")