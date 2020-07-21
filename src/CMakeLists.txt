# Create a default setup for targets

function(target_default_setup TARGET)
	target_compile_options(${TARGET} PRIVATE -Wall -fno-math-errno -msse3)
	target_compile_features(${TARGET} PRIVATE cxx_std_17)

	set_target_properties(${TARGET} PROPERTIES CXX_EXTENSIONS OFF)
	set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
	set_target_properties(${TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

	target_include_directories(${TARGET} PRIVATE
		${PROJECT_SOURCE_DIR}/extern
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_BINARY_DIR}/src)

	target_include_directories(${TARGET} PUBLIC
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
		$<INSTALL_INTERFACE:include>)
endfunction()

# Find common dependencies

find_package(TBB REQUIRED)

mark_as_advanced(TBB_DIR)

# Add subdirectories here...

# Create a binary target

add_executable(${PROJECT_NAME}
	main.cpp)

# Give target a default setup

target_default_setup(${PROJECT_NAME})

# Link private dependencies

target_link_libraries(${PROJECT_NAME} PRIVATE
	TBB::tbb)

# Setup target for install

install(TARGETS ${PROJECT_NAME}
	EXPORT ${PROJECT_NAME}Targets
	RUNTIME DESTINATION bin)
