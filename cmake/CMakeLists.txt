# Add install target, configure and version files
# https://cliutils.gitlab.io/modern-cmake/chapters/install/installing.html

install(EXPORT ${PROJECT_NAME}Targets
	NAMESPACE ${PROJECT_NAME}::
	DESTINATION cmake)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
	@ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
	DESTINATION cmake)
