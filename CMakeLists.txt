cmake_minimum_required(VERSION 3.12)

# (step 1) Create the project and set the version

project(generator VERSION 0.0.1 LANGUAGES C CXX)

# (step 2) Prevent in source builds
# https://cliutils.gitlab.io/modern-cmake/chapters/basics/structure.html

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)

if(EXISTS "${LOC_PATH}")
	message(FATAL_ERROR "You cannot build in a source directory. Please make a
		build subdirectory. Remove remove CMakeCache.txt and CMakeFiles.")
endif()

# (step 3) Add the additional module paths
# https://cliutils.gitlab.io/modern-cmake/chapters/basics/structure.html

list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")

# (step 4) Set a safer default for the install prefix
# https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT.html

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH
		"Install path prefix." FORCE)
endif()

# (step 5) Set the options for the build
# https://blog.kitware.com/cmake-and-the-default-build-type/

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING
		"Options are Debug, Release, MinSizeRel or RelWithDebInfo." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
		MinSizeRel RelWithDebInfo)
endif()

# Add other binary options here with option()...

# (step 6) Include some modules

include(EnableTesting)

# (step 7) Add subdirectories

add_subdirectory(cmake)
add_subdirectory(src)
