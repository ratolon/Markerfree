cmake_minimum_required(VERSION 3.5)

include_directories(${MSGMAP_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/src)
link_directories (${PROJECT_BINARY_DIR}/lib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(src)
#add_subdirectory(lib)
