cmake_minimum_required(VERSION 2.8.11)

add_library(libslic slic.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( libslic ${OpenCV_LIBS} )

target_include_directories(libslic 
   PUBLIC "$<INSTALL_INTERFACE:include>")

install(TARGETS libslic EXPORT libslicConfig 
   ARCHIVE DESTINATION lib )

install(EXPORT libslicConfig 
   DESTINATION lib/cmake/libslic)

install(FILES slic.h DESTINATION include)

set_property(TARGET libslic PROPERTY FOLDER libs)
