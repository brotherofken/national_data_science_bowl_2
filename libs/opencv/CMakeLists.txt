cmake_minimum_required(VERSION 2.8.11)

add_library(liblocal_opencv seeds.cpp slic.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(liblocal_opencv ${OpenCV_LIBS} )
add_definitions(-D__OPENCV_BUILD)
target_include_directories(liblocal_opencv PUBLIC "$<INSTALL_INTERFACE:include>")

install(TARGETS liblocal_opencv EXPORT liblocal_opencvConfig 
   ARCHIVE DESTINATION lib )

install(EXPORT liblocal_opencvConfig 
   DESTINATION lib/cmake/liblocal_opencv)

install(FILES slic.hpp seeds.hpp DESTINATION include)

set_property(TARGET liblocal_opencv PROPERTY FOLDER libs)
