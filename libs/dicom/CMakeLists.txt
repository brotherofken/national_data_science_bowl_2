cmake_minimum_required(VERSION 2.8.11)

file (GLOB HEADERS "*.hpp")
file (GLOB SOURCES "[^_]*.cpp")

source_group("Header files" FILES ${HEADERS})

add_library(libdicom STATIC ${SOURCES} ${HEADERS})

#include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})


#target_link_libraries(libd ${OpenCV_LIBS} )

target_include_directories(libslic PUBLIC "$<INSTALL_INTERFACE:include>")

install(TARGETS libdicom EXPORT libdicomConfig ARCHIVE DESTINATION lib )

install(EXPORT libdicomConfig DESTINATION lib/cmake/libdicom)

install(FILES dicom.hpp DESTINATION include)