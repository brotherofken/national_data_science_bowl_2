cmake_minimum_required(VERSION 2.8.11)

set(LIBRARY libmrf)

file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

source_group("Header files" FILES ${HEADERS})


add_library(${LIBRARY} STATIC ${SOURCES} ${HEADERS})

target_include_directories(libmrf 
   PUBLIC "$<INSTALL_INTERFACE:include>")

install(TARGETS libmrf EXPORT libmrfConfig 
   ARCHIVE DESTINATION lib )

install(EXPORT libmrfConfig 
   DESTINATION lib/cmake/libmrf)

install(FILES ${HEADERS} DESTINATION include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")