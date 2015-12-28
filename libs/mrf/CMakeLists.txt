file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

set(LIBRARY mrf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


add_library(${LIBRARY} STATIC ${SOURCES})
if(RESEARCH)
    add_dependencies(${LIBRARY} ext_gco)
    target_link_libraries(${LIBRARY} gco)
endif()
