cmake_minimum_required(VERSION 3.0.0)

project(chan_vese)

# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake
# defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(EXISTS "${PROJECT_SOURCE_DIR}/config.cmake")
	message(STATUS "Include additional config from config.cmake")
	include("config.cmake")
endif()

macro(configure_msvc_runtime)
  if(MSVC)
    # Default to statically-linked runtime.
    if("${MSVC_RUNTIME}" STREQUAL "")
      set(MSVC_RUNTIME "static")
    endif()
    # Set compiler options.
    set(variables
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    if(${MSVC_RUNTIME} STREQUAL "static")
      message(STATUS
        "MSVC -> forcing use of statically-linked runtime."
      )
      foreach(variable ${variables})
        if(${variable} MATCHES "/MD")
          string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
        endif()
      endforeach()
    else()
      message(STATUS
        "MSVC -> forcing use of dynamically-linked runtime."
      )
      foreach(variable ${variables})
        if(${variable} MATCHES "/MT")
          string(REGEX REPLACE "/MT" "/MD" ${variable} "${${variable}}")
        endif()
      endforeach()
    endif()
  endif()
endmacro()
configure_msvc_runtime()

find_package(OpenCV REQUIRED)

#add_definitions( -DBOOST_ALL_NO_LIB )
set(Boost_USE_STATIC_LIBS        ON)
#set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     ON)
find_package(Boost REQUIRED filesystem program_options system)

add_executable(chan_vese
	./src/main.cpp
	./src/FontParameters.cpp
	./src/InteractiveData.cpp
	./src/InteractiveDataCirc.cpp
	./src/InteractiveDataRect.cpp
	./src/ParallelPixelFunction.cpp
	./src/VideoWriterManager.cpp
	./include/ChanVeseCommon.hpp
	./include/FontParameters.hpp
	./include/InteractiveData.hpp
	./include/InteractiveDataCirc.hpp
	./include/InteractiveDataRect.hpp
	./include/ParallelPixelFunction.hpp
	./include/VideoWriterManager.hpp
	)
	
message(STATUS >> ${Boost_LIBRARIES} ${Boost_LIBRARY_DIR})
message(STATUS >> ${Boost_INCLUDE_DIRS})
link_directories("D:/lib/boost_1_59_0/lib64-msvc-14.0")
target_link_libraries(chan_vese ${OpenCV_LIBS} ${Boost_LIBRARIES})


include_directories(include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

if(MSVC)
	set_target_properties (${name} PROPERTIES LINK_FLAGS /FORCE:MULTIPLE)
endif()


