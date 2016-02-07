cmake_minimum_required(VERSION 3.0.0)

project(ndsb2)

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
#configure_msvc_runtime()

find_package(OpenCV REQUIRED)

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
find_package(Boost REQUIRED filesystem program_options system)

set(GDCM_USE_FILE "${PROJECT_SOURCE_DIR}/libs/gdcm/lib/gdcm-2.6")
set(GDCM_DIR "${PROJECT_SOURCE_DIR}/libs/gdcm/lib/gdcm-2.6")
find_package(GDCM REQUIRED)

add_subdirectory(apps)
add_subdirectory(libs/slic)
add_subdirectory(libs/mrf)
add_subdirectory(libs/opencv)
add_subdirectory(libs/chan_vese)
#add_subdirectory(chan_vese_segmentation)

macro(set_directory ttarget tdir)
set_property(TARGET ${ttarget} PROPERTY FOLDER ${tdir})
endmacro()

macro(link_gdcm)
	set(GDCM_STANDALONE ON)
	#add_subdirectory(libs/gdcm-2.6.2)
	set_directory(gdcmcharls "GDCM")
	set_directory(gdcmopenjpeg "GDCM")
	set_directory(gdcmzlib "GDCM")
	set_directory(gdcmgetopt "GDCM")
	set_directory(socketxx "GDCM")
	set_directory(gdcmexpat "GDCM")
	set_directory(gdcmDICT "GDCM")
	set_directory(gdcmDSED "GDCM")
	set_directory(gdcmIOD "GDCM")
	set_directory(gdcmMEXD "GDCM")
	set_directory(gdcmMSFF "GDCM")
	set_directory(gdcmCommon "GDCM")
	set_directory(gdcmjpeg8 "GDCM")
	set_directory(gdcmjpeg12 "GDCM")
	set_directory(gdcmjpeg16 "GDCM")
endmacro()
link_gdcm()

add_dependencies(test_slic libslic)
add_dependencies(test_mrf libmrf)
