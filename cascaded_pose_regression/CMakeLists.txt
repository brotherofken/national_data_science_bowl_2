cmake_minimum_required(VERSION 2.8)

get_filename_component(PROJECT_SOURCE_PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
if(EXISTS "${PROJECT_SOURCE_PARENT_DIR}/config.cmake")
	message(STATUS "Include additional config from config.cmake")
	include("${PROJECT_SOURCE_PARENT_DIR}/config.cmake")
endif()

set(GDCM_USE_FILE "${PROJECT_SOURCE_PARENT_DIR}/libs/gdcm/lib/gdcm-2.6")
set(GDCM_DIR "${PROJECT_SOURCE_PARENT_DIR}/libs/gdcm/lib/gdcm-2.6")
find_package(GDCM REQUIRED)

find_package( OpenCV REQUIRED )


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

set(source ./FaceAlignment.h ./FernCascade.cpp ./Fern.cpp ./ShapeRegressor.cpp ./Utils.cpp ./TrainDemo.cpp)
add_executable(TrainDemo.out ${source})
include_directories(${OpenCV_INCLUDE_DIRS} ${GDCM_INCLUDE_DIRS})
link_directories(${GDCM_LIBRARY_DIRS})
target_link_libraries(TrainDemo.out ${OpenCV_LIBS} gdcmCommon gdcmcharls gdcmMEXD gdcmMSFF)

set(source ./FaceAlignment.h ./FernCascade.cpp ./Fern.cpp ./ShapeRegressor.cpp ./Utils.cpp ./TestDemo.cpp)
add_executable(TestDemo.out ${source})
include_directories(${OpenCV_INCLUDE_DIRS} ${GDCM_INCLUDE_DIRS})
link_directories(${GDCM_LIBRARY_DIRS})
target_link_libraries(TestDemo.out ${OpenCV_LIBS} gdcmCommon gdcmcharls gdcmMEXD gdcmMSFF)


