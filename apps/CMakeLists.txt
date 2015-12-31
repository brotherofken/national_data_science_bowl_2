cmake_minimum_required(VERSION 2.8.11)

# SLIC test
add_executable(test_slic test_slic.cpp )
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/libs)
target_link_libraries(test_slic libslic ${OpenCV_LIBS})

install(TARGETS test_slic 
        RUNTIME DESTINATION bin)

# MRF test
add_executable(test_mrf test_mrf.cpp )
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/libs)
target_link_libraries(test_mrf libmrf ${OpenCV_LIBS})

install(TARGETS test_mrf 
        RUNTIME DESTINATION bin)
		
# DICOM test
add_executable(test_dicom test_dicom.cpp )
include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/libs ${GDCM_INCLUDE_DIRS})
#include(${GDCM_USE_FILE})

link_directories(${GDCM_LIBRARY_DIRS})
if(MSVC)
	#target_link_libraries(test_dicom INTERFACE "/FORCE:MULTIPLE")
	set_target_properties (test_dicom PROPERTIES LINK_FLAGS /FORCE:MULTIPLE)
	target_link_libraries(test_dicom  ${OpenCV_LIBS} gdcmCommon gdcmcharls gdcmMEXD gdcmMSFF)
else()
	target_link_libraries(test_dicom  ${OpenCV_LIBS} gdcmCommon gdcmcharls gdcmMEXD gdcmMSFF)
endif()
install(TARGETS test_dicom RUNTIME DESTINATION bin)

	