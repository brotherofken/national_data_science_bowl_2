cmake_minimum_required(VERSION 2.8.11)

function(add_application_target name)
	add_executable(${name} ${name}.cpp )
	include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/libs ${GDCM_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
	link_directories(${GDCM_LIBRARY_DIRS})

	if(MSVC)
		set_target_properties (${name} PROPERTIES LINK_FLAGS /FORCE:MULTIPLE)
	endif()

	target_link_libraries(${name} libslic liblocal_opencv libmrf libchan_vese ${OpenCV_LIBS} ${Boost_LIBRARIES} gdcmCommon gdcmcharls gdcmMEXD gdcmMSFF)
	install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction()

# sandbox applications
add_application_target(test_mrf)
add_application_target(test_slic)
add_application_target(test_dicom)

add_application_target(annotate_lv)
add_application_target(chan_vese)
	