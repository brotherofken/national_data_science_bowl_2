cmake_minimum_required(VERSION 2.8.11)

function(add_application_target name sources headers)
	add_executable(${name} ${name}.cpp ${sources} ${headers})
	include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/libs ${PROJECT_SOURCE_DIR}/elibs/eigen ${GDCM_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
	link_directories(${GDCM_LIBRARY_DIRS})

	if(MSVC)
		set_target_properties (${name} PROPERTIES LINK_FLAGS /FORCE:MULTIPLE)
	endif()

	target_link_libraries(${name} libslic liblocal_opencv libmrf libchan_vese ${OpenCV_LIBS} ${Boost_LIBRARIES} gdcmCommon gdcmcharls gdcmMEXD gdcmMSFF)
	install(TARGETS ${name} RUNTIME DESTINATION bin)
	set_property(TARGET ${name} PROPERTY FOLDER apps)
endfunction()

# sandbox applications
add_application_target(test_mrf "" "")
add_application_target(test_slic "" "")
add_application_target(test_dicom "" "")

add_application_target(annotate_lv "" "")

# Chan-Vese segmentation attempt
SET(SOURCE dicom_reader.cpp contour_extraction.cpp)
SET(HEADERS dicom_reader.hpp contour_extraction.hpp)
add_application_target(chan_vese "${SOURCE}" "${HEADERS}")

# Cascaded pose regression binaries
SET(SOURCE  dicom_reader.cpp hog_lv_detector.cpp contour_extraction.cpp ./cpr/FernCascade.cpp ./cpr/Fern.cpp ./cpr/ShapeRegressor.cpp ./cpr/Utils.cpp)
SET(HEADERS dicom_reader.hpp hog_lv_detector.hpp contour_extraction.hpp ./cpr/FaceAlignment.h)
add_application_target(cpr_train "${SOURCE}" "${HEADERS}")

SET(SOURCE  dicom_reader.cpp hog_lv_detector.cpp contour_extraction.cpp ./cpr/FernCascade.cpp ./cpr/Fern.cpp ./cpr/ShapeRegressor.cpp ./cpr/Utils.cpp)
SET(HEADERS dicom_reader.hpp hog_lv_detector.hpp contour_extraction.hpp ./cpr/FaceAlignment.h)
add_application_target(cpr_test "${SOURCE}" "${HEADERS}")

# HOG object detection
SET(SOURCE dicom_reader.cpp)
SET(HEADERS dicom_reader.hpp hog_parameters)
add_application_target(hog_prepare_samples "${SOURCE}" "${HEADERS}")

SET(SOURCE dicom_reader.cpp) # hog_svm_trainer.cpp)
SET(HEADERS dicom_reader.hpp hog_parameters.hpp)
add_application_target(hog_train "${SOURCE}" "${HEADERS}")


# End-to-end
SET(SOURCE dicom_reader.cpp contour_extraction.cpp hog_lv_detector.cpp ./cpr/FernCascade.cpp ./cpr/Fern.cpp ./cpr/ShapeRegressor.cpp ./cpr/Utils.cpp)
SET(HEADERS dicom_reader.hpp contour_extraction.hpp hog_lv_detector.hpp ./cpr/FaceAlignment.h)
add_application_target(compute_volume "${SOURCE}" "${HEADERS}")