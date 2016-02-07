cmake_minimum_required(VERSION 2.8.11)

add_library(libchan_vese
	FontParameters.cpp
	InteractiveData.cpp
	InteractiveDataCirc.cpp
	InteractiveDataRect.cpp
	ParallelPixelFunction.cpp
	VideoWriterManager.cpp
	ChanVeseCommon.hpp
	FontParameters.hpp
	InteractiveData.hpp
	InteractiveDataCirc.hpp
	InteractiveDataRect.hpp
	ParallelPixelFunction.hpp
	VideoWriterManager.hpp
)
include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries( libchan_vese ${OpenCV_LIBS} ${Boost_LIBRARIES})

target_include_directories(libchan_vese PUBLIC "$<INSTALL_INTERFACE:include>")

install(TARGETS libchan_vese EXPORT libchan_veseConfig 
   ARCHIVE DESTINATION lib )

install(EXPORT libchan_veseConfig 
   DESTINATION lib/cmake/libchan_vese)

install(FILES chan_vese.h DESTINATION include)