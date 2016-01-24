add_executable(chan_vese
	basic.c
	chanvese.c
	chanvesecli.c
	cliio.c
	gifwrite.c
	imageio.c
	rgb2ind.c
	basic.h
	chanvese.h
	cliio.h
	gifwrite.h
	imageio.h
	num.h
	rgb2ind.h
	)


target_link_libraries(chan_vese ${OpenCV_LIBS})
include_directories(include ${OpenCV_INCLUDE_DIRS})

if(MSVC)
	set_target_properties (chan_vese PROPERTIES LINK_FLAGS /FORCE:MULTIPLE)
endif()


