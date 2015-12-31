// -*- c++ -*-
//
// Time-stamp: <2015-04-22 09:59:37 zophos>
//
#define DEBUG

#include <iostream>
#include <fstream>

#include <gdcmImageReader.h>

#include <opencv2/opencv.hpp>

cv::Mat1d read_dcm(const char * filename)
{
	// Read DCM
	gdcm::ImageReader ir;
	ir.SetFileName(filename);
	if (!ir.Read()) {
		return cv::Mat1d();
	}

	std::cout << "Getting image from ImageReader..." << std::endl;

	const gdcm::Image &gimage = ir.GetImage();

	std::vector<short> vbuffer(gimage.GetBufferLength());
	gimage.GetBuffer((char*)&vbuffer[0]);

	//const unsigned int* const dimension = gimage.GetDimensions();
	const unsigned int size_x = gimage.GetDimensions()[0];
	const unsigned int size_y = gimage.GetDimensions()[1];

	cv::Mat1d image(size_y, size_x);

	std::copy(vbuffer.begin(), vbuffer.end(), image.begin());
	return image;
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		return 1;
	}
	const char *filename = argv[1];

	cv::Mat1d image = read_dcm(filename);

	double vmin, vmax;
	cv::minMaxLoc(image, &vmin, &vmax);
	image = (image - vmin) / (vmax - vmin);

	cv::imshow("", image);
	cv::waitKey(0);

	cv::imwrite("1.png", 255*image);

	return 0;
}
