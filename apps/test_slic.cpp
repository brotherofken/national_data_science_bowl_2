/*
 * test_slic.cpp.
 *
 * Written by: Pascal Mettes.
 *
 * This file creates an over-segmentation of a provided image based on the SLIC
 * superpixel algorithm, as implemented in slic.h and slic.cpp.
 */
 
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;

#include <slic/slic.h>

int main(int argc, char *argv[]) {
    /* Load the image and convert to Lab colour space. */
    cv::Mat3b image_uc = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::resize(image_uc, image_uc, cv::Size(), 2.0, 2.0);
	cv::Mat3b lab_image_uc;
	cv::cvtColor(image_uc, lab_image_uc, cv::COLOR_BGR2Lab);
	cv::Mat3d image, lab_image;
	lab_image_uc.convertTo(lab_image, CV_64FC3);
	image_uc.convertTo(image, CV_64FC3, 1 / 255.);

    
    /* Yield the number of superpixels and weight-factors from the user. */
	const int w = image.cols;
	const int h = image.rows;
    const int nr_superpixels = std::stoi(argv[2]);
	const int nc = std::stoi(argv[3]);

    const double step = std::sqrt((w * h) / double(nr_superpixels));
    
    /* Perform the SLIC superpixel algorithm. */
    Slic slic;
    slic.generate_superpixels(lab_image, step, nc);
    slic.create_connectivity(lab_image);
    
    /* Display the contours and show the result. */
    slic.display_contours(image, cv::Vec3d(0,0,255), 3.0);
	//cv::Mat3b cvimage(image->height, image->width, (cv::Vec3b*)image->imageData, image->widthStep);
	//cv::resize(cvimage, cvimage, cv::Size(0, 0), 3.0, 3.0, cv::INTER_LANCZOS4);
    //cv::imshow("result", cvimage);
	cv::imshow("result", image);
    cv::waitKey(0);
    cv::imwrite(argv[4], 255*image);
}
