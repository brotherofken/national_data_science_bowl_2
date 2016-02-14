/*
 * test_slic.cpp.
 *
 * Initial version written by: Pascal Mettes.
 * Revised by: Rasim Akhunzyanov.
 *
 * This file creates an over-segmentation of a provided image based on the SLIC
 * superpixel algorithm, as implemented in slic.h and slic.cpp.
 */
 
#include <opencv2/opencv.hpp>

#include <math.h>
#include <vector>

#include <slic/slic.h>
#include <opencv/seeds.hpp>
#include <opencv/slic.hpp>

#include <gdcmImageReader.h>

cv::Mat1d read_dcm(const std::string& filename)
{
	// Read DCM
	gdcm::ImageReader ir;
	ir.SetFileName(filename.c_str());
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


int main(int argc, char *argv[]) {

#if 0
	// Fast check of SLIC and SEEDS from opencv.
	cv::Mat1b image_ss = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	cv::resize(image_ss, image_ss, cv::Size(), 2.0, 2.0, CV_INTER_CUBIC);

	//cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds = cv::ximgproc::createSuperpixelSEEDS(image_ss.cols, image_ss.rows, image_ss.channels(), 400, 10);
	//seeds->iterate(image_ss);
	cv::Ptr<cv::ximgproc::SuperpixelSLIC> seeds = cv::ximgproc::createSuperpixelSLIC(image_ss, cv::ximgproc::SLIC, 20, 10);// image_ss.cols, image_ss.rows, image_ss.channels(), 400, 10);
	seeds->iterate();

	cv::Mat labels;
	seeds->getLabels(labels);

	cv::Mat mask;
	seeds->getLabelContourMask(mask, false);
	mask = mask | image_ss;
	cv::imshow("result", mask);
	cv::imwrite("1_segm_ocv.png", mask);

	cv::waitKey(0);
	return 0;
#endif 
    // Load the image and convert to Lab color space
	cv::Mat1b image_dcm = read_dcm(argv[1]);
	
	cv::Mat3b image_uc;// cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::cvtColor(image_dcm, image_uc, CV_GRAY2BGR);

	cv::resize(image_uc, image_uc, cv::Size(), 2.0, 2.0);
	cv::Mat3b lab_image_uc;
	cv::cvtColor(image_uc, lab_image_uc, cv::COLOR_BGR2Lab);
	cv::Mat1d lab_image;
	image_dcm.convertTo(lab_image, CV_64FC1);
	cv::Mat3d image;
	cv::merge(std::vector<cv::Mat1d>(3, lab_image / 255), image);

    // Get number of superpixels and threshold
	const int superpixel_num = std::stoi(argv[2]);
	const int nc = std::stoi(argv[3]);
    
    // Apply SLIC
    Slic slic;
    slic.generate_superpixels(lab_image, superpixel_num, nc);
    slic.create_connectivity(lab_image);

	std::string outfile = argv[4];
	outfile.erase(outfile.find(".png"));
	outfile += "-input.png";

	std::cout << outfile << std::endl;;

	cv::imwrite(outfile, 255 * image);
	slic.display_contours(image, cv::Vec3d(0, 0, 255), 3.0);
#if defined(_DEBUG)    
    // Display the contours and show the result.
	cv::imshow("result", image);

	// Display clusters
	cv::Mat3d image_clst = lab_image.clone();
	slic.colour_with_cluster_means(image_clst);
	cv::Mat3b image_clst_uc;
	image_clst.convertTo(image_clst_uc, CV_8UC3);
	cv::cvtColor(image_clst_uc, image_clst_uc, cv::COLOR_Lab2BGR);

	cv::imshow("result_clustered", image_clst_uc);
    cv::waitKey(0);
#endif
	cv::imwrite(argv[4], 255 * image);
}
