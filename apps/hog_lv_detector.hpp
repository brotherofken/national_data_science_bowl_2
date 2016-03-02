#ifndef __HOLG_LV_DETECTOR
#define __HOLG_LV_DETECTOR

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class HogLvDetector {

public:
	cv::Ptr<cv::ml::SVM> svm;
	cv::HOGDescriptor my_hog;

	HogLvDetector()
		: svm(cv::ml::StatModel::load<cv::ml::SVM>("lv_detector.yml")) // Load the trained SVM.
		, my_hog(cv::Size(32, 32), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1, -1.0, cv::HOGDescriptor::L2Hys, 0.05, false, 64, false)

	{
		// Set the trained svm to my_hog
		std::vector<float> hog_detector;
		get_svm_detector(svm, hog_detector);
		my_hog.setSVMDetector(hog_detector);
	}

	cv::Rect detect(const cv::Mat1d& imaged, const cv::Point approximate_location, const bool draw_lv = false);
private:
	void get_svm_detector(const cv::Ptr<cv::ml::SVM>& svm, std::vector<float> & hog_detector);
	void draw_locations(cv::Mat & img, const std::vector<cv::Rect> & locations, const cv::Scalar & color);
};

#endif // !__HOLG_LV_DETECTOR
