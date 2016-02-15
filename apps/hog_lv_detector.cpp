#include "hog_lv_detector.hpp"

cv::Rect HogLvDetector::detect(const cv::Mat1d& imaged, const cv::Point approximate_location, const bool draw_lv)
{
	char key = 27;
	cv::Scalar reference(0, 255, 0);
	cv::Scalar trained(0, 0, 255);
	cv::Mat img;

	std::vector<cv::Rect> locations;

	cv::Mat1b image;
	imaged.convertTo(image, image.type(), 255);
	locations.clear();
	my_hog.detectMultiScale(image, locations, -0.9, cv::Size(), cv::Size(), 1.05, 2);

	cv::Mat draw = imaged.clone();
	if (draw_lv) {
		draw_locations(draw, locations, reference);
	}
	auto dist_to_lv = [&approximate_location](cv::Rect& r) {
		return cv::norm((r.tl() + r.br()) / 2 - approximate_location);
	};


	cv::Rect closest_rect = locations.size() ? *std::min_element(locations.begin(), locations.end(), [&](cv::Rect& a, cv::Rect& b) {
		return dist_to_lv(a) < dist_to_lv(b);
	}) : cv::Rect();
	if (dist_to_lv(closest_rect) < 0.1 * imaged.cols) {
		locations = { closest_rect };
	}
	else {
		locations = {};
	}

	if (draw_lv) {
		cv::merge(std::vector<cv::Mat1d>(3, draw), draw);
		draw_locations(draw, locations, trained);
		cv::imshow("LV_detection", draw);
	}

	if (dist_to_lv(closest_rect) < 0.1 * imaged.cols) {
		return closest_rect;
	}
	else {
		return cv::Rect();
	}
}

void HogLvDetector::get_svm_detector(const cv::Ptr<cv::ml::SVM>& svm, std::vector<float> & hog_detector)
{
	// get the support vectors
	cv::Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	cv::Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

void HogLvDetector::draw_locations(cv::Mat & img, const std::vector<cv::Rect> & locations, const cv::Scalar & color)
{
	if (!locations.empty())
	{
		std::vector<cv::Rect>::const_iterator loc = locations.begin();
		std::vector<cv::Rect>::const_iterator end = locations.end();
		for (; loc != end; ++loc) {
			cv::rectangle(img, *loc, color, 1);
		}
	}
}
