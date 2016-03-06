/*
Author: Bi Sai
Date: 2014/06/18
This program is a reimplementation of algorithms in "Face Alignment by Explicit
Shape Regression" by Cao et al.
If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

Copyright (c) 2014 Bi Sai
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <boost/filesystem.hpp>

#include <opencv2/photo.hpp>

#include "cpr/FaceAlignment.h"
#include "hog_lv_detector.hpp"
#include "dicom_reader.hpp"
#include "contour_extraction.hpp"

namespace bfs = boost::filesystem;

cv::RotatedRect get_ellipse_for_point(const cv::Mat1f& img, const cv::Point& estimated_center)
{
// = cur_slice.estimated_center;// cur_slice.point_to_image(inter.p);
	const double width = 0.2 * img.cols;
	cv::Rect roi(estimated_center - cv::Point(width, width), estimated_center + cv::Point(width, width));
	cv::Mat1f img_roi = img.clone()(roi);

	{
		const size_t dilate_width = 10. / 256.*img.cols;
		//cv::dilate(img_roi, img_roi, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_width, dilate_width)));
		//cv::erode(img_roi, img_roi, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_width, dilate_width)));
		cv::Mat1b tmp;
		img_roi.convertTo(tmp, CV_8UC1, 255);
		cv::medianBlur(tmp, tmp, 7);
		tmp.convertTo(img_roi, CV_32FC1, 1. / 255);
	}
	cv::Mat1f kernel = cv::getGaussianKernel(roi.width, 0.3*((roi.width - 1)*0.5 - 1) + 0.8, CV_32FC1);
	kernel /= kernel(roi.width / 2);
	kernel = kernel * kernel.t();
	cv::Mat1f dx, dy, magnitude;
	cv::Sobel(img_roi, dx, CV_32FC1, 1, 0);
	cv::Sobel(img_roi, dy, CV_32FC1, 0, 1);
	cv::magnitude(dx, dy, magnitude);

	cv::Mat1f weighed_magnitude = magnitude.mul(kernel);
	double wmmax;
	cv::minMaxLoc(weighed_magnitude, nullptr, &wmmax);
	weighed_magnitude /= wmmax;

	cv::Point2d mean_point = estimated_center - roi.tl();// .cols / 2, img_roi.rows / 2);
	double R = 0;
	for (size_t x{}; x < weighed_magnitude.cols; ++x) {
		for (size_t y{}; y < weighed_magnitude.rows; ++y) {
			if (weighed_magnitude(y, x) > 0.5) {
				R += cv::norm(mean_point - cv::Point2d(x, y));
			}
		}
	}
	R /= cv::countNonZero(weighed_magnitude > 0.5);

	std::vector<cv::Point> points;
	std::vector<double> weights;
	for (size_t i = 0; i < weighed_magnitude.rows; i++) {
		for (size_t j = 0; j < weighed_magnitude.cols; j++) {
			if (i % 2 && j % 2 && weighed_magnitude(cv::Point(j, i)) > 0.3) {
				points.push_back(cv::Point(j, i));
				weights.push_back(weighed_magnitude(cv::Point(j, i))*wmmax);
			}
		}
	}

	cv::RotatedRect rect = ::fitEllipseToCenter(points, weights, mean_point);
	rect.center += cv::Point2f(roi.tl());// + cv::Point2f(mean_point);

	cv::Mat cur_image = img.clone();
	if (cur_image.channels() == 1) cv::merge(std::vector<cv::Mat>(3, cur_image), cur_image);

	cv::circle(cur_image, estimated_center, 2, cv::Scalar(1., 0., 1.));
	cv::ellipse(cur_image, rect, cv::Scalar(1., 0., 1.));
	//cv::circle(cur_image, roi.tl() + cv::Point2i(mean_point), R, cv::Scalar(1., 1., 0.));
	cv::imshow("ROI magnitude", weighed_magnitude);
	return rect;
}

cv::Mat1i gmm_segmentaiton(const cv::Mat1f& img_roi)
{
	cv::Mat tmp;
	img_roi.convertTo(tmp, CV_8UC1, 255);
	cv::medianBlur(tmp, tmp, 7);
	tmp.convertTo(tmp, CV_32FC1, 1. / 255);


	cv::Mat1f samples(0, 1);
	for (size_t i = 0; i < img_roi.rows; i++) {
		for (size_t j = 0; j < img_roi.cols; j++) {
			samples.push_back(tmp.at<float>(i, j));
		}
	}

#if 0
	cv::Ptr<cv::ml::EM> gmm = cv::ml::EM::create();
	gmm->setClustersNumber(3);
	gmm->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
	gmm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 300, 0.1));

	cv::Mat labels;
	gmm->trainEM(samples, cv::noArray(), labels, cv::noArray());
#else
	cv::Mat labels;
	cv::kmeans(samples, 3, labels, cv::TermCriteria(), 3, cv::KMEANS_PP_CENTERS);
#endif

	labels = labels.reshape(1, img_roi.rows);
	cv::imshow("segmentation", labels * 128 * 255);
	return labels;

}

double get_circle_for_point(const cv::Mat1f& img, const cv::Point& estimated_center)
{
	// = cur_slice.estimated_center;// cur_slice.point_to_image(inter.p);
	const double width = 0.2 * img.cols;
	cv::Rect roi(estimated_center - cv::Point(width, width), estimated_center + cv::Point(width, width));
	cv::Mat1f img_roi = img.clone();
	img_roi = img_roi(roi);

	cv::Mat1i labels = gmm_segmentaiton(img_roi);
	cv::Mat1b segments = labels == labels(estimated_center - roi.tl());

	using contours_t = std::vector<std::vector<cv::Point>>;
	contours_t contours;
	findContours(segments.clone(), contours, {}, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	contours = { *std::find_if(contours.begin(), contours.end(), [&](std::vector<cv::Point>& c) {
		return cv::pointPolygonTest(c, estimated_center - roi.tl(), false) >= 0;
	}) };
	segments.setTo(0);
	{
		std::vector<cv::Point> tmp;
		cv::convexHull(contours[0], tmp, true, true);
		contours[0] = tmp;
	}
	cv::drawContours(segments, contours, 0, cv::Scalar::all(255), 1);
	cv::imshow("gmm segmentation", segments);

	cv::Mat1f kernel = cv::getGaussianKernel(roi.width, 0.3*((roi.width - 1)*0.5 - 1) + 0.8, CV_32FC1);
	kernel /= kernel(roi.width / 2);
	kernel = kernel * kernel.t();
	cv::Mat1f magnitude;
	segments.convertTo(magnitude, CV_32FC1, 1 / 255.);
	cv::Mat1f weighed_magnitude = magnitude.mul(kernel);

	double wmmax;
	cv::minMaxLoc(weighed_magnitude, nullptr, &wmmax);
	weighed_magnitude /= wmmax;
	//kernel.setTo(wmmax, kernel > 0.6 * wmmax);

	cv::Point2d mean_point = estimated_center - roi.tl();// .cols / 2, img_roi.rows / 2);
	double R = 0;
	for (size_t x{}; x < weighed_magnitude.cols; ++x) {
		for (size_t y{}; y < weighed_magnitude.rows; ++y) {
			if (weighed_magnitude(y, x) > 0.5) {
				R += cv::norm(mean_point - cv::Point2d(x, y));
			}
		}
	}
	R /= cv::countNonZero(weighed_magnitude > 0.5);

	return R;
}


int main() {


	std::vector<cv::Mat1b> images;
	std::vector<BoundingBox> bbox;

	//HogLvDetector lv_detector;

	std::cout << "Read images..." << std::endl;
	std::vector<cv::Mat1d> ground_truth_shapes;
	std::vector<BoundingBox> bounding_box;
	std::ifstream fin("lm_dataset/landmarks_annotation.csv");

#if 1

	const int img_num = 95; // WHATTA
	const int candidate_pixel_num = 400; // TOO MUCH
	const int fern_pixel_num = 5;
	const int first_level_num = 10;   // cascades
	const int second_level_num = 500; // trees per cascade
	const int landmark_num = 15;
	const int initial_number = 75;
	bool show_train = true;

	for (int i = 0; i < img_num; i++) {
		std::string image_name;
		BoundingBox bbox;
		fin >> image_name >> bbox.start_x >> bbox.start_y >> bbox.width >> bbox.height;
		bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
		bbox.centroid_y = bbox.start_y + bbox.height / 2.0;
		// Read image
		cv::Mat1d imaged  = cv::imread(image_name, cv::IMREAD_GRAYSCALE);

		cv::Mat1d landmarks(landmark_num, 2);
		for (int j = 0; j < landmark_num; j++) {
			fin >> landmarks(j, 0) >> landmarks(j, 1);
			landmarks(j, 0);
			landmarks(j, 1);
		}

		cv::Mat1b image;
		imaged.convertTo(image, image.type());

		ground_truth_shapes.push_back(landmarks.clone());
		images.push_back(image.clone());
		bounding_box.push_back(BoundingBox(bbox));

		//cv::flip(image, image, 1);
		//
		//for (size_t r{}; r < landmarks.rows; ++r) {
		//	landmarks(r, 0) = image.cols - landmarks(r, 0);
		//}
		//
		//bbox.start_x = image.cols - bbox.start_x - bbox.width;
		//bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
		//
		//ground_truth_shapes.push_back(landmarks);
		//images.push_back(image.clone());
		//bounding_box.push_back(bbox);

		if (show_train) {
			cv::Mat test_image_1 = images.back().clone();
			cv::cvtColor(test_image_1, test_image_1, CV_GRAY2BGR);
			double scale = 1;
			cv::resize(test_image_1, test_image_1, cv::Size(), scale, scale);

			cv::putText(test_image_1, image_name, cv::Point(15, 15), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));
			for (int i = 0; i < landmark_num; i++) {
				circle(test_image_1, cv::Point2d(landmarks(i, 0), landmarks(i, 1))*scale, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
			}

			cv::Rect roi(bbox.start_x*scale, bbox.start_y*scale, bbox.width*scale, bbox.height*scale);
			cv::rectangle(test_image_1, roi, cv::Scalar(255, 0, 0));

			//cv::circle(test_image_1, estimated_center, R, cv::Scalar(255, 0, 255));

			imshow("gt", test_image_1);
			int key = cv::waitKey(0);
			show_train = key != 'q';
		}
		int key = cv::waitKey(1);

	}
	fin.close();
#else

	const int img_num = 500; // WHATTA
	const int candidate_pixel_num = 300; // TOO MUCH
	const int fern_pixel_num = 5;
	const int first_level_num = 10; // cascades
	const int second_level_num = 250; // trees per cascade
	const int landmark_num = 16;
	const int initial_number = 40;
	bool show_train = true;

	for (int i = 0; i < img_num; i++) {
		std::string image_name;
		BoundingBox bbox;
		fin >> image_name >> bbox.start_x >> bbox.start_y >> bbox.width >> bbox.height;

		// Read image
		Slice slice(image_name);

		std::string estimated_centers_filename = "G:/ndsb2/data/lvs/NDSB_" + std::to_string(slice.patient_id) + ".csv";
		const std::map<std::string, cv::Point> points = read_estimated_points(estimated_centers_filename);
		const std::string basename = bfs::basename(bfs::path(image_name));
		cv::Point estimated_center = points.count(basename) ? points.at(basename) : cv::Point(-1, -1);
		estimated_center.x *= slice.pixel_spacing[0];
		estimated_center.y *= slice.pixel_spacing[1];
		
		cv::Mat1d landmarks(landmark_num, 2);
		for (int j = 0; j < landmark_num; j++) {
			fin >> landmarks(j, 0) >> landmarks(j, 1);
			landmarks(j, 0) *= slice.pixel_spacing[0];
			landmarks(j, 1) *= slice.pixel_spacing[1];
		}

		cv::Mat1d imaged = slice.image.clone();
		cv::resize(imaged, imaged, cv::Size(), slice.pixel_spacing[0], slice.pixel_spacing[1], cv::INTER_CUBIC);

		cv::Mat1b image;
		imaged.convertTo(image, image.type(), 255);

		if (estimated_center.x > 0 && estimated_center.y > 0) {
			cv::Mat1f imagef;
			imaged.convertTo(imagef, CV_32FC1);
			cv::RotatedRect lv_rect = get_ellipse_for_point(imagef, estimated_center);
			double R = get_circle_for_point(imagef, estimated_center);
			
			if (lv_rect.size.width > 2.5 * lv_rect.size.height) {
				lv_rect.size.width = 1.5 * lv_rect.size.height;
			} else if (lv_rect.size.height > 2.5 * lv_rect.size.width) {
				lv_rect.size.height = 1.5 * lv_rect.size.width;
			}
			cv::Rect lv_brect(lv_rect.center - cv::Point2f(lv_rect.size.width, lv_rect.size.height)/2, lv_rect.size);// = lv_rect.boundingRect();


			bbox.start_x = estimated_center.x - R * 1.1;
			bbox.start_y = estimated_center.y - R * 1.1;
			bbox.width =  2 * R * 1.1;
			bbox.height = 2 * R * 1.1;
			bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
			bbox.centroid_y = bbox.start_y + bbox.height / 2.0;

			ground_truth_shapes.push_back(landmarks);

			images.push_back(image);
			bounding_box.push_back(bbox);

			if (show_train) {
				cv::Mat test_image_1 = images.back().clone();
				cv::cvtColor(test_image_1, test_image_1, CV_GRAY2BGR);
				double scale = 1;// 256. / test_image_1.cols;
				cv::resize(test_image_1, test_image_1, cv::Size(), scale, scale);

				cv::putText(test_image_1, image_name, cv::Point(15, 15), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));
				for (int i = 0; i < landmark_num; i++) {
					circle(test_image_1, cv::Point2d(landmarks(i, 0), landmarks(i, 1))*scale, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
				}

				cv::Rect roi(bbox.start_x*scale, bbox.start_y*scale, bbox.width*scale, bbox.height*scale);
				cv::rectangle(test_image_1, roi, cv::Scalar(255, 0, 0));

				cv::circle(test_image_1, estimated_center, R, cv::Scalar(255, 0, 255));

				imshow("gt", test_image_1);
				int key = cv::waitKey(0);
				show_train = key != 'q';
			}
			int key = cv::waitKey(1);
		}
	}
	fin.close();
#endif

	ShapeRegressor regressor;
	regressor.Train(images, ground_truth_shapes, bounding_box, first_level_num, second_level_num, candidate_pixel_num, fern_pixel_num, initial_number);
#if 1
	regressor.Save("cpr_model_ch2.txt");
#else
	regressor.Save("cpr_model_circled_kmeans_smooth.txt");
#endif
	return 0;
}

