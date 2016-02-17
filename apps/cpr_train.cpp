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

#include "cpr/FaceAlignment.h"
#include "hog_lv_detector.hpp"
#include "dicom_reader.hpp"

int main() {
	const int img_num = 500; // WHATTA
	const int candidate_pixel_num = 300; // TOO MUCH
	const int fern_pixel_num = 5;
	const int first_level_num = 10; // cascades
	const int second_level_num = 250; // trees per cascade
	const int landmark_num = 16;
	const int initial_number = 40;
	bool show_train = true;

	std::vector<cv::Mat1b> images;
	std::vector<BoundingBox> bbox;

	//HogLvDetector lv_detector;

	std::cout << "Read images..." << std::endl;
	std::vector<cv::Mat1d> ground_truth_shapes;
	std::vector<BoundingBox> bounding_box;
	std::ifstream fin("dataset/lv_keypointse16_train.txt");

	for (int i = 0; i < img_num; i++) {
		std::string image_name;
		BoundingBox bbox;
		fin >> image_name >> bbox.start_x >> bbox.start_y >> bbox.width >> bbox.height;

		// Read image
		Slice slice(image_name);
		bbox.start_x *= slice.pixel_spacing[0];
		bbox.start_x *= slice.pixel_spacing[1];
		bbox.width *= slice.pixel_spacing[0];
		bbox.height *= slice.pixel_spacing[1];

		if (bbox.width > bbox.height) {
			bbox.start_y -= (bbox.width - bbox.height) / 2;
			bbox.height = bbox.width;
		} else {
			bbox.start_x -= (bbox.height - bbox.width) / 2;
			bbox.width = bbox.height;
		}
		bbox.start_x -= 0.15 * bbox.width;
		bbox.start_y -= 0.15 * bbox.height;
		bbox.width *= 1.3;
		bbox.height *= 1.3;
		bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
		bbox.centroid_y = bbox.start_y + bbox.height / 2.0;
		
		cv::Mat1d landmarks(landmark_num, 2);
		for (int j = 0; j < landmark_num; j++) {
			fin >> landmarks(j, 0) >> landmarks(j, 1);
			landmarks(j, 0) *= slice.pixel_spacing[0];
			landmarks(j, 1) *= slice.pixel_spacing[1];
		}
		ground_truth_shapes.push_back(landmarks);
		//// Caculate contour mean point
		cv::Point2d mean_point(cv::mean(landmarks.col(0))[0], cv::mean(landmarks.col(1))[0]);

		double R = 0;
		for (int j{}; j < landmark_num; ++j) {
			R += cv::norm(mean_point - cv::Point2d(landmarks(j, 0), landmarks(j, 1)));
		}
		R /= landmark_num;

		

		cv::Mat1d imaged = slice.image.clone();
		cv::resize(imaged, imaged, cv::Size(), slice.pixel_spacing[0], slice.pixel_spacing[1], cv::INTER_CUBIC);

		cv::Mat1b image;
		imaged.convertTo(image, image.type(), 255);
		images.push_back(image);
		//
		//cv::Rect2d lv_rect = lv_detector.detect(imaged, mean_point, true);
		//BoundingBox lv_bbox = { lv_rect.x, lv_rect.y, lv_rect.width, lv_rect.height, lv_rect.x + lv_rect.width / 2.0, lv_rect.y + lv_rect.height / 2.0 };
		//bbox = lv_rect.area() > 0 ? lv_bbox : bbox;
		//
		const int max_dim = std::min(image.cols * slice.pixel_spacing[0], image.rows * slice.pixel_spacing[1]);
		bbox.start_x = mean_point.x - 0.05 * max_dim;
		bbox.start_y = mean_point.y - 0.05 * max_dim;
		bbox.width = 0.1 * max_dim;
		bbox.height = 0.1 * max_dim;
		bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
		bbox.centroid_y = bbox.start_y + bbox.height / 2.0;

		bounding_box.push_back(bbox);

		if (show_train) {
			cv::Mat test_image_1 = images.back().clone();
			cv::cvtColor(test_image_1, test_image_1, CV_GRAY2BGR);
			double scale = 256. / test_image_1.cols;
			cv::resize(test_image_1, test_image_1, cv::Size(), scale, scale);

			cv::putText(test_image_1, image_name, cv::Point(15, 15), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));
			for (int i = 0; i < landmark_num; i++) {
				circle(test_image_1, cv::Point2d(landmarks(i, 0), landmarks(i, 1))*scale, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
			}
			cv::Rect roi(bbox.start_x*scale, bbox.start_y*scale, bbox.width*scale, bbox.height*scale);
			cv::rectangle(test_image_1, roi, cv::Scalar(255, 0, 0));

			cv::circle(test_image_1, mean_point*scale, R*scale, cv::Scalar(255, 0, 255));
			cv::circle(test_image_1, mean_point*scale, 1, cv::Scalar(255, 0, 255));

			imshow("gt", test_image_1);
			int key = cv::waitKey(0);
			show_train = key != 'q';
		}
	}
	fin.close();

	ShapeRegressor regressor;
	regressor.Train(images, ground_truth_shapes, bounding_box, first_level_num, second_level_num, candidate_pixel_num, fern_pixel_num, initial_number);
	regressor.Save("cpr_model_circled.txt");

	return 0;
}

