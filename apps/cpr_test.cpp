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

#include <opencv2/ml.hpp>

#include "cpr/FaceAlignment.h"
#include "hog_lv_detector.hpp"
#include "dicom_reader.hpp"

using namespace std;
using namespace cv;

int main(){
	vector<string> names;
    vector<Mat1b> images;
    vector<BoundingBox> bounding_box;
	std::vector<cv::Mat1d> ground_truth_shapes;
    int img_num = 49;
    int initial_number = 20;
    int landmark_num = 16;

	std::ifstream fin("dataset/lv_keypointse16_test.txt");

	//HogLvDetector lv_detector;

	bool show_train = true;

	for (int i = 0; i < img_num; i++) {
		std::string image_name;
		names.push_back(image_name);
		BoundingBox bbox;
		fin >> image_name >> bbox.start_x >> bbox.start_y >> bbox.width >> bbox.height;
		if (bbox.width > bbox.height) {
			bbox.start_y -= (bbox.width - bbox.height) / 2;
			bbox.height = bbox.width;
		}
		else {
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
		}
		ground_truth_shapes.push_back(landmarks);

		// Read image
		Slice slice(image_name);
		cv::Mat1d imaged = slice.image.clone();

		cv::Mat1b image;
		imaged.convertTo(image, image.type(), 255);
		images.push_back(image);

		// Caculate contour mean point
		cv::Scalar mean_point_tmp = cv::mean(landmarks.reshape(2, landmark_num / 2));
		cv::Point mean_point(mean_point_tmp[0], mean_point_tmp[1]);

		//cv::Rect2d lv_rect = lv_detector.detect(imaged, mean_point, true);
		//BoundingBox lv_bbox = { lv_rect.x, lv_rect.y, lv_rect.width, lv_rect.height, lv_rect.x + lv_rect.width / 2.0, lv_rect.y + lv_rect.height / 2.0 };
		//bbox = lv_rect.area() > 0 ? lv_bbox : bbox;

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
			double scale = 1;//256. / std::max(test_image_1.cols, test_image_1.rows);
			cv::resize(test_image_1, test_image_1, cv::Size(), scale, scale);

			cv::putText(test_image_1, image_name, cv::Point(15, 15), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));
			for (int i = 0; i < landmark_num; i++) {
				circle(test_image_1, cv::Point2d(landmarks(i, 0), landmarks(i, 1))*scale, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
			}
			cv::Rect roi(bbox.start_x*scale, bbox.start_y*scale, bbox.width*scale, bbox.height*scale);
			cv::rectangle(test_image_1, roi, cv::Scalar(255, 0, 0));
			imshow("gt", test_image_1);
			int key = cv::waitKey(0);
			show_train = key != 'q';
		}
	}
	cv::destroyWindow("gt");
	fin.close();

    ShapeRegressor regressor;
    regressor.Load("cpr_model_circled.txt");
	int index = 0;
	int key = 0;
    while (key != 'q') {
        Mat_<double> current_shape = regressor.Predict(images[index], bounding_box[index],initial_number);
        Mat test_image_1 = images[index].clone();

		cv::cvtColor(test_image_1, test_image_1, CV_GRAY2BGR);
		cv::Mat1d gt_shape = ground_truth_shapes[index];

		double scale = 512. / test_image_1.cols;
		cv::resize(test_image_1, test_image_1, cv::Size(), scale, scale);
		cv::putText(test_image_1, std::to_string(index) + "/" + std::to_string(images.size()) + " " + names[index], cv::Point(15, 15), CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 0));

		for (int i = 0; i < landmark_num; i++) {
			if (!gt_shape.empty()) {
				//cv::line(test_image_1, Point2d(current_shape(i, 0), current_shape(i, 1))*scale, Point2d(gt_shape(i, 0), gt_shape(i, 1))*scale, Scalar(255, 0, 0));
				circle(test_image_1, Point2d(gt_shape(i, 0), gt_shape(i, 1))*scale, 1, Scalar(0, 255, 0), -1, 8, 0);
			}
			circle(test_image_1, Point2d(current_shape(i, 0), current_shape(i, 1))*scale, 1, Scalar(0, 0, 255), -1, 8, 0);
		}
		cv::Rect roi(bounding_box[index].start_x*scale, bounding_box[index].start_y*scale, bounding_box[index].width*scale, bounding_box[index].height*scale);
		cv::rectangle(test_image_1, roi, cv::Scalar(255, 0, 0));

        imshow("result",test_image_1);
		
		imwrite("results/"+std::to_string(index)+".png", test_image_1(roi));
        key = waitKey(0);
		index = (index + 1) % images.size();
    }
    return 0;
}


