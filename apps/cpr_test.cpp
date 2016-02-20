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

#include "cpr/FaceAlignment.h"
#include "hog_lv_detector.hpp"
#include "dicom_reader.hpp"
#include "contour_extraction.hpp"

using namespace std;
using namespace cv;
namespace bfs = boost::filesystem;

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
		BoundingBox bbox;
		fin >> image_name >> bbox.start_x >> bbox.start_y >> bbox.width >> bbox.height;
		names.push_back(image_name);
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
			//cv::RotatedRect lv_rect = get_ellipse_for_point(imagef, estimated_center);
			double R = get_circle_for_point(imagef, estimated_center);

			//if (lv_rect.size.width > 2.5 * lv_rect.size.height) {
			//	lv_rect.size.width = 1.5 * lv_rect.size.height;
			//}
			//else if (lv_rect.size.height > 2.5 * lv_rect.size.width) {
			//	lv_rect.size.height = 1.5 * lv_rect.size.width;
			//}
			//cv::Rect lv_brect(lv_rect.center - cv::Point2f(lv_rect.size.width, lv_rect.size.height) / 2, lv_rect.size);// = lv_rect.boundingRect();


			bbox.start_x = estimated_center.x - R * 1.1;
			bbox.start_y = estimated_center.y - R * 1.1;
			bbox.width = 2 * R * 1.1;
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

				//lv_rect.center *= scale;
				//lv_rect.size.width *= scale;
				//lv_rect.size.height *= scale;
				//cv::ellipse(test_image_1, lv_rect, cv::Scalar(255, 0, 0));

				cv::circle(test_image_1, estimated_center, R, cv::Scalar(255, 0, 255));

				imshow("gt", test_image_1);
				int key = cv::waitKey(0);
				show_train = key != 'q';
			}
			int key = cv::waitKey(1);
		}
	}
	fin.close();

    ShapeRegressor regressor;
    regressor.Load("cpr_model_circled_kmeans_smooth.txt");
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


