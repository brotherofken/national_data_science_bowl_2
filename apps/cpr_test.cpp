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

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

#include <gdcmImageReader.h>

cv::Mat1d read_dcm(const std::string& filename)
{
	// Read DCM
	gdcm::ImageReader ir;
	ir.SetFileName(filename.c_str());
	if (!ir.Read()) {
		return cv::Mat1d();
	}

	//std::cout << "Getting image from ImageReader..." << std::endl;

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

std::pair<double, double> get_quantile_uchar(cv::Mat &input, cv::MatND &hist, double nmin, double nmax, int channel = 0)
{
	double imin, imax;
	cv::minMaxLoc(input, &imin, &imax);

	int const hist_size = 100;// std::numeric_limits<uchar>::max() + 1;
	float const hranges[2] = { imin, imax };
	float const *ranges[] = { hranges };

	//compute and cumulate the histogram
	cv::Mat1f inputf;
	input.convertTo(inputf, inputf.type());
	cv::calcHist(&inputf, 1, &channel, cv::Mat(), hist, 1, &hist_size, ranges);
	hist /= cv::sum(hist)[0];
	auto *hist_ptr = hist.ptr<float>(0);
	for (size_t i = 1; i != hist_size; ++i) {
		hist_ptr[i] += hist_ptr[i - 1];
	}

	// get the new min/max
	std::pair<size_t, size_t> min_max(0, hist_size - 1);
	while (min_max.first != (hist_size - 1) && hist_ptr[min_max.first] <= nmin) {
		++min_max.first; // the corresponding histogram value is the current cell position
	}

	while (min_max.second > 0 && hist_ptr[min_max.second] > nmax) {
		--min_max.second; // the corresponding histogram value is the current cell position
	}

	if (min_max.second < hist_size - 2)
		++min_max.second;

	min_max = { imin * (min_max.first / 100.), imax * (min_max.second / 100.) };

	return min_max;
}

int main(){
	vector<string> names;
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_box;
	std::vector<cv::Mat1d> ground_truth_shapes;
    int test_img_num = 49;
    int initial_number = 20;
    int landmark_num = 16;

	std::ifstream fin("dataset/lv_keypointse16_test.txt");

	bool load_landmarks = true;

	for (int i = 0; i < test_img_num; i++) {
		std::string image_name;
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

		names.push_back(image_name);
		cv::Mat1d imaged = read_dcm(image_name);
		cv::Mat hist;
		const auto minmax = get_quantile_uchar(imaged, hist, 0.1, 0.95);
		imaged = (imaged - minmax.first) / (minmax.second - minmax.first);
		imaged.setTo(1, imaged > 1.0);
		cv::Mat1b image;
		imaged.convertTo(image, image.type(), 255);
		test_images.push_back(image);

		bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
		bbox.centroid_y = bbox.start_y + bbox.height / 2.0;
		test_bounding_box.push_back(bbox);

		cv::Mat1d landmarks(landmark_num, 2);
		for (int j = 0; j < landmark_num; j++) {
			fin >> landmarks(j, 0) >> landmarks(j, 1);
		}
		if (load_landmarks) {
			ground_truth_shapes.push_back(landmarks);
		}
	}
	fin.close();

	cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();


    ShapeRegressor regressor;
    regressor.Load("model.txt");
	int index = 0;
	int key = 0;
    while (key != 'q') {
        Mat_<double> current_shape = regressor.Predict(test_images[index],test_bounding_box[index],initial_number);
        Mat test_image_1 = test_images[index].clone();

		cv::cvtColor(test_image_1, test_image_1, CV_GRAY2BGR);
		cv::Mat1d gt_shape = load_landmarks ? ground_truth_shapes[index] : cv::Mat1d();

		double scale = 1024. / test_image_1.cols;
		cv::resize(test_image_1, test_image_1, cv::Size(), scale, scale);
		cv::putText(test_image_1, std::to_string(index) + "/" + std::to_string(test_images.size()) + " " + names[index], cv::Point(15, 15), CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 0));

		for (int i = 0; i < landmark_num; i++) {
			if (!gt_shape.empty()) {
				//cv::line(test_image_1, Point2d(current_shape(i, 0), current_shape(i, 1))*scale, Point2d(gt_shape(i, 0), gt_shape(i, 1))*scale, Scalar(255, 0, 0));
				circle(test_image_1, Point2d(gt_shape(i, 0), gt_shape(i, 1))*scale, 1, Scalar(0, 255, 0), -1, 8, 0);
			}
			circle(test_image_1, Point2d(current_shape(i, 0), current_shape(i, 1))*scale, 1, Scalar(0, 0, 255), -1, 8, 0);
		}
		cv::Rect roi(test_bounding_box[index].start_x*scale, test_bounding_box[index].start_y*scale, test_bounding_box[index].width*scale, test_bounding_box[index].height*scale);
		cv::rectangle(test_image_1, roi, cv::Scalar(255, 0, 0));

        imshow("result",test_image_1);
		
		imwrite("results/"+std::to_string(index)+".png", test_image_1(roi));
        key = waitKey(0);
		index = (index + 1) % test_images.size();
    }
    return 0;
}


