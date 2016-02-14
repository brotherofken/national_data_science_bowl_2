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

#include "FaceAlignment.h"


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

int main() {
	const int img_num = 500; // WHATTA
	const int candidate_pixel_num = 100; // TOO MUCH
	const int fern_pixel_num = 5;
	const int first_level_num = 10; // cascades
	const int second_level_num = 250; // trees per cascade
	const int landmark_num = 16;
	const int initial_number = 20;
	bool show_train = true;

	std::vector<cv::Mat1b> images;
	std::vector<BoundingBox> bbox;

	std::cout << "Read images..." << std::endl;
	std::vector<cv::Mat1d> ground_truth_shapes;
	std::vector<BoundingBox> bounding_box;
	std::ifstream fin("dataset/lv_keypointse16_train.txt");
	for (int i = 0; i < img_num; i++) {
		std::string image_name;
		BoundingBox bbox;
		fin >> image_name >> bbox.start_x >> bbox.start_y >> bbox.width >> bbox.height;
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


		cv::Mat1d imaged = read_dcm(image_name);
		cv::Mat hist;
		const auto minmax = get_quantile_uchar(imaged, hist, 0.1, 0.95);
		imaged = (imaged - minmax.first) / (minmax.second - minmax.first);
		imaged.setTo(1, imaged > 1.0);
		cv::Mat1b image;
		imaged.convertTo(image, image.type(), 255);
		images.push_back(image);

		bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
		bbox.centroid_y = bbox.start_y + bbox.height / 2.0;
		bounding_box.push_back(bbox);
		
		{
			cv::Rect roi(bbox.start_x, bbox.start_y, bbox.width, bbox.height);
			cv::Mat1d sample;
			cv::resize(imaged(roi) * 255, sample, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);
			cv::imwrite("samples/" + std::to_string(i) + ".png", sample);
		}
		cv::Mat1d landmarks(landmark_num, 2);
		for (int j = 0; j < landmark_num; j++) {
			fin >> landmarks(j, 0) >> landmarks(j, 1);
		}

		ground_truth_shapes.push_back(landmarks);

		if (show_train) {
			cv::Mat test_image_1 = images.back().clone();
			cv::cvtColor(test_image_1, test_image_1, CV_GRAY2BGR);
			double scale = 512. / test_image_1.cols;
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
	fin.close();

	ShapeRegressor regressor;
	regressor.Train(images, ground_truth_shapes, bounding_box, first_level_num, second_level_num, candidate_pixel_num, fern_pixel_num, initial_number);
	regressor.Save("model.txt");

	return 0;
}

