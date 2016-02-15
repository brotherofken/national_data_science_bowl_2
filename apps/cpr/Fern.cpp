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

#include <omp.h>

namespace {
	template <class T> inline T clamp(const T& v, const T& v_min, const T& v_max)
	{
		return std::max(v_min, std::min(v, v_max));
	}
}


void Fern::FeatureSelection(int fern_pixel_num, const std::vector<cv::Mat1d> &regression_targets, int candidate_pixel_num, const std::vector<std::vector<double>> &candidate_pixel_intensity, const cv::Mat1d& covariance, const cv::Mat1d& candidate_pixel_locations, const cv::Mat1i& nearest_landmark_index)
{
	this->selected_pixel_index_.resize(fern_pixel_num);

	this->selected_pixel_locations_.first.create(fern_pixel_num, 2);
	this->selected_pixel_locations_.second.create(fern_pixel_num, 2);

	this->selected_nearest_landmark_index_.create(fern_pixel_num, 2);
	// threshold_: thresholds for each pair of pixels in fern 
	this->threshold_.create(fern_pixel_num, 1);

	// Algorithm 3: CorrelationBasedFeatureSelection
	// select pixel pairs from candidate pixels, this selection is based on the correlation between pixel 
	// densities and regression targets
	// for details, please refer to "Face Alignment by Explicit Shape Regression" 


	// get a random direction
	cv::RNG random_generator(cv::getTickCount());

#pragma omp parallel for
	for (int i = 0; i < fern_pixel_num; i++) {
		// draw random projection
		cv::Mat1d random_direction(landmark_num_, 2);

		// TODO: Uniform distribution was changed to normal. See point 2.5 paper 
		//random_generator.fill(random_direction, cv::RNG::UNIFORM, -1.1, 1.1);
		random_generator.fill(random_direction, cv::RNG::NORMAL, 0.0, 1.0);

		normalize(random_direction, random_direction);
		std::vector<double> projection_result(regression_targets.size(), 0);

		// random projection
		for (int j = 0; j < regression_targets.size(); j++) {
			projection_result[j] = regression_targets[j].dot(random_direction); //sum(regression_targets[j].mul(random_direction))[0];
		}

		// compute target-pixel covariance
		cv::Mat1d covariance_projection_density(candidate_pixel_num, 1);
		for (int j = 0; j < candidate_pixel_num; j++) {
			covariance_projection_density(j) = calculate_covariance(projection_result, candidate_pixel_intensity[j]);
		}

		// find max correlation
		double max_correlation = -1;
		int max_pixel_index_1 = 0;
		int max_pixel_index_2 = 0;

		for (int m = 0; m < candidate_pixel_num; ++m) {
			for (int n = 0; n < candidate_pixel_num; ++n) {
				const double denominator = std::sqrt(covariance(m, m) + covariance(n, n) - 2 * covariance(m, n));

				if (abs(denominator) < 1e-10) continue;

				// Skip if pixel was selected already
				bool flag = false;
				for (int p = 0; p < i; p++) {
					if (m == selected_pixel_index_[p].x && n == selected_pixel_index_[p].y ||
						m == selected_pixel_index_[p].y && n == selected_pixel_index_[p].x) {
						flag = true;
						break;
					}
				}
				if (flag) continue;

				const double correlation = (covariance_projection_density(m) - covariance_projection_density(n)) / denominator;
				if (abs(correlation) > max_correlation) {
					max_correlation = correlation;
					max_pixel_index_1 = m;
					max_pixel_index_2 = n;
				}
			}
		}

		selected_pixel_index_[i].x = max_pixel_index_1;
		selected_pixel_index_[i].y = max_pixel_index_2;
		selected_pixel_locations_.first(i, 0) = candidate_pixel_locations(max_pixel_index_1, 0);
		selected_pixel_locations_.first(i, 1) = candidate_pixel_locations(max_pixel_index_1, 1);
		selected_pixel_locations_.second(i, 0) = candidate_pixel_locations(max_pixel_index_2, 0);
		selected_pixel_locations_.second(i, 1) = candidate_pixel_locations(max_pixel_index_2, 1);
		selected_nearest_landmark_index_(i, 0) = nearest_landmark_index(max_pixel_index_1);
		selected_nearest_landmark_index_(i, 1) = nearest_landmark_index(max_pixel_index_2);

		// get threshold for this pair
		double max_diff = -1;
		for (int j = 0; j < candidate_pixel_intensity[max_pixel_index_1].size(); j++) {
			const double diff = std::abs(candidate_pixel_intensity[max_pixel_index_1][j] - candidate_pixel_intensity[max_pixel_index_2][j]);
			if (diff > max_diff) {
				max_diff = diff;
			}
		}

		threshold_(i) = random_generator.uniform(-0.2*max_diff, 0.2*max_diff);
	}
}

std::vector<cv::Mat1d> Fern::Train(const std::vector<std::vector<double>>& candidate_pixel_intensity,
	const cv::Mat1d& covariance,
	const cv::Mat1d& candidate_pixel_locations,
	const cv::Mat1i& nearest_landmark_index,
	const std::vector<cv::Mat1d>& regression_targets,
	int fern_pixel_num)
{
	// Algorithm 4

	fern_pixel_num_ = fern_pixel_num;
	landmark_num_ = regression_targets[0].rows;
	int candidate_pixel_num = candidate_pixel_locations.rows;
	FeatureSelection(fern_pixel_num, regression_targets, candidate_pixel_num, candidate_pixel_intensity, covariance, candidate_pixel_locations, nearest_landmark_index);

	// determine the bins of each shape
	const int bin_num = std::pow(2, fern_pixel_num);
	std::vector<std::vector<int>> shapes_in_bin(bin_num);

	for (int i = 0; i < regression_targets.size(); i++) {
		int index = 0;
		for (int j = 0; j < fern_pixel_num; j++) {
			const double value_1(candidate_pixel_intensity[selected_pixel_index_[j].x][i]);
			const double value_2(candidate_pixel_intensity[selected_pixel_index_[j].y][i]);
			if (value_1 - value_2 >= threshold_(j)) {
				index = index + pow(2, j);
			}
		}
		shapes_in_bin[index].push_back(i);
	}

	// get bin output
	std::vector<cv::Mat1d > prediction;
	prediction.resize(regression_targets.size());
	bin_output_.resize(bin_num);
	for (int i = 0; i < bin_num; i++) {
		cv::Mat1d target = cv::Mat::zeros(landmark_num_, 2, CV_64FC1);
		const int bin_size(shapes_in_bin[i].size());
		for (int j = 0; j < bin_size; j++) {
			const int index = shapes_in_bin[i][j];
			target = target + regression_targets[index];
		}
		if (bin_size == 0) {
			bin_output_[i] = target;
			continue;
		}

		// Shrinkage
		// TODO: look for better value for beta?
		const double beta = 1000.0;
		target = target / ((1.0 + beta / bin_size) * bin_size);
		bin_output_[i] = target;
		for (int j = 0; j < bin_size; j++) {
			int index = shapes_in_bin[i][j];
			prediction[index] = target;
		}
	}
	return prediction;
}


const cv::Mat1d& Fern::Predict(const cv::Mat1b& image,
	const cv::Mat1d& shape,
	const cv::Mat1d& rotation,
	const BoundingBox& bounding_box,
	const double scale) const
{
	const auto get_intensity = [&](const cv::Mat1d& pixel_locations, const int i, const bool is_y_coord) {
		const int& nearest_landmark_index = selected_nearest_landmark_index_(i, is_y_coord ? 1 : 0);
		const double& x = pixel_locations(i, 0);
		const double& y = pixel_locations(i, 1);
		const double project_x = scale * (rotation(0, 0) * x + rotation(0, 1) * y) * bounding_box.width / 2.0 + shape(nearest_landmark_index, 0);
		const double project_y = scale * (rotation(1, 0) * x + rotation(1, 1) * y) * bounding_box.height / 2.0 + shape(nearest_landmark_index, 1);
		const double bounded_project_x = ::clamp(project_x, 0.0, image.cols - 1.0);
		const double bounded_project_y = ::clamp(project_y, 0.0, image.rows - 1.0);
		const double intensity = double(image(bounded_project_y, bounded_project_x));
		return intensity;
	};

	int index = 0;
	for (int i = 0; i < fern_pixel_num_; i++) {
		const double intensity_1 = get_intensity(selected_pixel_locations_.first, i, false);
		const double intensity_2 = get_intensity(selected_pixel_locations_.second, i, true);

		if (intensity_1 - intensity_2 >= threshold_(i)) {
			index += std::pow(2, i);
		}
	}
	return bin_output_[index];
}



void Fern::Write(std::ostream& fout) {
	io::write_scalar(fout, fern_pixel_num_);
	io::write_scalar(fout, landmark_num_);

	cv::Mat1d selected_pixel_locations_m(fern_pixel_num_, 4);
	for (int i = 0; i < selected_pixel_locations_m.rows; ++i) {
		selected_pixel_locations_m(i, 0) = selected_pixel_locations_.first(i, 0);
		selected_pixel_locations_m(i, 1) = selected_pixel_locations_.first(i, 1);
		selected_pixel_locations_m(i, 2) = selected_pixel_locations_.second(i, 0);
		selected_pixel_locations_m(i, 3) = selected_pixel_locations_.second(i, 1);
	}


	io::write_mat(fout, selected_pixel_locations_m);

	io::write_mat(fout, selected_nearest_landmark_index_);
	io::write_mat(fout, threshold_);

	for (int i = 0; i < bin_output_.size(); i++) {
		io::write_mat(fout, bin_output_[i]);
	}
}

void Fern::Read(std::istream& fin) {
	io::read_scalar(fin, fern_pixel_num_);
	io::read_scalar(fin, landmark_num_);

	selected_nearest_landmark_index_.create(fern_pixel_num_, 2);
	cv::Mat1d selected_pixel_locations_m(fern_pixel_num_, 4);

	threshold_.create(fern_pixel_num_, 1);

	io::read_mat(fin, selected_pixel_locations_m);
	selected_pixel_locations_.first.create(fern_pixel_num_, 2);
	selected_pixel_locations_.second.create(fern_pixel_num_, 2);

	for (int i = 0; i < selected_pixel_locations_m.rows; ++i) {
		selected_pixel_locations_.first(i, 0) = selected_pixel_locations_m(i, 0);
		selected_pixel_locations_.first(i, 1) = selected_pixel_locations_m(i, 1);
		selected_pixel_locations_.second(i, 0) = selected_pixel_locations_m(i, 2);
		selected_pixel_locations_.second(i, 1) = selected_pixel_locations_m(i, 3);
	}

	io::read_mat(fin, selected_nearest_landmark_index_);
	io::read_mat(fin, threshold_);

	int bin_num = pow(2.0, fern_pixel_num_);
	bin_output_.resize(bin_num);
	for (int i = 0; i < bin_output_.size(); i++) {
		io::read_mat(fin, bin_output_[i]);
	}
}
