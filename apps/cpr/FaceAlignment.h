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


#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>   
#include <utility> 

#define BINARY_IO

struct BoundingBox{
        double start_x;
        double start_y;
        double width;
        double height;
        double centroid_x;
        double centroid_y;
};


struct FeatureExtractor
{
	template <class T> inline T clamp(const T& v, const T& v_min, const T& v_max) { return std::max(v_min, std::min(v, v_max)); }
	cv::Mat1i selected_nearest_landmark_index_;
	cv::Mat1d threshold_;

	// the index of selected pixels pairs in fern
	std::vector<cv::Point2d> selected_pixel_index_;

	// the locations of selected pixel pairs stored in the format (x_1,y_1)(x_2,y_2) for each row 
	std::pair<cv::Mat1d, cv::Mat1d> selected_pixel_locations_;

	std::vector<bool> feature_vector(const cv::Mat_<uchar>& image, const cv::Mat1d& shape, const cv::Mat1d& rotation, const BoundingBox& bounding_box, const double scale)
	{
		const auto get_intensity = [&](const cv::Mat1d& pixel_locations, const int i, const bool is_y_coord) {
			const int& nearest_landmark_index = selected_nearest_landmark_index_(i, is_y_coord ? 1 : 0);
			const double& x = pixel_locations(i, 0);
			const double& y = pixel_locations(i, 1);
			const double project_x = scale * (rotation(0, 0) * x + rotation(0, 1) * y) * bounding_box.width / 2.0 + shape(nearest_landmark_index, 0);
			const double project_y = scale * (rotation(1, 0) * x + rotation(1, 1) * y) * bounding_box.height / 2.0 + shape(nearest_landmark_index, 1);
			const double bounded_project_x = clamp(project_x, 0.0, image.cols - 1.0);
			const double bounded_project_y = clamp(project_y, 0.0, image.rows - 1.0);
			const double intensity = double(image(bounded_project_y, bounded_project_x));
			return intensity;
		};
	}
};

class Fern
{
	int fern_pixel_num_;
	int landmark_num_;

	cv::Mat1i selected_nearest_landmark_index_;
	cv::Mat1d threshold_;

	// fern_pixel_num*2 matrix, the index of selected pixels pairs in fern
	//cv::Mat1i selected_pixel_index_;
	std::vector<cv::Point2d> selected_pixel_index_;

	// fern_pixel_num*4 matrix, the locations of selected pixel pairs stored in the format (x_1,y_1,x_2,y_2) for each row 
	//cv::Mat1d selected_pixel_locations_;
	std::pair<cv::Mat1d, cv::Mat1d> selected_pixel_locations_;

	std::vector<cv::Mat1d> bin_output_;
public:
	std::vector<cv::Mat1d> Train(const std::vector<std::vector<double> >& candidate_pixel_intensity,
		const cv::Mat1d& covariance,
		const cv::Mat1d& candidate_pixel_locations,
		const cv::Mat1i& nearest_landmark_index,
		const std::vector<cv::Mat1d >& regression_targets,
		int fern_pixel_num);

	void FeatureSelection(int fern_pixel_num,
		const std::vector<cv::Mat1d> &regression_targets,
		int candidate_pixel_num,
		const std::vector<std::vector<double>> &candidate_pixel_intensity,
		const cv::Mat1d& covariance,
		const cv::Mat1d& candidate_pixel_locations,
		const cv::Mat1i& nearest_landmark_index);

	const cv::Mat1d& Predict(const cv::Mat_<uchar>& image,
		const cv::Mat1d& shape,
		const cv::Mat1d& rotation,
		const BoundingBox& bounding_box,
		const double scale) const;
	void Read(std::istream& fin);
	void Write(std::ostream& fout);
};

class FernCascade{
    public:
        std::vector<cv::Mat_<double> > Train(const std::vector<cv::Mat_<uchar> >& images,
                                             const std::vector<cv::Mat_<double> >& current_shapes,
                                             const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                                             const std::vector<BoundingBox> & bounding_box,
                                             const cv::Mat_<double>& mean_shape,
                                             int second_level_num,
                                             int candidate_pixel_num,
                                             int fern_pixel_num,
                                             int curr_level_num,
                                             int first_level_num);  
        cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, 
                                 const BoundingBox& bounding_box, 
                                 const cv::Mat_<double>& mean_shape,
                                 const cv::Mat_<double>& shape);
        void Read(std::istream& fin);
        void Write(std::ostream& fout);
    private:
        std::vector<Fern> ferns_;
        int second_level_num_;
};

class ShapeRegressor{
    public:
        ShapeRegressor(); 
        void Train(const std::vector<cv::Mat_<uchar> >& images, 
                   const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                   const std::vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num);
        cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num, const cv::Mat1d& initial_contour = cv::Mat1d());
        void Read(std::istream& fin);
        void Write(std::ostream& fout);
        void Load(std::string path);
        void Save(std::string path);
    private:
        int first_level_num_;
        int landmark_num_;
        std::vector<FernCascade> fern_cascades_;
        cv::Mat_<double> mean_shape_;
        std::vector<cv::Mat_<double> > training_shapes_;
        std::vector<BoundingBox> bounding_box_;
};

cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& shapes,
                              const std::vector<BoundingBox>& bounding_box);
cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<double>& shape1, const cv::Mat_<double>& shape2, 
                         cv::Mat_<double>& rotation,double& scale);
double calculate_covariance(const std::vector<double>& v_1, 
                            const std::vector<double>& v_2);

namespace io {

	//void draw_landmarks(cv::Mat3b& image, const cv::Mat1d& landmarks, const double thickness, const cv::Scalar& color);

	// TODO: Code is not portable, just fast error-prone solution
	template<class T>
	void write_scalar(std::ostream& o, const T& s)
	{
		o.write((char*)&s, sizeof(T));
	}
	template<class T>
	void read_scalar(std::istream& i, T& s)
	{
		i.read((char*)&s, sizeof(T));
	}

	template<class T>
	void write_mat(std::ostream& o, cv::Mat_<T>& m)
	{
		static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value || std::is_same<T, int>::value || std::is_same<T, char>::value, "");

		io::write_scalar(o, m.cols);
		io::write_scalar(o, m.rows);
		const int m_bytesize = m.total() * m.elemSize();
		o.write(reinterpret_cast<char*>(m.data), m_bytesize);
	}

	template<class T>
	void read_mat(std::istream& i, cv::Mat_<T>& m)
	{
		static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value || std::is_same<T, int>::value || std::is_same<T, char>::value, "");

		io::read_scalar(i, m.cols);
		io::read_scalar(i, m.rows);
		m.create(m.rows, m.cols);

		const int m_bytesize = m.cols * m.rows * m.elemSize();
		i.read(reinterpret_cast<char*>(m.data), m_bytesize);
	}

	template<class T>
	void write_vector(std::ostream& o, const std::vector<T>& v)
	{
		static_assert(!std::is_pointer<T>::value, "");
		io::write_scalar(o, v.size());
		o.write((char*)&v[0], v.size() * sizeof(T));
	}

	template<class T>
	void read_vector(std::istream& i, std::vector<T>& v)
	{
		static_assert(!std::is_pointer<T>::value, "");
		size_t size;
		io::read_scalar(i, size);
		v.resize(size);
		i.read((char*)&v[0], v.size() * sizeof(T));
	}

}

#endif
