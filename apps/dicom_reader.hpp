#pragma once

#include <opencv2/opencv.hpp>

struct Slice
{
	cv::Mat1d image;
	cv::Vec3d row_dc;
	cv::Vec3d col_dc;
	cv::Vec3d position;
	cv::Vec3d pixel_spacing;
	double slice_location;
	double slice_thicknes;

	cv::Mat1d rotation_matrix() const
	{
		cv::Mat1d rm;
		rm.push_back(row_dc * pixel_spacing[0]);
		rm.push_back(col_dc * pixel_spacing[0]);
		rm.push_back(normal() * pixel_spacing[0]);
		return rm.reshape(1, 3);
	}
	cv::Vec3d normal() const { return row_dc.cross(col_dc); };

	cv::Point2d point_to_image(const cv::Vec3d& p)
	{
		const cv::Mat1d projection = rotation_matrix().inv().t() * cv::Mat1d(p - position);
		return{ projection(0, 0), projection(1,0) };
	}

	cv::Vec3d point_to_3d(const cv::Point2d& p)
	{
		const cv::Vec3d p3da{p.x, p.y, 0.};
		const cv::Mat1d projection = rotation_matrix() * cv::Mat1d(p3da) + position;
		return cv::Vec3d(projection);
	}
};

Slice read_dcm(const std::string& filename);
