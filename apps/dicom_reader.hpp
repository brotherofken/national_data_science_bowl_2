#pragma once

#include <opencv2/opencv.hpp>

struct OrientedObject
{
	double slice_location;
	double slice_thickness;
	cv::Vec3d row_dc;
	cv::Vec3d col_dc;
	cv::Vec3d position;
	cv::Vec3d pixel_spacing;
	cv::Mat1d rm;

	inline cv::Mat1d rotation_matrix() const { return rm.reshape(1, 3); }

	inline cv::Vec3d normal() const { return row_dc.cross(col_dc); };

	cv::Point2d point_to_image(const cv::Vec3d& p)
	{
		const cv::Mat1d projection = rotation_matrix().inv().t() * cv::Mat1d(p - position);
		return{ projection(0, 0), projection(1,0) };
	}

	cv::Vec3d point_to_3d(const cv::Point2d& p)
	{
		const cv::Vec3d p3da{ p.x, p.y, 0. };
		const cv::Mat1d projection = rotation_matrix() * cv::Mat1d(p3da) + position;
		return cv::Vec3d(projection);
	}
};

struct Slice : public OrientedObject
{
	Slice(const std::string& filename);

	std::string filename; // relative path to image
	size_t frame_number;
	cv::Mat1d image;

	using Vector = std::vector<Slice>;
};


struct Sequence : public OrientedObject
{
	enum class Type : size_t {
		sax = 0,
		ch2 = 1,
		ch4 = 2,
	};

	Sequence() {}
	Sequence(const std::string& directory);

	Type type;
	std::string name; // 2ch_*, 4ch_* or sax_*
	size_t number; // Number that goes after _
	Slice::Vector slices;

	using Vector = std::vector<Sequence>;

	cv::Vec3d row_dc;
	cv::Vec3d col_dc;
	cv::Vec3d position;
	double slice_location;
	double slice_thickness;
	cv::Mat1d rm;
};

struct PatientData
{
	PatientData(const std::string& directory);

	size_t number;

	Sequence ch2_seq;
	Sequence ch4_seq;
	Sequence::Vector sax_seqs;
};
