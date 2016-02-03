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

	inline const cv::Mat1d& rotation_matrix() const { return rm; }

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
};

struct PatientData
{
	PatientData(const std::string& directory);

	size_t number;

	Sequence ch2_seq;
	Sequence ch4_seq;
	Sequence::Vector sax_seqs;
};

//

cv::Vec3d slices_intersection(const OrientedObject& s1, const OrientedObject& s2, const OrientedObject& s3)
{
	cv::Mat1d normals;
	normals.push_back(s1.normal());
	normals.push_back(s2.normal());
	normals.push_back(s3.normal());
	normals = normals.reshape(1, 3);

	cv::Mat1d d = (cv::Mat1d(3, 1) <<
		s1.normal().dot(s1.position),
		s2.normal().dot(s2.position),
		s3.normal().dot(s3.position)
		);

	cv::Mat1d intersection;
	cv::solve(normals, d, intersection, cv::DECOMP_SVD);
	return cv::Vec3d(intersection);
}

