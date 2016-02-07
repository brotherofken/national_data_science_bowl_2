#pragma once

#include <opencv2/opencv.hpp>

#include <utility>
#include <functional>

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

	cv::Point2d point_to_image(const cv::Vec3d& p) const
	{
		const cv::Mat1d projection = rotation_matrix().inv().t() * cv::Mat1d(p - position);
		return{ projection(0, 0), projection(1,0) };
	}

	cv::Vec3d point_to_3d(const cv::Point2d& p) const
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

	std::map<std::string, cv::Mat> aux;

	using Vector = std::vector<Slice>;
};


struct Sequence : public OrientedObject
{
	enum class Type : size_t {
		sax = 0,
		ch2 = 1,
		ch4 = 2,
	};

	Sequence() : empty(true) {}
	Sequence(const std::string& directory);

	bool empty;
	Type type;
	std::string name; // 2ch_*, 4ch_* or sax_*
	size_t number; // Number that goes after _
	Slice::Vector slices;

	using Vector = std::vector<Sequence>;
};

using line_eq_t = std::function<cv::Vec3d(double)>;

struct PatientData
{
	struct Intersection {
		line_eq_t l24;
		line_eq_t ls2;
		line_eq_t ls4;
		cv::Vec3d p;
		cv::Point2d p_sax;
		cv::Point2d p_ch2;
		cv::Point2d p_ch4;
		using Vector = std::vector<Intersection>;
	};

	PatientData(const std::string& directory);

	std::pair<double, double> get_min_max_bp_level() const;

	size_t number;

	Sequence ch2_seq;
	Sequence ch4_seq;
	Sequence::Vector sax_seqs;
	Intersection::Vector intersections;
};

line_eq_t slices_intersection(const OrientedObject& s1, const OrientedObject& s2);
cv::Vec3d slices_intersection(const OrientedObject& s1, const OrientedObject& s2, const OrientedObject& s3);

