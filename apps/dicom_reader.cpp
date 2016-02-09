#include "dicom_reader.hpp"

#include <gdcmReader.h>
#include <gdcmStringFilter.h>
#include <gdcmImageReader.h>
#include <gdcmAttribute.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <numeric>
#include <algorithm>

const std::string PatientData::AUX_LV_MASK = "lv_mask";
const std::string PatientData::AUX_CONTOUR = "lv_annotation";


namespace fs = ::boost::filesystem;

namespace {
	// return the filenames of all files that have the specified extension
	// in the specified directory and all subdirectories
	std::vector<fs::path> get_all(const fs::path& root, const std::string& ext)
	{
		if (!fs::exists(root) || !fs::is_directory(root))
			return{};

		fs::recursive_directory_iterator it(root);
		fs::recursive_directory_iterator endit;

		std::vector<fs::path> ret;

		const bool get_directories = (ext == "~");
		while (it != endit) {
			auto fn = *it;
			const bool is_reg = get_directories ? fs::is_directory(fn) :fs::is_regular_file(fn);
			const auto fn_ext = get_directories ? true : (fn.path().extension() == ext);
			if (is_reg && fn_ext) ret.push_back(it->path().filename());
			++it;
		}

		return ret;
	}

	std::vector<std::string> string_split(const std::string& str, const std::string& delimeters)
	{
		std::vector<std::string> strs;
		boost::split(strs, str, boost::is_any_of(delimeters));
		return strs;
	}
}

size_t get_frame_number(const std::string& dcm_filename)
{
	const auto strs = string_split(fs::path(dcm_filename).stem().string(), "-");

	// expect one of two patterns for filename
	// IM-6720-0004.dcm
	// IM-2774-0006-0001.dcm
	//          ^    ^ This one is slice index in one position
	//           This is always slice frame number (I hope)
	if (strs.size() == 3 || strs.size() == 4) {
		return std::stoul(strs[2]);
	} else {
		std::cerr << "Error get_frame_number: " << dcm_filename << std::endl;
		throw std::runtime_error("get_frame_number error");
	}
}

// Bad style, no time for refactoring
Slice::Slice(const std::string& filename)
	: filename(filename)
	, frame_number(get_frame_number(filename))
{
	{
		// Read to image
		gdcm::ImageReader ir;
		ir.SetFileName(filename.c_str());
		if (!ir.Read()) {
			std::cerr << "Could not read: " << filename << std::endl;
			throw std::runtime_error("Shit happened");
		}

		const gdcm::Image &gimage = ir.GetImage();

		std::vector<short> vbuffer(gimage.GetBufferLength());
		gimage.GetBuffer((char*)&vbuffer[0]);

		const unsigned int size_x = gimage.GetDimensions()[0];
		const unsigned int size_y = gimage.GetDimensions()[1];
		image = cv::Mat1d(size_y, size_x);
		std::copy(vbuffer.begin(), vbuffer.end(), image.begin());

		// Read non-image fields

		row_dc = cv::Vec3d(gimage.GetDirectionCosines());
		col_dc = cv::Vec3d(gimage.GetDirectionCosines() + 3);
		position = cv::Vec3d(gimage.GetOrigin());
		pixel_spacing = cv::Vec3d(gimage.GetSpacing());
		// Rotation matrix
		rm = cv::Mat1d(0, 3);
		rm.push_back(row_dc * pixel_spacing[0]);
		rm.push_back(col_dc * pixel_spacing[1]);
		rm.push_back(normal() * pixel_spacing[0]);
		rm = rm.reshape(1, 3);
	}
	{
		gdcm::Reader reader;
		reader.SetFileName(filename.c_str());
		if (!reader.Read()) {
			std::cerr << "Could not read: " << filename << std::endl;
			throw std::runtime_error("Crap happened");
		}
		gdcm::File &file = reader.GetFile();

		gdcm::StringFilter sf;
		sf.SetFile(reader.GetFile());
		std::pair<std::string, std::string> slice_location_p = sf.ToStringPair(gdcm::Tag(0x0020, 0x1041));
		std::pair<std::string, std::string> slice_thickness_p = sf.ToStringPair(gdcm::Tag(0x0018, 0x0050));

#if 0 && defined(_DEBUG)
		std::cout << "File meta: " << filename << std::endl;
		std::cout << slice_location_p.first << " " << slice_location_p.second << std::endl;
		std::cout << slice_thickness_p.first << " " << slice_thickness_p.second << std::endl;
#endif


		slice_location = std::stod(slice_location_p.second);
		slice_thickness = std::stod(slice_thickness_p.second);
	}
}

Sequence::Sequence(const std::string& directory)
	: empty(false)
{
	name = fs::path(directory).stem().string();
	std::clog << " sequence " << name << " : ";
	const auto strs = string_split(name, "_");
	assert(strs.size() == 2);
	number = std::stoul(strs[1]);

	if (strs[0] == "2ch") { 
		type = Type::ch2;
	} else if (strs[0] == "4ch") {
		type = Type::ch4;
	} else if (strs[0] == "sax") {
		type = Type::sax;
	} else {
		std::cerr << "Sequence::Sequence : somthing wrong with directory name " << directory << std::endl;
		throw std::runtime_error("Crap happened");
	}

	// for *dcm read
	std::vector<fs::path> dcm_files = get_all(directory, ".dcm");

	for (const auto& file : dcm_files) {
		slices.push_back(Slice((directory / file).string()));
		std::clog << slices.back().frame_number << " ";
	}
	std::clog << std::endl;

	// I'm sure that all DCM in sax contain same information.
	if (slices.size()) {
		row_dc = slices[0].row_dc;
		col_dc = slices[0].col_dc;
		position = slices[0].position;
		slice_location = slices[0].slice_location;
		slice_thickness = slices[0].slice_thickness;
		rm = slices[0].rm.reshape(1, 3).clone();
		assert(rm.cols == 3 && rm.rows == 3);
	}
}

PatientData::PatientData(const std::string& directory_)
{
	directory = directory_;
	std::clog << "Reading patient " << directory << std::endl;
	number = std::stoul(fs::path(directory).stem().string());

	// for *dcm read
	fs::path sequences_location = fs::path(directory) / "study";
	std::vector<fs::path> slice_directories = get_all(sequences_location, "~");

	for (const auto& dir : slice_directories) {
		const Sequence s((sequences_location / dir).string());
		assert(s.rm.cols == 3 && s.rm.rows == 3);

		if (s.slices.size() && s.slices.size() <= 30) {
			if (s.type == Sequence::Type::sax) sax_seqs.push_back(s);
			if (s.type == Sequence::Type::ch2) ch2_seq = s;
			if (s.type == Sequence::Type::ch4) ch4_seq = s;
		}
	}

	const bool chamber_views_ok = !ch2_seq.empty && !ch4_seq.empty;

	std::sort(sax_seqs.begin(), sax_seqs.end(), [](Sequence& a, Sequence& b) { return a.slice_location < b.slice_location; });

	if (chamber_views_ok)
	for (const Sequence& sax : sax_seqs) {
		const cv::Vec3d point_3d = slices_intersection(sax, ch2_seq, ch4_seq);
		intersections.push_back({
			slices_intersection(ch2_seq, ch4_seq),
			slices_intersection(sax, ch2_seq),
			slices_intersection(sax, ch4_seq),
			point_3d,
			sax.point_to_image(point_3d),
			ch2_seq.point_to_image(point_3d),
			ch4_seq.point_to_image(point_3d)
		});
	}
}

std::pair<double, double> PatientData::get_min_max_bp_level() const
{
	const auto accumulate_ = [=](size_t i, double& value, std::function<double(double, double)> reduce_) {
		for (size_t j{}; j < ch2_seq.slices.size(); ++j) {
			if (cv::Rect(cv::Point{ 0, 0 }, ch2_seq.slices[j].image.size()).contains(intersections[i].p_ch2))
				value = reduce_(value, ch2_seq.slices[j].image(intersections[i].p_ch2));// cv::mean(ch2_seq.slices[j].image(cv::Rect(intersections[i].p_ch2 - cv::Point2d{ 1,1 }, intersections[i].p_ch2 + cv::Point2d{ 1,1 })))[0]);
		}
		for (size_t j{}; j < ch4_seq.slices.size(); ++j) {
			if (cv::Rect(cv::Point{ 0, 0 }, ch4_seq.slices[j].image.size()).contains(intersections[i].p_ch4))
				value = reduce_(value, ch4_seq.slices[j].image(intersections[i].p_ch4));//cv::mean(ch4_seq.slices[j].image(cv::Rect(intersections[i].p_ch4 - cv::Point2d{ 1,1 }, intersections[i].p_ch4 + cv::Point2d{ 1,1 })))[0]);
		}
		const auto& s = sax_seqs[i];
		for (size_t j{}; j < s.slices.size(); ++j) {
			if (cv::Rect(cv::Point{ 0, 0 }, s.slices[j].image.size()).contains(intersections[i].p_sax))
				value = reduce_(value, s.slices[j].image(intersections[i].p_sax)); // cv::mean(s.slices[j].image(cv::Rect(intersections[i].p_sax - cv::Point2d{ 1,1 }, intersections[i].p_sax + cv::Point2d{ 1,1 })))[0]);
		}
	};

	double minv = std::numeric_limits<double>::max();
	double maxv = std::numeric_limits<double>::min();

	for (size_t i{}; i < intersections.size(); ++i) {
		accumulate_(i, minv, [](double v1, double v2) {return std::min(v1, v2); });
		accumulate_(i, maxv, [](double v1, double v2) {return std::max(v1, v2); });
	}

	return std::pair<double, double>(0.5 * minv, maxv * 1.5);
}


void PatientData::save_contours() const {
	std::string annotation_file_name = this->directory + ".pts";
	std::ofstream out(annotation_file_name);

	std::cout << "saving contour to " + annotation_file_name << std::endl;
	for (const Sequence& seq : sax_seqs) {
		for (const Slice& slice : seq.slices) {
			if (slice.aux.count(AUX_LV_MASK) != 0) {
				std::cout << "saving contour " + slice.filename << std::endl;
				out << slice.filename << "\t";
				const cv::Mat contour = slice.aux.at(AUX_CONTOUR);
				for (size_t i{}; i < contour.rows; ++i) {
					out << contour.at<double>(i) << "\t";
				}
				out << std::endl;
			}
		}
	}
}

line_eq_t slices_intersection(const OrientedObject& s1, const OrientedObject& s2)
{
	// Implementaion based on http://paulbourke.net/geometry/pointlineplane/, Section "The intersection of two planes"
	const cv::Vec3d N1 = s1.normal();
	const cv::Vec3d N2 = s2.normal();
	const double d1 = s1.normal().dot(s1.position);
	const double d2 = s2.normal().dot(s2.position);

	const double dp1 = N1.dot(N1) * N2.dot(N2);
	const double dp2 = std::pow(N1.dot(N2), 2);
	const double det = dp1 - dp2;

	const double c1 = (d1 * N2.dot(N2) - d2 * N1.dot(N2)) / det;
	const double c2 = (d2 * N1.dot(N1) - d1 * N1.dot(N2)) / det;

	return line_eq_t([=](double u) { return c1 * N1 + c2 * N2 + u * N1.cross(N2); });
}

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
