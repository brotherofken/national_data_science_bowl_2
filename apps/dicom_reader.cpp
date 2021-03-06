#include "dicom_reader.hpp"

#include <gdcmReader.h>
#include <gdcmStringFilter.h>
#include <gdcmImageReader.h>
#include <gdcmAttribute.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <numeric>
#include <algorithm>
#include <unordered_map>


#include "hog_lv_detector.hpp"

const std::string PatientData::AUX_LV_MASK = "lv_mask";
const std::string PatientData::AUX_CONTOUR = "lv_annotation";

const line_eq_t no_line = [](double) {return cv::Vec3d::all(0); };

namespace fs = ::boost::filesystem;

std::map<std::string, cv::Point> read_estimated_points(const std::string& path)
{
	std::ifstream fin(path);
	std::map<std::string, cv::Point> points;
	std::string filename;
	int x, y;

	while (fin >> filename >> x >> y) {
		std::string basename = fs::basename(fs::path(filename));
		points[basename] = cv::Point(x, y);
	}
	return points;
}


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

	HogLvDetector lv_detector;// .detect(cv::Mat1d(cur_image), {}, true);

	cv::Point try_locate_lv(const cv::Mat1d& image_)
	{

		cv::Mat hist;
		cv::Rect roi = cv::Rect({ 0,0 }, image_.size());
		const auto minmax = get_quantile_uchar(image_(roi), hist, 0.05, 0.95);
		cv::Mat1d image = (image_ - minmax.first) / (minmax.second - minmax.first);
		image.setTo(1, image > 1.0);
		image.setTo(0, image < 0.0);

		const cv::Rect r = lv_detector.detect(image, cv::Point{ image.cols / 2, image.rows / 2 }, true);
		return (r.br() + r.tl()) / 2;
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

size_t get_patient_id(const std::string& dcm_filename)
{
	const std::vector<std::string> strs = string_split(fs::path(dcm_filename).string(), "/\\");
	auto it = std::find(strs.begin(), strs.end(), "study");
	try {
		return std::stoi(*(it - 1));
	} catch (...) {
		return -1;
	}
}

// Bad style, no time for refactoring
Slice::Slice(const std::string& _filename)
	: empty(false)
	, filename(_filename)
	, frame_number(get_frame_number(_filename))
	, patient_id(get_patient_id(_filename))
{
	std::replace(filename.begin(), filename.end(), char('\\'), char('/'));
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

		//imaged.convertTo(image, image.type(), 255);
		//test_images.push_back(image);


		// Read non-image fields

		row_dc = cv::Vec3d(gimage.GetDirectionCosines());
		col_dc = cv::Vec3d(gimage.GetDirectionCosines() + 3);
		position = cv::Vec3d(gimage.GetOrigin());
		pixel_spacing = cv::Vec3d(gimage.GetSpacing());
		// Rotation matrix
		rm = cv::Mat1d(0, 3);
		rm.push_back(row_dc*pixel_spacing[0]);
		rm.push_back(col_dc*pixel_spacing[1]);
		rm.push_back(row_dc.cross(col_dc)*pixel_spacing[0]);
		rm = rm.reshape(1, 3);
		rm = rm.t();
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
		std::pair<std::string, std::string> acquisition_time_p = sf.ToStringPair(gdcm::Tag(0x0008, 0x0031));
		
#if 0 && defined(_DEBUG)
		std::cout << "File meta: " << filename << std::endl;
		std::cout << slice_location_p.first << " " << slice_location_p.second << std::endl;
		std::cout << slice_thickness_p.first << " " << slice_thickness_p.second << std::endl;
#endif

		acquisition_time = std::stod(acquisition_time_p.second);
		slice_location = std::stod(slice_location_p.second);
		slice_thickness = std::stod(slice_thickness_p.second);
	}
}

Sequence::Sequence(const std::string& directory)
	: OrientedObject{
		0., //double slice_location;
		1., //double slice_thickness;
		cv::Vec3d(1,0,0), //cv::Vec3d row_dc;
		cv::Vec3d(0,1,0), //cv::Vec3d col_dc;
		cv::Vec3d(0,0,0), //cv::Vec3d position;
		cv::Vec3d(1,1,0), //cv::Vec3d pixel_spacing;
		cv::Mat1d::eye(3, 3) //cv::Mat1d rm;
	}
	, empty(false)
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

PatientData::PatientData(const std::string& _data_path, const std::string& directory_)
	: data_path(_data_path)
	, ch2_annotations("")
{
	directory = data_path + "/" + directory_;
	std::clog << "Reading patient " << directory << std::endl;
	number = std::stoul(fs::path(directory).stem().string());

	// for *dcm read
	fs::path sequences_location = fs::path(directory) / "study";
	std::vector<fs::path> slice_directories = get_all(sequences_location, "~");

	for (const auto& dir : slice_directories) {
		const Sequence s((sequences_location / dir).string());
		if (s.slices.size() == 0) continue;
		assert(s.rm.cols == 3 && s.rm.rows == 3);

		if (s.slices.size() && s.slices.size() <= 30) {
			if (s.type == Sequence::Type::sax) sax_seqs.push_back(s);
			if (s.type == Sequence::Type::ch2) ch2_seq = s;
			if (s.type == Sequence::Type::ch4) ch4_seq = s;
		}
	}

	std::string estimated_centers_filename = data_path + "/lvs/NDSB_" + std::to_string(number) + ".csv";
	std::map<std::string, cv::Point> points = read_estimated_points(estimated_centers_filename);

	for (Sequence& sax : sax_seqs) {
		for (Slice& s : sax.slices) {
			std::string basename = boost::filesystem::basename(boost::filesystem::path(s.filename));
			s.estimated_center = points.count(basename) ? points[basename] : try_locate_lv(s.image);// cv::Point(-1, -1);
			if (!cv::Rect({ 0,0 }, s.image.size()).contains(s.estimated_center)) {
				s.estimated_center = try_locate_lv(s.image);
			}
		}
	}

	const bool chamber_views_ok = !ch2_seq.empty && !ch4_seq.empty;

	// Remove badly oriented saxes
	{
		const size_t initial_sax_count = sax_seqs.size();
		std::map<int, double> angles_hist;// (sax_seqs.size());
		for (Sequence& sax : sax_seqs) {
			const double some_angle = 180 * std::acos(cv::Vec3d(0, 0, 1).dot(sax.normal()) / cv::norm(sax.normal())) / CV_PI; // I know, I know...
			if (angles_hist.count(int(some_angle)))
				angles_hist[int(some_angle)]++;
			else
				angles_hist[int(some_angle)] = 1;
		}

		// find most frequent angle
		auto pr = std::max_element(angles_hist.begin(), angles_hist.end(),
				[](const std::pair<int, double>& p1, const std::pair<int, double>& p2) { return p1.second < p2.second; }
		);

		// Remove poor slices
		sax_seqs.erase(std::remove_if(sax_seqs.begin(), sax_seqs.end(), [pr](Sequence& sax) {
			const double some_angle = 180 * std::acos(cv::Vec3d(0, 0, 1).dot(sax.normal()) / cv::norm(sax.normal())) / CV_PI;
			const bool remove = int(some_angle) != pr->first;
			if (remove) std::cout << "Remove " << sax.name << " due to bad orientation." << std::endl;
			return remove;
		}), sax_seqs.end());
		std::cout << (initial_sax_count - sax_seqs.size()) << " badly oriented slices were removed" << std::endl;
	}

	// Fill intersections
	//if (chamber_views_ok) {
		for (Sequence& sax : sax_seqs) {
			line_eq_t ch2ch4_line = slices_intersection(ch2_seq, ch4_seq);
			line_eq_t saxch2_line = ch2_seq.empty ? no_line : slices_intersection(sax, ch2_seq);
			line_eq_t saxch4_line = ch4_seq.empty ? no_line : slices_intersection(sax, ch4_seq);

			// Set center of sax if intersection not available...
			const cv::Vec3d point_3d = chamber_views_ok ? slices_intersection(sax, ch2_seq, ch4_seq) : sax.point_to_3d(cv::Point2d(sax.slices[0].image.cols/2, sax.slices[0].image.rows / 2));
			sax.intersection = {
				ch2ch4_line, saxch2_line, saxch4_line,
				point_3d,
				sax.point_to_image(point_3d),
				ch2_seq.empty ? cv::Vec2d::all(0) : ch2_seq.point_to_image(point_3d),
				ch4_seq.empty ? cv::Vec2d::all(0) : ch4_seq.point_to_image(point_3d)
			};
		}
	//}

	// Remove duplicated slices
	const auto compare_vec3i = [](const cv::Vec3i& a) -> bool {
		return std::hash<int>()(a[0]) + std::hash<int>()(a[1]) + std::hash<int>()(a[2]);
	};
	std::unordered_map<cv::Vec3i, Sequence, std::function<size_t(cv::Vec3i)>> uniq_slices(1000, compare_vec3i);
	for (Sequence& sax : sax_seqs) {
		const cv::Vec3i point = chamber_views_ok ? cv::Vec3i(sax.intersection.p) : cv::Vec3i(sax.position);
		if (uniq_slices.count(point)) {
			if (uniq_slices[point].slices[0].acquisition_time < sax.slices[0].acquisition_time) {
				uniq_slices[point] = sax;
			}
		}
		else {
			uniq_slices[point] = sax;
		}
	}
	sax_seqs.clear();
	for (auto sax : uniq_slices) {
		sax_seqs.push_back(sax.second);
	}

	// Sort saxes
	if (chamber_views_ok) {
		// By Y coordinate of slices intersection
		std::sort(sax_seqs.begin(), sax_seqs.end(), [](Sequence& a, Sequence& b) { return a.intersection.p[1] > b.intersection.p[1]; });
	} else {
		// By Y of sax position
		std::sort(sax_seqs.begin(), sax_seqs.end(), [](Sequence& a, Sequence& b) { return a.position[1] > b.position[1]; });
	}

	// TODO: Remove duplicated code
	for (Sequence& sax : sax_seqs) {
		for (Slice& s : sax.slices) {
			cv::Mat hist;
			const double min_size = std::min(s.image.cols, s.image.rows);
			cv::Rect roi = s.estimated_center == cv::Point(-1, -1) ? 
				cv::Rect({ 0,0 }, s.image.size()) :
				cv::Rect(s.estimated_center - cv::Point(0.2*min_size, 0.2*min_size), cv::Size(0.4*min_size, 0.4*min_size));
			roi = roi & cv::Rect({ 0,0 }, s.image.size());
			const auto minmax = get_quantile_uchar(s.image(roi), hist, 0.05, 0.95);
			s.image = (s.image - minmax.first) / (minmax.second - minmax.first);
			s.image.setTo(1, s.image > 1.0);
			s.image.setTo(0, s.image < 0.0);
		}
	}

	for (Slice& s : ch2_seq.slices) {
		cv::Mat hist;
		const auto minmax = get_quantile_uchar(s.image, hist, 0.1, 0.95);
		s.image = (s.image - minmax.first) / (minmax.second - minmax.first);
		s.image.setTo(1, s.image > 1.0);
	}

	for (Slice& s : ch4_seq.slices) {
		cv::Mat hist;
		const auto minmax = get_quantile_uchar(s.image, hist, 0.1, 0.95);
		s.image = (s.image - minmax.first) / (minmax.second - minmax.first);
		s.image.setTo(1, s.image > 1.0);
	}

	// Read min/max volumes from cache if available
	if (boost::filesystem::exists(boost::filesystem::path(data_path) / "min_max_frame_idxs.csv")) {
		std::ifstream fin((boost::filesystem::path(data_path) / "min_max_frame_idxs.csv").string());
		int fin_pat_id, fin_min, fin_max;
		while (fin >> fin_pat_id >> fin_min >> fin_max) {
			if (fin_pat_id == this->number) {
				volume_idx.min = fin_min;
				volume_idx.max = fin_max;
				break;
			}
		}
	} else {
		volume_idx.min = 12;
		volume_idx.max = 0;
	}


	// Read annotations for ch2 slices
	const auto ch2_landmarks_path = fs::path(data_path) / "ch2_landmarks/landmarks.csv";
	if (fs::exists(ch2_landmarks_path)) {
		ch2_annotations = LandmarksAnnotation(ch2_landmarks_path.string());
	}
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

void PatientData::save_goodness() const {
	std::string annotation_file_name = this->directory + "_goodness.csv";
	std::ofstream out(annotation_file_name);

	std::cout << "saving goodness to " + annotation_file_name << std::endl;
	for (const Sequence& seq : sax_seqs) {
		for (const Slice& slice : seq.slices) {
			if (slice.aux.count("GOOD") != 0) {
				out << slice.filename << "\t" << slice.aux.at("GOOD").at<int>(0) << std::endl;
			}
		}
	}
}

void PatientData::save_estimated_points() const {
	std::string annotation_file_name = this->directory + "_good_estimated_lv.csv";
	std::ofstream out(annotation_file_name);

	std::cout << "saving estimated centers to " + annotation_file_name << std::endl;
	for (const Sequence& seq : sax_seqs) {
		for (const Slice& slice : seq.slices) {
			if (slice.aux.count("ESTIMATED") != 0) {
				out
					<< slice.filename << "\t"
					<< slice.aux.at("ESTIMATED").at<int>(0) << "\t"
					<< slice.aux.at("ESTIMATED").at<int>(1) << std::endl;
			}
		}
	}
}

PatientData::Ch2NormedData PatientData::get_normalized_2ch(size_t frame_number)
{
	if (ch2_seq.empty) {
		return Ch2NormedData{ cv::Mat1d(), 0, cv::Mat1d::eye(3,3), cv::Mat1d::eye(3,3), cv::Rect(), cv::Mat() };
	}
	const Slice& ch2_slice = ch2_seq.slices[frame_number];
	const Intersection& inter = sax_seqs[sax_seqs.size()/2].intersection; // get intersection from the middle

	const cv::Point2d p1 = ch2_slice.point_to_image(inter.ls2(-10));
	const cv::Point2d p2 = ch2_slice.point_to_image(inter.ls2(10));
	double angle = 180 * std::atan2((p1.y - p2.y), (p1.x - p2.x)) / CV_PI;

	cv::Mat ch2_image_wrp = ch2_slice.image.clone();
	const cv::Point rotation_center{ ch2_image_wrp.cols / 2, ch2_image_wrp.rows / 2 };
	cv::Mat rm_2d = cv::getRotationMatrix2D(rotation_center, angle, 1);
	cv::Mat inv_rm_2d = cv::getRotationMatrix2D(rotation_center, -angle, 1);

	cv::warpAffine(ch2_image_wrp, ch2_image_wrp, rm_2d, ch2_image_wrp.size());

	const cv::Point2d sax_0_p1 = ch2_slice.point_to_image(sax_seqs.front().intersection.ls2(0));
	const cv::Point2d sax_N_p2 = ch2_slice.point_to_image(sax_seqs.back().intersection.ls2(0));
	const cv::Point top_point = rm_2d * sax_0_p1;
	const cv::Point low_point = rm_2d * sax_N_p2;
	size_t min_row = std::min(top_point.y, low_point.y);
	size_t max_row = std::max(top_point.y, low_point.y);
	size_t height = max_row - min_row;
	min_row -= std::min(0.2 * height, 20.);
	max_row += std::min(0.2 * height, 20.);

	std::vector<cv::Point2d> estimated_centers_ch2;
	for (Sequence& sax : sax_seqs) {
		for (Slice& s : sax.slices) {
			cv::Point2d ch2_point = ch2_slice.point_to_image(s.point_to_3d(s.estimated_center));
			ch2_point = rm_2d * ch2_point;
			estimated_centers_ch2.push_back(ch2_point);
		}
	}
	std::sort(estimated_centers_ch2.begin(), estimated_centers_ch2.end(), [](cv::Point2d a, cv::Point2d b){return a.x < b.x; });
	cv::Point2d median_point = estimated_centers_ch2[estimated_centers_ch2.size() / 2];
	//cv::circle(ch2_image_wrp, median_point, 2, 255, -1);

	cv::Rect lv_roi = cv::Rect(cv::Point(median_point.x - 0.5*height, min_row), cv::Point(median_point.x + 0.5*height, max_row));

	std::string filename = ch2_slice.filename;
	filename.erase(0, data_path.size() + 1);

	std::vector<cv::Point2d> landmarks;
	if (ch2_annotations.annotations.count(filename)) {
		for (auto& point : ch2_annotations.annotations[filename]) {
			landmarks.push_back(rm_2d * point);//cv::circle(ch2_image, point, 2, cv::Scalar(max, 0, max), -1, 8, 0);
		}
	}

	return Ch2NormedData{ ch2_image_wrp, angle, rm_2d, inv_rm_2d, lv_roi, cv::Mat(landmarks, CV_64FC2) };
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

LandmarksAnnotation::LandmarksAnnotation(const std::string& path)
{
	std::string image_name;

	std::ifstream fin(path);

	if (!fin.is_open()) return;

	std::string line;
	while (std::getline(fin, line)) {
		std::vector<cv::Point2d> points;
		std::stringstream ss(line);
		ss >> image_name;
		double x, y;
		while (ss >> x >> y) {
			points.push_back(cv::Point2d{ x, y });
		}
		annotations[image_name] = points;
	}
}
