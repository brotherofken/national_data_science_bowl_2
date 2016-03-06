#include <iostream> // std::cout, std::cerr
#include <cstdlib> // EXIT_SUCCESS, EXIT_FAILURE
#include <vector> // std::vector<>
#include <algorithm> // std::min(), std::max()
#include <cmath> // std::pow(), std::sqrt(), std::sin(), std::atan()
#include <exception> // std::exception
#include <string> // std::string, std::to_string()
#include <functional> // std::function<>, std::bind(), std::placeholders::_1
#include <limits> // std::numeric_limits<>
#include <map> // std::map<>
#include <fstream>
#include <iterator>

#include <boost/algorithm/string/predicate.hpp> // boost::iequals()
#include <boost/algorithm/string/join.hpp> // boost::algorithm::join()

#include <boost/program_options/options_description.hpp> // boost::program_options::options_description, boost::program_options::value<>
#include <boost/program_options/variables_map.hpp> // boost::program_options::variables_map,
												   // boost::program_options::store(),
												   // boost::program_options::notify()
#include <boost/program_options/parsers.hpp> // boost::program_options::cmd_line::parser
#include <boost/filesystem/operations.hpp>   // boost::filesystem::exists()
#include <boost/filesystem/convenience.hpp>  // boost::filesystem::change_extension()

#include "slic/slic.h"
#include "cpr/FaceAlignment.h"
#include "contour_extraction.hpp"
#include "hog_lv_detector.hpp"
#include "dicom_reader.hpp"

#include "opencv/plot.hpp"

void visualize_mat(cv::Mat1d areas, const std::string& win_name, double cell_size = 32.) {
	double min_area, max_area;
	cv::minMaxLoc(areas, &min_area, &max_area);
	cv::Mat areas_show(areas.size(), areas.type());
	cv::resize(areas, areas_show, areas.size() * int(cell_size), 0, 0, cv::INTER_NEAREST);
	areas_show = (areas_show - min_area) / (max_area - min_area);
	cv::merge(std::vector<cv::Mat1d>{3, areas_show}, areas_show);
	if (cell_size >= 32)
		for (auto it = areas.begin(); it != areas.end(); ++it) {
			std::string value = std::to_string(int(*it));
			double intensity = *it / max_area;
			cv::putText(areas_show, value, it.pos()*cell_size + cv::Point(cell_size / 10, cell_size / 2), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar::all((intensity > 0.6 ? 0.2 : 0.8)));
		}
	cv::imshow(win_name, areas_show);
}

cv::Mat1i gmm_segmentaiton(const cv::Mat1f& img_roi)
{
	cv::Mat tmp;
	img_roi.convertTo(tmp, CV_8UC1, 255);
	cv::medianBlur(tmp, tmp, 7);
	tmp.convertTo(tmp, CV_32FC1, 1. / 255);


	cv::Mat1f samples(0, 1);
	for (size_t i = 0; i < img_roi.rows; i++) {
		for (size_t j = 0; j < img_roi.cols; j++) {
			samples.push_back(tmp.at<float>(i, j));
		}
	}

#if 0
	cv::Ptr<cv::ml::EM> gmm = cv::ml::EM::create();
	gmm->setClustersNumber(3);
	gmm->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
	gmm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 300, 0.1));

	cv::Mat labels;
	gmm->trainEM(samples, cv::noArray(), labels, cv::noArray());
#else
	cv::Mat labels;
	cv::kmeans(samples, 3, labels, cv::TermCriteria(), 3, cv::KMEANS_PP_CENTERS);
#endif

	labels = labels.reshape(1, img_roi.rows);
	cv::imshow("segmentation", labels * 128 * 255);
	return labels;
}

double get_circle_for_point(const cv::Mat1f& img, const cv::Point& estimated_center)
{
	// = cur_slice.estimated_center;// cur_slice.point_to_image(inter.p);
	const double width = 0.2 * img.cols;
	cv::Rect roi(estimated_center - cv::Point(width, width), estimated_center + cv::Point(width, width));
	roi = roi & cv::Rect({ 0,0 }, img.size());
	cv::Mat1f img_roi = img.clone();
	img_roi = img_roi(roi);

	cv::Mat1i labels = gmm_segmentaiton(img_roi);
	cv::Mat1b segments = labels == labels(estimated_center - roi.tl());
	cv::Mat1b segments_mask(segments.size(), 0);
	cv::Rect segments_roi(cv::Point(0.125*segments.cols, 0.125*segments.rows), cv::Size(0.75*segments.cols, 0.75*segments.rows));
	cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(0.75*segments.cols, 0.75*segments.rows)).copyTo(segments_mask(segments_roi));
	segments = segments.mul(segments_mask);

	using contours_t = std::vector<std::vector<cv::Point>>;
	contours_t contours;
	findContours(segments.clone(), contours, {}, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	contours = { *std::find_if(contours.begin(), contours.end(), [&](std::vector<cv::Point>& c) {
		return cv::pointPolygonTest(c, estimated_center - roi.tl(), false) >= 0;
	}) };
	segments.setTo(0);
	{
		std::vector<cv::Point> tmp;
		cv::convexHull(contours[0], tmp, true, true);
		contours[0] = tmp;
	}
	cv::drawContours(segments, contours, 0, cv::Scalar::all(255), 1);
	cv::imshow("gmm segmentation", segments);

	cv::Mat1f kernel = cv::getGaussianKernel(roi.width, 0.3*((roi.width - 1)*0.5 - 1) + 0.8, CV_32FC1);
	kernel /= kernel(roi.width / 2);
	kernel = kernel * kernel.t();
	cv::Mat1f magnitude;
	segments.convertTo(magnitude, CV_32FC1, 1 / 255.);
	cv::resize(kernel, kernel, magnitude.size(), 0, 0, cv::INTER_CUBIC);
	cv::Mat1f weighed_magnitude = magnitude.mul(kernel);

	double wmmax;
	cv::minMaxLoc(weighed_magnitude, nullptr, &wmmax);
	weighed_magnitude /= wmmax;
	//kernel.setTo(wmmax, kernel > 0.6 * wmmax);

	cv::Point2d mean_point = estimated_center - roi.tl();// .cols / 2, img_roi.rows / 2);
	double R = 0;
	for (size_t x{}; x < weighed_magnitude.cols; ++x) {
		for (size_t y{}; y < weighed_magnitude.rows; ++y) {
			if (weighed_magnitude(y, x) > 0.5) {
				R += cv::norm(mean_point - cv::Point2d(x, y));
			}
		}
	}
	R /= cv::countNonZero(weighed_magnitude > 0.5);

	return R;
}

/// Adds suffix to the file name
std::string add_suffix(const std::string & path, const std::string & suffix, const std::string & delim = "_")
{
	namespace fs = boost::filesystem;
	const fs::path p(path);
	const fs::path nw_p = p.parent_path() / fs::path(p.stem().string() + delim + suffix + p.extension().string());
	return nw_p.string();
}


/// Displays error message surrounded by newlines and exits.
void msg_exit(const std::string & msg)
{
	std::cerr << "\n" << msg << "\n\n";
	std::exit(EXIT_FAILURE);
}

cv::Point2d mean_shift(const cv::Mat1d& img, const cv::Point2d init, const double thresh = 0.75, const size_t max_iter = 5) {
	cv::Mat1d gb_img, gb_img_draw;

	const size_t max_size = std::max(img.cols, img.rows);
	const size_t gb_side_size = 2 * size_t(0.015 * max_size) + 1;
	//cv::GaussianBlur(img, gb_img, cv::Size(gb_side_size, gb_side_size), 0, 0);
	gb_img = img.clone();
	cv::Point2d cur_pose(init), prev_pose(0, 0);

	const size_t sz = (10. / 256.) * max_size;

	cv::Mat1d mask = cv::Mat1d::zeros(2*sz+1, 2*sz+1);
	cv::circle(mask, cv::Point(sz, sz), sz, cv::Scalar::all(1.), -1);

	int iter{};
	while (cv::norm(cur_pose - prev_pose) > thresh) {
		prev_pose = cur_pose;
		cv::Mat1d roi(gb_img(cv::Rect(cur_pose.x - sz + 0.5, cur_pose.y - sz + 0.5, 2 * sz + 1, 2 * sz + 1)));
		roi = roi.mul(mask);
		cv::Moments moments = cv::moments(roi, false);
		cv::Point2d diff(moments.m10 / moments.m00, moments.m01 / moments.m00);
		cur_pose = cur_pose + (diff - cv::Point2d(sz, sz));
		if (iter++ > max_iter) break;
	}
	return cur_pose;
};

cv::Mat1b rectify_lv_segment(const cv::Mat1d& img, const cv::Point& seed, const std::vector<std::vector<cv::Point>>& contours, const int target_idx) 
{
	cv::Mat1b watershed_contours(img.size(), 0);
	cv::drawContours(watershed_contours, contours, target_idx, cv::Scalar(255, 255, 255), -1);

	cv::Mat opening, sure_bg, sure_fg;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 3,3 });
	cv::morphologyEx(watershed_contours, opening, cv::MORPH_OPEN, kernel);
	cv::dilate(opening, sure_bg, kernel, { -1,-1 }, 3);

	cv::Mat markers_dt;
	cv::distanceTransform(watershed_contours, markers_dt, cv::DistanceTypes::DIST_L2, cv::DistanceTransformMasks::DIST_MASK_3);
	double max;
	cv::minMaxLoc(markers_dt, nullptr, &max);
	cv::Mat markers;
	cv::threshold(markers_dt, markers, 0.6 * max, 255, cv::THRESH_BINARY);
	sure_fg = markers != 0;

	cv::Mat unknown = sure_bg - sure_fg;
	markers.convertTo(markers, CV_8UC1);
	cv::connectedComponents(markers, markers);
	markers += 1;
	markers.setTo(0, unknown);

	cv::Mat ws_markers = markers.clone();
	cv::Mat3b watershed_contours_3b;
	cv::merge(std::vector<cv::Mat1b>{watershed_contours, watershed_contours, watershed_contours}, watershed_contours_3b);
	cv::watershed(watershed_contours_3b, ws_markers);

	cv::Mat1b lv_mask = ws_markers == ws_markers.at<int>(seed);
	cv::dilate(lv_mask, lv_mask, cv::getStructuringElement(cv::MORPH_RECT, { 3,3 }));
	return lv_mask;
};

cv::Mat1b global_mask;
cv::Mat3b global_mask_tmp;
cv::Mat1d global_contour;

using contours_t = std::vector<std::vector<cv::Point>>;
contours_t draw_global_mask(const double scale = 1.)
{
	if (global_mask.empty()) return{};

	contours_t contours;
	cv::Mat1b masc_c = global_mask.clone();
	findContours(masc_c, contours, {}, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	contours_t hull(contours.size());

	cv::Mat mask = global_mask.clone();
	cv::merge(std::vector<cv::Mat>{mask, mask, mask}, global_mask_tmp);

	for (size_t i{}; i < contours.size(); ++i) {
		cv::convexHull(contours[i], hull[i], false, true);
	}

	if (hull.size() > 1) {
		const auto max_hull_it = std::max_element(hull.cbegin(), hull.cend(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
			return cv::contourArea(c1) < cv::contourArea(c2);
		});
		const contours_t::value_type max_hull(max_hull_it->cbegin(), max_hull_it->cend());
		hull = contours_t{ { max_hull } };
	}

	cv::drawContours(global_mask_tmp, hull, -1, cv::Scalar(0, 0, 255), 1);

	cv::resize(global_mask_tmp, global_mask_tmp, cv::Size(), scale, scale);

	cv::imshow("segment", global_mask_tmp);
	return hull;
}

double compute_polyon_area(const std::vector<cv::Point2d>& polygon)
{
	double area{};

	size_t n = polygon.size();
	if (n < 3) return 0;  // a degenerate polygon

	std::vector<cv::Point2d> V = polygon;
	V.push_back(V[0]);

	int  i, j, k;
	for (i = 1, j = 2, k = 0; i < n; i++, j++, k++) {
		area += V[i].x * (V[j].y - V[k].y);
	}
	area += V[n].x * (V[1].y - V[n - 1].y);  // wrap-around term
	return area / 2.0;
}

enum Keys {
	Down  = 's',
	Left  = 'a',
	Up    = 'w',
	Right = 'd',
	DecSP = '[',
	IncSP = ']',
	SwitchRType = 'r',
};

size_t get_prev_sax_idx(const Sequence& seq, const size_t id) { return (int(id) - 1 <= 0) ? (seq.slices.size() - 1) : (id - 1); }
size_t get_next_sax_idx(const Sequence& seq, const size_t id) { return (id + 1) >= seq.slices.size() ? 0 : id + 1; }
Slice& get_prev_slice(Sequence& seq, const size_t id) { return seq.slices[get_prev_sax_idx(seq, id)]; }
Slice& get_next_slice(Sequence& seq, const size_t id) { return seq.slices[get_next_sax_idx(seq, id)]; }


int main(int argc, char ** argv)
{
	std::string input_patient{};
	std::string data_path{};

	bool save_landmarks = false;
	bool show_windows = false;
	bool save_normalized_ch2 = false;
	size_t cpr_repeats{ 5 };

	//-- Parse command line arguments
	//   Negative values in multitoken are not an issue, b/c it doesn't make much sense
	//   to use negative values for lambda1 and lambda2
	try {
		namespace po = boost::program_options;
		po::options_description desc("Allowed options", 120);
		desc.add_options()
			("help,h", "this message")
			("input,i", po::value<std::string>(&input_patient), "patient directory")
			("data,d", po::value<std::string>(&data_path), "data directory")
			("repeats,r", po::value<size_t>(&cpr_repeats), "CPR repeats count")
			("show,s", "show gui with contours")
			("landmarks,l", "compute and save landmarks")
			("save_norm_ch2", "compute and save landmarks")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);

		std::replace(data_path.begin(), data_path.end(), char('\\'), char('/'));
		std::replace(input_patient.begin(), input_patient.end(), char('\\'), char('/'));

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return EXIT_SUCCESS;
		}
		if (!vm.count("input")) msg_exit("Error: you have to specify input file name!");
		if (vm.count("input") && !boost::filesystem::exists(data_path + "/" + input_patient)) msg_exit("Error: file \"" + input_patient + "\" does not exists!");
		save_landmarks = vm.count("landmarks");
		show_windows = vm.count("show");
		save_normalized_ch2 = vm.count("save_norm_ch2");
		if (!vm.count("repeats")) cpr_repeats = 5;
	}
	catch (std::exception & e) {
		msg_exit("error: " + std::string(e.what()));
	}

	PatientData patient_data(data_path, input_patient);

	if (save_normalized_ch2) {
		
		for (int slice_id{}; slice_id < patient_data.ch2_seq.slices.size(); ++slice_id) {

			std::string filename = patient_data.ch2_seq.slices[slice_id].filename;
			filename.erase(0, data_path.size() + 1);

			std::vector<cv::Point2d> landmarks;
			if (patient_data.ch2_annotations.annotations.count(filename) == 0) continue;

			PatientData::Ch2NormedData ch2_image_normalized = patient_data.get_normalized_2ch(slice_id);

			//for (auto& point : cv::Mat2d(ch2_image_normalized.landmarks)) {
			//	cv::circle(ch2_image_normalized.image, cv::Point(point[0], point[1]), 2, cv::Scalar(max, 0, max), -1, 8, 0);
			//}
			//cv::rectangle(ch2_image_normalized.image, ch2_image_normalized.lv_location, cv::Scalar(255, 255, 0));
			//cv::imshow("ch2_image_wrp", ch2_image_normalized.image);
			cv::imshow("ch2_image_wrp", ch2_image_normalized.image);
			std::string image_path = (boost::filesystem::path(data_path) / (input_patient + "_2ch_" + std::to_string(slice_id) + "_max_vol")).string();
			cv::imwrite(image_path + ".png", ch2_image_normalized.image * 255);

			std::ofstream fout(image_path + ".csv");
			fout
				<< image_path + ".png" << "\t"
				<< ch2_image_normalized.lv_location.x << "\t"
				<< ch2_image_normalized.lv_location.y << "\t"
				<< ch2_image_normalized.lv_location.width << "\t"
				<< ch2_image_normalized.lv_location.height;

			for (auto& point : cv::Mat2d(ch2_image_normalized.landmarks)) {
				fout << "\t" << point[0] << "\t" << point[1];
			}
			fout << std::endl;
			cv::waitKey(10);
		}
	}

	if (!show_windows && !save_landmarks)
		return EXIT_SUCCESS;

	int sax_id = 0;
	int slice_id = 0;
	int key = 0;
	int superpixel_num = 200;
	size_t landmark_num = 16;

	//HogLvDetector lv_detector;

	ShapeRegressor regressor;
	regressor.Load("cpr_model_circled_kmeans_smooth.txt");

	ShapeRegressor ch2_regressor;
	ch2_regressor.Load("cpr_model_ch2.txt");

	HogLvDetector lv_detector;

	auto shape2polygon = [] (const cv::Mat1d& shape, const cv::Vec3d& spacing) {
		std::vector<cv::Point2d> result;
		for (size_t i{}; i < shape.rows; ++i) {
			cv::Point2d point2d{ shape(i, 0)*spacing[0], shape(i, 1) *spacing[1] };
			result.push_back(point2d);
		}
		return result;
	};

	// Compute ch2 contours and savetheir data
	{
		std::ofstream fout(patient_data.directory + "_ch2_info.txt");

		{

			PatientData::Ch2NormedData ch2_image_normalized = patient_data.get_normalized_2ch(0);


			std::vector<size_t> sax_locations_y;
			for (auto& seq : patient_data.sax_seqs) {
				fout << seq.name << '\t';
			}
			fout << std::endl;
			
			for (auto& seq : patient_data.sax_seqs) {
				const cv::Point2d sax_location = ch2_image_normalized.rotation_mat * patient_data.ch2_seq.point_to_image(seq.intersection.ls2(0));
				sax_locations_y.push_back(sax_location.y);
				fout << sax_location.y << '\t';
			}
			fout << std::endl;
			fout << ch2_image_normalized.andle << std::endl;
			fout << ch2_image_normalized.lv_location.x 	  << "\t"
				<< ch2_image_normalized.lv_location.y 	  << "\t"
				<< ch2_image_normalized.lv_location.width << "\t"
				<< ch2_image_normalized.lv_location.height << std::endl;
		}

		for (size_t slice_id{}; slice_id < patient_data.ch2_seq.slices.size(); ++slice_id) {
			PatientData::Ch2NormedData ch2_image_normalized = patient_data.get_normalized_2ch(slice_id);
			cv::Mat ch2_image_norm_img = ch2_image_normalized.image.clone();

			if (ch2_image_normalized.landmarks.rows) {
				for (auto& point : cv::Mat2d(ch2_image_normalized.landmarks)) {
					cv::circle(ch2_image_norm_img, cv::Point(point[0], point[1]), 2, cv::Scalar(1, 0, 1), -1, 8, 0);
				}
			}

			cv::Mat1b image1b;
			ch2_image_normalized.image.convertTo(image1b, image1b.type(), 255);

			cv::merge(std::vector<cv::Mat>(3, ch2_image_norm_img), ch2_image_norm_img);

			BoundingBox bbox = { ch2_image_normalized.lv_location.x, ch2_image_normalized.lv_location.y, ch2_image_normalized.lv_location.width, ch2_image_normalized.lv_location.height };
			bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
			bbox.centroid_y = bbox.start_y + bbox.height / 2.0;
			cv::Mat1d current_shape = ch2_regressor.Predict(image1b, bbox, 15);// cpr_repeats);

			for (int i = 0; i < 15; i++) {
				cv::Point2d ch2_point_wrp = cv::Point2d(current_shape(i, 0), current_shape(i, 1));
				cv::Point2d ch2_point = ch2_image_normalized.inv_rotation_mat * cv::Point2d(current_shape(i, 0), current_shape(i, 1));
				cv::circle(ch2_image_norm_img, cv::Point2d(current_shape(i, 0), current_shape(i, 1)), 1, cv::Scalar(1, 0, 1), -1);
				fout << ch2_point_wrp.x << '\t' << ch2_point_wrp.y << '\t';
			}
			fout << std::endl;
			fout
				<< patient_data.ch2_seq.slices[slice_id].pixel_spacing[0] << "\t"
				<< patient_data.ch2_seq.slices[slice_id].pixel_spacing[0] << std::endl;

			cv::rectangle(ch2_image_norm_img, ch2_image_normalized.lv_location, cv::Scalar(255, 255, 0));
			cv::imshow("ch2_image_wrp", ch2_image_norm_img);
			cv::waitKey(1);
		}
		fout.close();
	}
	//return EXIT_SUCCESS;

	cv::Mat1d filtered_rads_final, rads, areas;
	{ // Pre-compute landmarks
		std::vector<std::vector<cv::Vec3d>> points3d(patient_data.sax_seqs[0].slices.size());
		std::cout << "Computing landmarks locations.. " << std::endl;

		//cv::Mat1d 
		rads = cv::Mat1d(patient_data.sax_seqs.size(), patient_data.sax_seqs[0].slices.size(), 0.);
		areas = cv::Mat1d(patient_data.sax_seqs.size(), patient_data.sax_seqs[0].slices.size(), 0.);

		int i{};
		for (Sequence& seq : patient_data.sax_seqs) {
			int j{};
			for (Slice& cur_slice : seq.slices) {
				std::cout << " " << seq.name << "/" << cur_slice.frame_number << "\t\t\t\r";
				cv::Mat cur_image = (cur_slice.image.clone());
				//cv::resize(cur_image, cur_image, cv::Size(), cur_slice.pixel_spacing[0], cur_slice.pixel_spacing[1], cv::INTER_CUBIC);

				cv::Mat1f imagef;
				cur_image.convertTo(imagef, imagef.type());
				const double R = get_circle_for_point(imagef, cur_slice.estimated_center);
				cur_slice.aux["R"] = cv::Mat1d(1, 1, R);
				rads(i, j) = R;
				j++;
			}
			i++;
		}

		cv::Mat1d rads_dx;
		cv::Mat1d kernel_dx = (cv::Mat1d(1, 2) << -1, 1);
		cv::hconcat(std::vector<cv::Mat1d>{3, rads}, rads_dx);
		cv::filter2D(rads_dx, rads_dx, -1, kernel_dx, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
		rads_dx = rads_dx(cv::Rect(cv::Point(rads.cols/*-1*/, 0), rads.size()/* + cv::Size(2,0)*/));
		cv::Mat1d abs_rads_dx = cv::abs(rads_dx);
		
		cv::Mat1b bad_elemets(rads.size(), 0);
		for (size_t r = 0; r < rads.rows; r++) {
			for (size_t c = 0; c < rads.cols; c++) {
				const bool large_R_absolute_change = abs_rads_dx(r, (c + 1) % rads.cols) > 5 * abs_rads_dx(r, c);
				const bool large_R_relative_change = 
					rads(r, (c + 1) % rads.cols) > 1.15 * rads(r, (c + 0) % rads.cols) ||
					rads(r, (c + 0) % rads.cols) > 1.15 * rads(r, (c + 1) % rads.cols);
				const bool big_gradient = abs_rads_dx(r, (c + 1) % rads.cols) > 2.5;
				if (large_R_absolute_change && large_R_relative_change && big_gradient && c < (rads.cols - 1)) {
					size_t start_c = c;
					c++;
					bad_elemets(r, c) = 255;
					c++;
					while (c < rads.cols &&
						((abs_rads_dx(r, c + 0) - std::abs(abs_rads_dx(r, (c + 1) % rads.cols)) < 2) ||
						(rads(r,c)/ rads(r, start_c) > 3))) {
						bad_elemets(r, c++) = 255;
					}
				}
			}
		}
		cv::imshow("bad_elemets", bad_elemets);

		//cv::Mat1d
		cv::Mat1d medians(rads.size());
		{
			cv::Mat1d rads_sorted;
			cv::sort(rads, rads_sorted, cv::SORT_ASCENDING | cv::SORT_EVERY_ROW);

			for (size_t r{}; r < rads.rows; ++r) {
				cv::Mat1d(rads.row(r).size(), rads_sorted(r, rads_sorted.cols / 2)).copyTo(medians.row(r));
			}
		}
		filtered_rads_final = rads.clone();
		medians.copyTo(filtered_rads_final, bad_elemets);

		i = 0;

		for (Sequence& seq : patient_data.sax_seqs) {
			int j{};
			for (Slice& cur_slice : seq.slices) {
				cv::Mat cur_image = (cur_slice.image.clone());
				const double R = filtered_rads_final(i,j);// cur_slice.aux["R"].at<double>(0, 0);
				BoundingBox lv_bbox;
				lv_bbox.start_x = cur_slice.estimated_center.x - R * 1.1;
				lv_bbox.start_y = cur_slice.estimated_center.y - R * 1.1;
				lv_bbox.width = 2 * R * 1.1;
				lv_bbox.height = 2 * R * 1.1;
				lv_bbox.centroid_x = lv_bbox.start_x + lv_bbox.width / 2.0;
				lv_bbox.centroid_y = lv_bbox.start_y + lv_bbox.height / 2.0;

				cv::Mat1b cur_image1b;
				cur_image.convertTo(cur_image1b, cur_image1b.type(), 255);
				//int prev_slice_id = ((j - 1) < 0 ? seq.slices.size() : j) - 1;
				//Slice& prev_slice = seq.slices[prev_slice_id];
				//cv::Mat1d initial_shape = cv::Mat1d();// prev_slice.aux.count("landmarks") > 0 ? prev_slice.aux["landmarks"] : cv::Mat1d();
				cv::Mat1d current_shape = lv_bbox.width*lv_bbox.height > 0 ? regressor.Predict(cur_image1b, lv_bbox, cpr_repeats) : cv::Mat1d::zeros(1, landmark_num);

				cur_slice.aux["landmarks"] = current_shape;
				cur_slice.aux["landmarks_area"] = cv::Mat1d(1, 1, compute_polyon_area(shape2polygon(current_shape, cur_slice.pixel_spacing)));
				areas(i, j) = cur_slice.aux["landmarks_area"].at<double>(0);
				j++;
			}
			i++;
		}

		std::cout << std::endl;
	}
#if 0
	{

		visualize_mat(areas, "areas", 32);
		cv::Mat1d areasf, areas_high;
		cv::hconcat(std::vector<cv::Mat1d>{3, areas}, areasf);
		for (size_t r{}; r < areasf.rows; ++r) {
			cv::Mat1f row;
			areasf.row(r).convertTo(row, row.type());
			cv::medianBlur(row, row, 5);
			row.convertTo(areasf.row(r), areasf.type());
		}
		areasf = areasf(cv::Rect(cv::Point(areas.cols, 0), areas.size()));

		//for (size_t c{}; c < areas.cols; ++c) {
		//	for (int r{ areas.rows / 2 }; r > 0; --r) {
		//
		//	}
		//	int diff = 1;
		//	for (int r{ areas.rows / 2 }; r < areas.rows - 1; ++r) {
		//		if (areas(r, c) < 10 * areas(r + diff, c)) {
		//			areas(r + diff, c) = -1;
		//			diff++; r--;
		//		}
		//	}
		//}

		//for (int slice = 0; slice < patient_data.sax_seqs[0].slices.size(); slice++) {
		//	for (int s = 0; s < patient_data.sax_seqs.size(); s++) {
		//		const Slice& cur_slice = patient_data.sax_seqs[s].slices[slice];
		//		const cv::Mat1d current_shape = cur_slice.aux["landmarks"];
		//		const cv::Mat1d current_shape = patient_data.sax_seqs[s].slices[slice_id].aux["landmarks"];
		//		const cv::Mat1d current_shape = patient_data.sax_seqs[s].slices[slice_id].aux["landmarks"];
		//		std::vector<cv::Point> contour;
		//		for (int i = 0; i < landmark_num; i++) {
		//			contour.push_back(cv::Point2d(current_shape(i, 0), current_shape(i, 1)));
		//		}
		//		cv::drawContours(cur_image, std::vector<std::vector<cv::Point>>{1, contour }, -1, cv::Scalar(0, 0, 255), 1);
		//	}
		//}

		visualize_mat(areasf, "areas_filtered", 32);
		cv::Mat1d col_sums;
		cv::reduce(areasf, col_sums, 0, cv::REDUCE_SUM);
		cv::GaussianBlur(col_sums, col_sums, cv::Size(3, 3), -1);
		visualize_mat(col_sums, "col_sums", 32);
	}
#endif
	if (save_landmarks) {
		std::string lms_savepath_2d = data_path + "/" + input_patient + "/landmarks_2d.pts";
		std::string lms_savepath_3d = data_path + "/" + input_patient + "/landmarks_3d.pts";
		std::cout << "Saving points to " << lms_savepath_3d << std::endl;
		std::ofstream fout_2d(lms_savepath_2d);
		std::ofstream fout_3d(lms_savepath_3d);
		for (Sequence& seq : patient_data.sax_seqs) {
			for (Slice& cur_slice : seq.slices) {
				cv::Mat1d shape = cur_slice.aux["landmarks"];
				for (size_t i{}; i < shape.rows; ++i) {
					cv::Point2d point2d{ shape(i, 0), shape(i, 1) };
					const cv::Vec3d point3d = cur_slice.point_to_3d(point2d);
					fout_2d << seq.name << "\t" << cur_slice.frame_number << "\t" << point2d.x << "\t" << point2d.y << std::endl;
					fout_3d << seq.name << "\t" << cur_slice.frame_number << "\t" << point3d[0] << "\t" << point3d[1] << "\t" << point3d[2] << std::endl;
				}
			}
		}
		std::string lms_savepath_areas = data_path + "/" + input_patient + "/contour_areas.csv";
		std::ofstream fout_area(lms_savepath_areas);
		for (Sequence& seq : patient_data.sax_seqs) {
			fout_area << seq.name;
			for (Slice& cur_slice : seq.slices) {
				const double area = cur_slice.aux["landmarks_area"].at<double>(0,0);
				fout_area << "\t" << area;
			}
			fout_area << std::endl;
		}
	}

	if (!show_windows)
		return EXIT_SUCCESS;

	//HogLvDetector lv_detector;
	const size_t sax_count = patient_data.sax_seqs.size();

	while (key != 'q') {
		const int prev_sax_id = sax_id;
		const int prev_slice_id = slice_id;
		bool save_contours = false;
		bool save_goodness = false;
		int marker = -1;
		const size_t sequence_len = patient_data.sax_seqs[sax_id].slices.size();
		switch (key) {
			case Down: sax_id = std::max(0, sax_id - 1); break;
			case Up: sax_id = std::min(patient_data.sax_seqs.size() - 1, size_t(sax_id + 1)); break;
			case Left: slice_id = (slice_id - 1) < 0 ? (sequence_len - 1) : (slice_id - 1); break;
			case Right: slice_id = (slice_id + 1) % sequence_len; break;
			case DecSP: superpixel_num = std::max(50, superpixel_num - 25); break;
			case IncSP: superpixel_num = std::min(250, superpixel_num + 25); break;
			case 'c': save_contours = true; break;
			case '1': marker = 1; break;
			case '2': marker = 0; break;
			case '4': marker = 11; break;
			case '5': marker = 10; break;
			case 'g': save_goodness = true; break;
			default: break;
		}

		// Get data for current slice
		Slice& prev_sax = patient_data.sax_seqs[prev_sax_id].slices[slice_id];
		Slice& cur_slice = patient_data.sax_seqs[sax_id].slices[slice_id];
		Slice no_slice;
		Slice& ch2_slice = patient_data.ch2_seq.empty ? no_slice : patient_data.ch2_seq.slices[slice_id];
		Slice& ch4_slice = patient_data.ch4_seq.empty ? no_slice : patient_data.ch4_seq.slices[slice_id];

		if (marker != -1) {
			// Mark whole sequence
			if (marker == 1 || marker == 0) {
				for (auto& c : patient_data.sax_seqs[sax_id].slices) c.aux["GOOD"] = cv::Mat1i(1,1,marker);
			}
			// Mark only current
			if (marker == 11 || marker == 10) {
				cur_slice.aux["GOOD"] = cv::Mat1i(1, 1, marker - 10);
			}
		}

		if (sax_id != prev_sax_id || slice_id != prev_slice_id) {
			// save mask
			if (!global_mask.empty()) {
				patient_data.sax_seqs[prev_sax_id].slices[prev_slice_id].aux[PatientData::AUX_LV_MASK] = global_mask.clone();
				patient_data.sax_seqs[prev_sax_id].slices[prev_slice_id].aux[PatientData::AUX_CONTOUR] = global_contour.clone();
				global_mask.release();
			}

			if (cur_slice.aux.count(PatientData::AUX_LV_MASK) != 0) {
				global_mask = cur_slice.aux[PatientData::AUX_LV_MASK].clone();
				draw_global_mask(256. / global_mask.cols);
			}
		}

		if (save_contours) {
			patient_data.save_contours();
		}

		if (save_goodness) {
			patient_data.save_goodness();
		}

		cv::Mat cur_image = (cur_slice.image.clone());


		cv::Mat1f imagef;
		cur_image.convertTo(imagef, imagef.type());

		
		lv_detector.detect(cv::Mat1d(cur_image), {}, true);

#if 0
		double R = get_circle_for_point(imagef, cur_slice.estimated_center);
		R = rtype ? rads(sax_id, slice_id) : filtered_rads_final(sax_id, slice_id);
		
		BoundingBox lv_bbox;
		lv_bbox.start_x = cur_slice.estimated_center.x - R * 1.1;
		lv_bbox.start_y = cur_slice.estimated_center.y - R * 1.1;
		lv_bbox.width = 2 * R * 1.1;
		lv_bbox.height = 2 * R * 1.1;
		lv_bbox.centroid_x = lv_bbox.start_x + lv_bbox.width / 2.0;
		lv_bbox.centroid_y = lv_bbox.start_y + lv_bbox.height / 2.0;
		cv::Mat1b cur_image1b;
		imagef.convertTo(cur_image1b, cur_image1b.type(), 255);
		cv::Mat1d current_shape = lv_bbox.width*lv_bbox.height > 0 ? regressor.Predict(cur_image1b, lv_bbox, cpr_repeats) : cv::Mat1d::zeros(1, landmark_num);
#else
		// Use precomputed contours
		cv::Mat1d current_shape = cur_slice.aux["landmarks"];
#endif
		
		std::cout
			<< "Area: " << compute_polyon_area(shape2polygon(current_shape, cur_slice.pixel_spacing)) << '\t'
			<< "Dist: " << cur_slice.distance_from_point(prev_sax.position) << '\t'
			<< std::endl;

		// Drawing
		{
			const double max = 1;

			const auto draw_line = [] (cv::Mat& image, const Slice& slice, const line_eq_t& line, cv::Scalar color, double scale, size_t width) {
				const cv::Point2d p1 = slice.point_to_image(line(-1000));
				const cv::Point2d p2 = slice.point_to_image(line(1000));
				cv::line(image, p1*scale, p2*scale, color, width);
			};

			const double scale = 384. / cur_image.cols;

			cur_image.convertTo(cur_image, CV_32FC1);
			cv::cvtColor(cur_image, cur_image, cv::COLOR_GRAY2BGR);
			if (!ch2_slice.empty && !ch4_slice.empty) {
				const Intersection& inter = patient_data.sax_seqs[sax_id].intersection;
				draw_line(cur_image, cur_slice, inter.ls2, cv::Scalar(max, 0, 0), 1, 1);
				draw_line(cur_image, cur_slice, inter.ls4, cv::Scalar(max, 0, 0), 1, 1);
				cv::circle(cur_image, inter.p_sax, 2, cv::Scalar(0., 0., max), -1);
			}
			cv::resize(cur_image, cur_image, cv::Size(0, 0), scale, scale, CV_INTER_LANCZOS4);
			const std::string navigtaion_text = std::to_string(sax_id+1) + "/" + std::to_string(patient_data.sax_seqs.size()) + "   " + std::to_string(slice_id+1)+"/" + std::to_string(sequence_len);
			cv::putText(cur_image, navigtaion_text, cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar::all(max));
			const std::string slice_info = std::to_string(cur_slice.pixel_spacing[0]) + " " + std::to_string(cur_slice.pixel_spacing[0]);
			cv::putText(cur_image, slice_info, cv::Point(10, 45), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar::all(max));
			for (int i = 0; i < landmark_num; i++) {
				cv::circle(cur_image, cv::Point2d(current_shape(i, 0), current_shape(i, 1)) * scale, 1, cv::Scalar(0, 0, 255), -1, 8, 0);
			}

			if (cur_slice.aux.count("GOOD") != 0) {
				cv::rectangle(cur_image, cv::Rect({ 0,0 }, cur_image.size()), cur_slice.aux["GOOD"].at<int>(0) == 1 ? cv::Scalar(0, max, 0) : cv::Scalar(0, 0, max), 4);
			}

			cv::circle(cur_image, cur_slice.estimated_center * scale, 3, cv::Scalar(255, 0, 255), -1, 8, 0);
			cv::imshow("cur_slice", cur_image);

			if (!ch2_slice.empty && !ch4_slice.empty) {



				const Intersection& inter = patient_data.sax_seqs[sax_id].intersection;
				cv::Mat ch2_image = ch2_slice.image.clone();
				const double scale_ch2 = 384. / ch2_image.cols;

				ch2_image.convertTo(ch2_image, CV_32FC1);
				cv::cvtColor(ch2_image, ch2_image, cv::COLOR_GRAY2BGR);
				draw_line(ch2_image, ch2_slice, inter.l24, cv::Scalar(0, max, 0), 1, 2);
				//draw_line(ch2_image, ch2_slice, inter.ls2, cv::Scalar(max, 0, 0), 1, 2);

				cv::circle(ch2_image, inter.p_ch2, 2, cv::Scalar(0., max, max), -1);
				{
					PatientData::Ch2NormedData ch2_image_normalized = patient_data.get_normalized_2ch(slice_id);
					cv::Mat ch2_image_norm_img = ch2_image_normalized.image.clone();

					if (ch2_image_normalized.landmarks.rows) {
						for (auto& point : cv::Mat2d(ch2_image_normalized.landmarks)) {
							cv::circle(ch2_image_norm_img, cv::Point(point[0], point[1]), 2, cv::Scalar(max, 0, max), -1, 8, 0);
						}
					}

					cv::Mat1b image1b;
					ch2_image_normalized.image.convertTo(image1b, image1b.type(), 255);

					cv::merge(std::vector<cv::Mat>(3, ch2_image_norm_img), ch2_image_norm_img);
					
					BoundingBox bbox = { ch2_image_normalized.lv_location.x, ch2_image_normalized.lv_location.y, ch2_image_normalized.lv_location.width, ch2_image_normalized.lv_location.height };
					bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
					bbox.centroid_y = bbox.start_y + bbox.height / 2.0;
					cv::Mat1d current_shape = ch2_regressor.Predict(image1b, bbox, 15);// cpr_repeats);

					for (int i = 0; i < 15; i++) {
						cv::circle(ch2_image_norm_img, cv::Point2d(current_shape(i, 0), current_shape(i, 1)), 1, cv::Scalar(max, 0, max), -1);
						cv::circle(ch2_image, ch2_image_normalized.inv_rotation_mat * cv::Point2d(current_shape(i, 0), current_shape(i, 1)), ch2_image.cols > 256 ? 2 : 1, cv::Scalar(0, 0, max), -1);
					}
					cv::rectangle(ch2_image_norm_img, ch2_image_normalized.lv_location, cv::Scalar(255, 255, 0));
					cv::resize(ch2_image_norm_img, ch2_image_norm_img, cv::Size(), 256. / ch2_image_norm_img.cols, 256. / ch2_image_norm_img.cols);
					cv::imshow("ch2_image_wrp", ch2_image_norm_img);
				}

				std::string filename = ch2_slice.filename;
				filename.erase(0, data_path.size() + 1);
				if (patient_data.ch2_annotations.annotations.count(filename)) {
					for (auto& point : patient_data.ch2_annotations.annotations[filename]) {
						cv::circle(ch2_image, point, 2, cv::Scalar(max, 0, max), -1, 8, 0);
					}
				}

				cv::resize(ch2_image, ch2_image, cv::Size(0, 0), scale_ch2, scale_ch2, CV_INTER_LANCZOS4);


				for (int i = 0; i < patient_data.sax_seqs.size(); i++) {
					draw_line(ch2_image, ch2_slice, patient_data.sax_seqs[i].intersection.ls2, cv::Scalar(max*0.35, 0, 0), scale_ch2, 1);
					cv::Vec3d estimated_3d = patient_data.sax_seqs[i].point_to_3d(patient_data.sax_seqs[i].slices[slice_id].estimated_center);
					cv::circle(ch2_image, patient_data.ch2_seq.point_to_image(estimated_3d) * scale_ch2, 2, cv::Scalar(max, 0, max), -1, 8, 0);
				
					cv::Mat1d slice_shape = patient_data.sax_seqs[i].slices[slice_id].aux["landmarks"];
					//for (int j = 0; j < landmark_num; j++) {
					//	cv::Vec3d estimated_3d = patient_data.sax_seqs[i].slices[slice_id].point_to_3d(cv::Point2d(slice_shape(j, 0), slice_shape(j, 1)));
					//	cv::circle(ch2_image, ch2_slice.point_to_image(estimated_3d) * scale_ch2, 1, cv::Scalar(0, 0, 255), -1, 8, 0);
					//}
				}

				cv::imshow("ch2", ch2_image);

				cv::Mat ch4_image = ch4_slice.image.clone();
				const double scale_ch4 = 384. / ch4_image.cols;

				ch4_image.convertTo(ch4_image, CV_32FC1);
				cv::cvtColor(ch4_image, ch4_image, cv::COLOR_GRAY2BGR);
				draw_line(ch4_image, ch4_slice, inter.l24, cv::Scalar(0, max, 0), 1, 2);
				//draw_line(ch4_image, ch4_slice, inter.ls4, cv::Scalar(max, 0, 0), 1, 2);
				cv::circle(ch4_image, inter.p_ch4, 2, cv::Scalar(0., 0., max), -1);
				cv::resize(ch4_image, ch4_image, cv::Size(0, 0), scale_ch4, scale_ch4, CV_INTER_LANCZOS4);

				for (int i = 0; i < patient_data.sax_seqs.size(); i++) {
					draw_line(ch4_image, ch4_slice, patient_data.sax_seqs[i].intersection.ls4, cv::Scalar(max*0.35, 0, 0), scale_ch4, 1);
					cv::Vec3d estimated_3d = patient_data.sax_seqs[i].point_to_3d(patient_data.sax_seqs[i].slices[slice_id].estimated_center);
					cv::circle(ch4_image, patient_data.ch4_seq.point_to_image(estimated_3d) * scale_ch4, 2, cv::Scalar(max, 0, max), -1, 8, 0);
				
					//cv::Mat1d slice_shape = patient_data.sax_seqs[i].slices[slice_id].aux["landmarks"];
					//for (int j = 0; j < landmark_num; j++) {
					//	cv::Vec3d estimated_3d = patient_data.sax_seqs[i].slices[slice_id].point_to_3d(cv::Point2d(slice_shape(j, 0), slice_shape(j, 1)));
					//	cv::circle(ch4_image, ch4_slice.point_to_image(estimated_3d) * scale_ch4, 1, cv::Scalar(0, 0, 255), -1, 8, 0);
					//}
				}


				cv::imshow("ch4", ch4_image);
			}
			cv::imwrite("tmp/slice_" + std::to_string(sax_id) + "_frame_" + std::to_string(slice_id) + ".png", 255*cur_image);
		}
		key = cv::waitKey();
	}
	return 0;
}
