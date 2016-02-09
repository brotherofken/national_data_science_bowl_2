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
#include "dicom_reader.hpp"
#include "contour_extraction.hpp"

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

void on_mouse(int event, int x, int y, int, void * id)
{
	Slic* ptr = static_cast<Slic*>(id);
	if (event == cv::EVENT_LBUTTONDOWN) {
		double scale = 256. / ptr->get_clusters().cols;
		cv::Point p(x / scale, y / scale);
		cv::Mat mask = ptr->get_clusters() == ptr->get_clusters()(p);

		if (global_mask.empty()) {
			global_mask = mask;
		} else {
			global_mask ^= mask;
		}

		contours_t contour = draw_global_mask(scale);

		if (contour.size() == 1) {
			global_contour = cv::Mat1d(0, 1);
			cv::Rect roi = ptr->get_roi();
			global_contour.push_back(roi.x);
			global_contour.push_back(roi.y);
			global_contour.push_back(roi.width);
			global_contour.push_back(roi.height);

			for (auto& p : contour[0]) {
				global_contour.push_back(p.x);
				global_contour.push_back(p.y);
			}
		}
	}
}

enum Keys {
	Down  = 's',
	Left  = 'a',
	Up    = 'w',
	Right = 'd',
	DecSP = '[',
	IncSP = ']',
};

int main(int argc, char ** argv)
{
	PeronaMalikArgs pm_args;
	struct ChanVeseArgs cv_args;

	std::string input_patient;

	bool object_selection = false;
	bool segment = false;
	bool show_windows = false;

	//-- Parse command line arguments
	//   Negative values in multitoken are not an issue, b/c it doesn't make much sense
	//   to use negative values for lambda1 and lambda2
	try {
		namespace po = boost::program_options;
		po::options_description desc("Allowed options", 120);
		desc.add_options()
			("help,h", "this message")
			("input,i", po::value<std::string>(&input_patient), "patient directory")
			("mu", po::value<double>(&cv_args.mu)->default_value(0.5), "length penalty parameter (must be positive or zero)")
			("nu", po::value<double>(&cv_args.nu)->default_value(0), "area penalty parameter")
			("dt", po::value<double>(&cv_args.dt)->default_value(1), "timestep")
			("lambda2", po::value<double>(&cv_args.lambda2)->default_value(1.), "penalty of variance outside the contour (default: 1's)")
			("lambda1", po::value<double>(&cv_args.lambda1)->default_value(1.), "penalty of variance inside the contour (default: 1's)")
			("epsilon,e", po::value<double>(&cv_args.eps)->default_value(1), "smoothing parameter in Heaviside/delta")
			("tolerance,t", po::value<double>(&cv_args.tol)->default_value(0.001), "tolerance in stopping condition")
			("max-steps,N", po::value<int>(&cv_args.max_steps)->default_value(1000), "maximum nof iterations (negative means unlimited)")
			("edge-coef,K", po::value<double>(&pm_args.K)->default_value(10), "coefficient for enhancing edge detection in Perona-Malik")
			("laplacian-coef,L", po::value<double>(&pm_args.L)->default_value(0.25), "coefficient in the gradient FD scheme of Perona-Malik (must be [0, 1/4])")
			("segment-time,T", po::value<double>(&pm_args.T)->default_value(20), "number of smoothing steps in Perona-Malik")
			("segment,S", po::bool_switch(&segment), "segment the image with Perona-Malik beforehand")
			("select,s", po::bool_switch(&object_selection), "separate the region encolosed by the contour (adds suffix '_selection')")
			("show", po::bool_switch(&show_windows), "");
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return EXIT_SUCCESS;
		}
		if (!vm.count("input")) msg_exit("Error: you have to specify input file name!");
		if (vm.count("input") && !boost::filesystem::exists(input_patient)) msg_exit("Error: file \"" + input_patient + "\" does not exists!");
		if (vm.count("dt") && cv_args.dt <= 0) msg_exit("Cannot have negative or zero timestep: " + std::to_string(cv_args.dt) + ".");
		if (vm.count("mu") && cv_args.mu < 0) msg_exit("Length penalty parameter cannot be negative: " + std::to_string(cv_args.mu) + ".");
		if (vm.count("lambda1") && cv_args.lambda1 < 0) msg_exit("Any value of lambda1 cannot be negative.");
		if (vm.count("lambda2") && cv_args.lambda2 < 0) msg_exit("Any value of lambda2 cannot be negative.");
		if (vm.count("eps") && cv_args.eps < 0) msg_exit("Cannot have negative smoothing parameter: " + std::to_string(cv_args.eps) + ".");
		if (vm.count("tol") && cv_args.tol < 0) msg_exit("Cannot have negative tolerance: " + std::to_string(cv_args.tol) + ".");
		if (vm.count("laplacian-coef") && (pm_args.L > 0.25 || pm_args.L < 0)) msg_exit("The Laplacian coefficient in Perona-Malik segmentation must be between 0 and 0.25.");
		if (vm.count("segment-time") && (pm_args.T < pm_args.L)) msg_exit("The segmentation duration must exceed the value of Laplacian coefficient, " + std::to_string(pm_args.L) + ".");
	}
	catch (std::exception & e) {
		msg_exit("error: " + std::string(e.what()));
	}

	PatientData patient_data(input_patient);

	auto& sax = patient_data.sax_seqs[8];

	cv::Vec3d point_3d = slices_intersection(patient_data.ch2_seq.slices[0], patient_data.ch4_seq.slices[0], sax.slices[0]);
	cv::Point2d point = sax.point_to_image(point_3d);

	cv::Mat1d img = sax.slices[11].image;
	std::string input_filename = sax.slices[0].filename;

	std::pair <double, double> min_max = patient_data.get_min_max_bp_level();
	std::cout << "min_max: " << min_max.first << " " << min_max.second << std::endl;

	int sax_id = 0;
	int slice_id = 0;
	int key = 0;
	int superpixel_num = 200;

	Slic slic;

	while (key != 'q') {
		const int prev_sax_id = sax_id;
		const int prev_slice_id = slice_id;
		bool save_contours = false;
		const size_t sequence_len = patient_data.sax_seqs[sax_id].slices.size();
		switch (key) {
			case Down: sax_id = std::max(0, sax_id - 1); break;
			case Up: sax_id = std::min(patient_data.sax_seqs.size() - 1, size_t(sax_id + 1)); break;
			case Left: slice_id = (slice_id - 1) < 0 ? (sequence_len - 1) : (slice_id - 1); break;
			case Right: slice_id = (slice_id + 1) % sequence_len; break;
			case DecSP: superpixel_num = std::max(50, superpixel_num - 25); break;
			case IncSP: superpixel_num = std::min(250, superpixel_num + 25); break;
			case 'c': save_contours = true;
			default: break;
		}


		// Get data for current slice
		Slice& cur_slice = patient_data.sax_seqs[sax_id].slices[slice_id];
		Slice& ch2_slice = patient_data.ch2_seq.slices[slice_id];
		Slice& ch4_slice = patient_data.ch4_seq.slices[slice_id];
		const PatientData::Intersection& inter = patient_data.intersections[sax_id];

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


		cv::Mat cur_image = (cur_slice.image.clone() - min_max.first)/(min_max.second - min_max.first);
		cv::Mat ch2_image = (ch2_slice.image.clone() - min_max.first)/(min_max.second - min_max.first);
		cv::Mat ch4_image = (ch4_slice.image.clone() - min_max.first)/(min_max.second - min_max.first);

		double roi_sz = 0.2 * cur_slice.image.cols;
		cv::Rect roi(inter.p_sax - cv::Point2d{ roi_sz, roi_sz }, inter.p_sax + cv::Point2d{ roi_sz, roi_sz });
		roi = roi & cv::Rect({ 0, 0 }, cur_image.size());

		// SLIC
		const double nc = 0.2;

		// Apply SLIC
		slic.generate_superpixels(cv::Mat1d(cur_image), superpixel_num, nc, roi);
		slic.create_connectivity(cv::Mat1d(cur_image));

		cv::Mat3d slic_result;
		cv::merge(std::vector<cv::Mat1d>{ cur_image,cur_image,cur_image }, slic_result);
		slic_result = slic_result(roi);

		// Drawing
		{
			const double max = 1;

			slic.display_contours(slic_result, cv::Vec3d(0, 0, 0.5 * max), 3.0);
			//slic.colour_with_cluster_means(slic_result);
			cur_slice.aux["SLIC"] = slic_result.clone();

			const double val_sax = cv::mean(cur_image(cv::Rect(inter.p_sax - cv::Point2d{ 1,1 }, inter.p_sax + cv::Point2d{ 1,1 })))[0];
			const double val_ch2 = cv::mean(ch2_image(cv::Rect(inter.p_ch2 - cv::Point2d{ 1,1 }, inter.p_ch2 + cv::Point2d{ 1,1 })))[0];
			const double val_ch4 = cv::mean(ch4_image(cv::Rect(inter.p_ch4 - cv::Point2d{ 1,1 }, inter.p_ch4 + cv::Point2d{ 1,1 })))[0];

			cur_image.convertTo(cur_image, CV_32FC1);
			ch2_image.convertTo(ch2_image, CV_32FC1);
			ch4_image.convertTo(ch4_image, CV_32FC1);
			cv::cvtColor(cur_image, cur_image, cv::COLOR_GRAY2BGR);
			cv::cvtColor(ch2_image, ch2_image, cv::COLOR_GRAY2BGR);
			cv::cvtColor(ch4_image, ch4_image, cv::COLOR_GRAY2BGR);

			const auto draw_line = [] (cv::Mat& image, const Slice& slice, const line_eq_t& line, cv::Scalar color) {
				const cv::Point2d p1 = slice.point_to_image(line(-1000));
				const cv::Point2d p2 = slice.point_to_image(line(1000));
				cv::line(image, p1, p2, color, 1);
			};
			
			draw_line(ch4_image, ch4_slice, inter.l24, cv::Scalar(0, max, 0));
			draw_line(ch2_image, ch2_slice, inter.l24, cv::Scalar(0, max, 0));
			draw_line(ch2_image, ch2_slice, inter.ls2, cv::Scalar(max, 0, 0));
			draw_line(cur_image, cur_slice, inter.ls2, cv::Scalar(max, 0, 0));
			draw_line(ch4_image, ch4_slice, inter.ls4, cv::Scalar(max, 0, 0));
			draw_line(cur_image, cur_slice, inter.ls4, cv::Scalar(max, 0, 0));
			cv::circle(cur_image, inter.p_sax, 2, cv::Scalar(0., 0., max), -1);
			cv::circle(ch2_image, inter.p_ch2, 2, cv::Scalar(0., 0., max), -1);
			cv::circle(ch4_image, inter.p_ch4, 2, cv::Scalar(0., 0., max), -1);

			cv::resize(cur_image, cur_image, cv::Size(0, 0), 256. / cur_image.cols, 256. / cur_image.cols, CV_INTER_LANCZOS4);
			cv::resize(ch2_image, ch2_image, cv::Size(0, 0), 256. / ch2_image.cols, 256. / ch2_image.cols, CV_INTER_LANCZOS4);
			cv::resize(ch4_image, ch4_image, cv::Size(0, 0), 256. / ch4_image.cols, 256. / ch4_image.cols, CV_INTER_LANCZOS4);
			cv::resize(slic_result, slic_result, cv::Size(0, 0), 256. / slic_result.cols, 256. / slic_result.cols, CV_INTER_LANCZOS4);

			const std::string navigtaion_text = std::to_string(sax_id+1) + "/" + std::to_string(patient_data.sax_seqs.size()) + "   " + std::to_string(slice_id+1)+"/" + std::to_string(sequence_len);
			cv::putText(cur_image, navigtaion_text, cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar::all(max));

			cv::putText(cur_image, std::to_string(val_sax), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar::all(max));
			cv::putText(ch2_image, std::to_string(val_ch2), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar::all(max));
			cv::putText(ch4_image, std::to_string(val_ch4), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar::all(max));

			cv::imshow("cur_slice", cur_image);
			cv::imshow("ch2", ch2_image);
			cv::imshow("ch4", ch4_image);
			cv::imshow("slic_result", slic_result);
			cv::setMouseCallback("slic_result", on_mouse, &slic);
		}
		key = cv::waitKey();
	}
	return 0;
	//-- Determine the constants and define functionals
	cv_args.max_steps = cv_args.max_steps < 0 ? std::numeric_limits<int>::max() : cv_args.max_steps;
	double max_size(std::max(img.cols, img.rows));
	double pixel_scale = 1.0;
	if (max_size > 256) {
		pixel_scale = 256. / max_size;
		cv::resize(img, img, cv::Size(), pixel_scale, pixel_scale, cv::INTER_CUBIC);
		max_size = std::max(img.cols, img.rows);
	}

#if 0
	cv::Size ms_window_sz(8, 8);
	cv::Rect ms_window(cv::Point(point.x * pixel_scale - ms_window_sz.width / 2, point.y * pixel_scale - ms_window_sz.height / 2), ms_window_sz);
	cv::meanShift(img, ms_window, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 1));
	const cv::Point seed = (ms_window.br() + ms_window.tl()) / 2;
#else
	const cv::Point seed = mean_shift(img, { point.x * pixel_scale, point.y * pixel_scale });
#endif

	//-- Construct the level set
	cv::Mat1d cv_init = cv::Mat1d::zeros(img.size());
	cv::circle(cv_init, seed, 2, cv::Scalar::all(1), 1);


	//-- Smooth the image with Perona-Malik
	const cv::Mat1d abs_img = cv::abs(img - cv::mean(img(cv::Rect(seed - cv::Point(2, 2), cv::Size(5, 5))))[0]);
	const cv::Mat smoothed_img = perona_malik(img, pm_args);

	// Actual Chen-Vese segmentation
	cv::Mat1d chan_vese_image = segmentation_chan_vese(smoothed_img, cv_init, cv_args);

	//-- Select the region enclosed by the contour and save it to the disk
	cv::Mat chan_vese_segmentation = separate(img, chan_vese_image);




	std::vector<std::vector<cv::Point> > contours;
	cv::Mat1b separated8uc;
	chan_vese_segmentation.convertTo(separated8uc, CV_8UC1, 255);
	separated8uc = separated8uc != 0;
	findContours(separated8uc, contours, {}, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	size_t target_idx = std::distance(contours.begin(), std::find_if(contours.begin(), contours.end(), [&](std::vector<cv::Point>& contour) { return 0 < cv::pointPolygonTest(contour, seed, false); }));

	cv::Mat1b final_mask = rectify_lv_segment(img, seed, contours, target_idx);

	findContours(final_mask, contours, {}, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	target_idx = 0;

	cv::Mat3d imgc;
	cv::merge(std::vector<cv::Mat1d>{ img, img, img }, imgc);
	cv::circle(imgc, seed, 2, cv::Scalar(0., 0., 255.), -1);
	cv::circle(imgc, cv::Point(point.x * pixel_scale, point.y * pixel_scale), 2, cv::Scalar(255., 0., 0.), -1);
	
	if (false && contours[target_idx].size() > 20) {
		cv::RotatedRect box = ::fitEllipse(contours[target_idx], seed, img.size());
		cv::ellipse(imgc, box, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
	}

	cv::drawContours(imgc, contours, target_idx, cv::Scalar(0, 255, 0));

	{
		double min, max;
		cv::minMaxLoc(smoothed_img, &min, &max);
		cv::imwrite(add_suffix(input_filename, "pm") + ".png", smoothed_img);
		cv::imwrite(add_suffix(input_filename, "selection") + ".png", chan_vese_segmentation);
		cv::imwrite(add_suffix(input_filename, "contour") + ".png", imgc);

		std::ofstream output(add_suffix(input_filename, "contour") + ".csv");
		std::transform(contours[target_idx].begin(), contours[target_idx].end(), std::ostream_iterator<std::string>(output, ","),
			[&](const cv::Point& p) {
			return std::to_string(p.x / pixel_scale) + "," + std::to_string(p.y / pixel_scale);
		});

	}

	if (show_windows) {
		double min, max;
		cv::minMaxLoc(img, &min, &max);
		max *= 0.5;
		chan_vese_segmentation = (chan_vese_segmentation - min) / (max - min);
		imgc = (imgc - min) / (max - min);
		cv::minMaxLoc(smoothed_img, &min, &max);
		cv::imshow("input", imgc);
		cv::imshow("smoothed", (smoothed_img - min) / (max - min));
		cv::imshow("separated", chan_vese_segmentation);
	}

	if (show_windows) cv::waitKey();
	return EXIT_SUCCESS;
}
