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


#include "dicom_reader.hpp"
#include "contour_extraction.hpp"


/**
 * @brief Adds suffix to the file name
 * @param path   Path to the file
 * @param suffix Suffix
 * @param delim  String separating the original base name and the suffix
 * @return New file name with the suffix
 */
std::string add_suffix(const std::string & path, const std::string & suffix, const std::string & delim = "_")
{
	namespace fs = boost::filesystem;
	const fs::path p(path);
	const fs::path nw_p = p.parent_path() / fs::path(p.stem().string() + delim + suffix + p.extension().string());
	return nw_p.string();
}

/**
 * @brief Displays error message surrounded by newlines and exits.
 * @param msg Message to display.
*/
void msg_exit(const std::string & msg)
{
	std::cerr << "\n" << msg << "\n\n";
	std::exit(EXIT_FAILURE);
}


/**
 * @brief Callback function for drawing contour on the image
 *        Calls InteractiveData virtual function mouse_on, which is implemented
 *        in its subclasses InteractiveDataRect (rectangular contour) and
 *        InteractiveDataCirc (circular contour)
 * @param event Event number
 * @param x     x-coordinate of the mouse in the window
 * @param y     y-coordinate of the mouse in the window
 * @param id    Additional data, which will be converted into InteractiveData pointer
 */
void
on_mouse(int event, int x, int y, int, void * id)
{
	InteractiveData * ptr = static_cast<InteractiveData *>(id);
	ptr->mouse_on(event, x, y);
}

cv::RotatedRect fitEllipse(const std::vector<cv::Point>& _points, const cv::Point& seed, const cv::Size& image_sz)
{
	using namespace cv;

	std::vector<cv::Point> close_points;
	for (size_t i{}; i < _points.size(); ++i) {
		if (cv::norm(seed - _points[i]) < 0.15 * double(std::max(image_sz.width, image_sz.height)))
			close_points.push_back(_points[i]);
	}

	Mat points = cv::Mat(close_points);
	int i, n = points.checkVector(2);
	int depth = points.depth();
	CV_Assert(n >= 0 && (depth == CV_32F || depth == CV_32S));

	RotatedRect box;

	if (n < 5)
		CV_Error(CV_StsBadSize, "There should be at least 5 points to fit the ellipse");

	// New fitellipse algorithm, contributed by Dr. Daniel Weiss
	Point2f c(0, 0);
	double gfp[5], rp[5], t;
	const double min_eps = 1e-8;
	bool is_float = depth == CV_32F;
	const Point* ptsi = points.ptr<Point>();
	const Point2f* ptsf = points.ptr<Point2f>();

	AutoBuffer<double> _Ad(n * 5), _bd(n);
	double *Ad = _Ad, *bd = _bd;

	// first fit for parameters A - E
	Mat A(n, 5, CV_64F, Ad);
	Mat b(n, 1, CV_64F, bd);
	Mat x(5, 1, CV_64F, gfp);

	for (i = 0; i < n; i++) {
		Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
		c += p;
	}
	c.x /= n;
	c.y /= n;

	for (i = 0; i < n; i++)
	{
		Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
		p -= c;
		bd[i] = 10000.0; // 1.0?
		Ad[i * 5] = -(double)p.x * p.x; // A - C signs inverted as proposed by APP
		Ad[i * 5 + 1] = -(double)p.y * p.y;
		Ad[i * 5 + 2] = -(double)p.x * p.y;
		Ad[i * 5 + 3] = p.x;
		Ad[i * 5 + 4] = p.y;
	}


	std::vector<double> distances(close_points.size());
	cv::Mat1d W = cv::Mat1d::eye(A.rows, A.rows);
	for (size_t i{}; i < close_points.size(); ++i) {
		distances[i] = cv::norm(seed - close_points[i]);
		W(i,i) = 1 / distances[i];
	}
	double wmin, wmax;
	cv::minMaxLoc(W, &wmin, &wmax);
	W *= 1 / wmax;

	cv::pow(W, 2, W);

	double l = 0.001;

	cv::Mat1d Reg = cv::Mat1d::eye(A.cols, A.cols);
	x = (A.t() * W * A + l * Reg).inv(DECOMP_SVD) * A.t() * W * b;

	//solve(A, b, x, DECOMP_SVD);

	// now use general-form parameters A - E to find the ellipse center:
	// differentiate general form wrt x/y to get two equations for cx and cy
	A = Mat(2, 2, CV_64F, Ad);
	b = Mat(2, 1, CV_64F, bd);
	x = Mat(2, 1, CV_64F, rp);
	Ad[0] = 2 * gfp[0];
	Ad[1] = Ad[2] = gfp[2];
	Ad[3] = 2 * gfp[1];
	bd[0] = gfp[3];
	bd[1] = gfp[4];
	//solve(A, b, x, DECOMP_SVD);
	Reg = cv::Mat1d::eye(A.cols, A.cols);
	x = (A.t() * A + l * Reg).inv(DECOMP_SVD) * A.t() * b;

	// re-fit for parameters A - C with those center coordinates
	A = Mat(n, 3, CV_64F, Ad);
	b = Mat(n, 1, CV_64F, bd);
	x = Mat(3, 1, CV_64F, gfp);
	for (i = 0; i < n; i++)
	{
		Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
		p -= c;
		bd[i] = 1.0;
		Ad[i * 3] = (p.x - rp[0]) * (p.x - rp[0]);
		Ad[i * 3 + 1] = (p.y - rp[1]) * (p.y - rp[1]);
		Ad[i * 3 + 2] = (p.x - rp[0]) * (p.y - rp[1]);
	}
	//solve(A, b, x, DECOMP_SVD);

	Reg = cv::Mat1d::eye(A.cols, A.cols);
	x = (A.t() * W * A + l * Reg).inv(DECOMP_SVD) * A.t() * W * b;

	// store angle and radii
	rp[4] = -0.5 * atan2(gfp[2], gfp[1] - gfp[0]); // convert from APP angle usage
	if (fabs(gfp[2]) > min_eps)
		t = gfp[2] / sin(-2.0 * rp[4]);
	else // ellipse is rotated by an integer multiple of pi/2
		t = gfp[1] - gfp[0];
	rp[2] = fabs(gfp[0] + gfp[1] - t);
	if (rp[2] > min_eps)
		rp[2] = std::sqrt(2.0 / rp[2]);
	rp[3] = fabs(gfp[0] + gfp[1] + t);
	if (rp[3] > min_eps)
		rp[3] = std::sqrt(2.0 / rp[3]);

	box.center.x = (float)rp[0] + c.x;
	box.center.y = (float)rp[1] + c.y;
	box.size.width = (float)(rp[2] * 2);
	box.size.height = (float)(rp[3] * 2);
	if (box.size.width > box.size.height)
	{
		float tmp;
		CV_SWAP(box.size.width, box.size.height, tmp);
		box.angle = (float)(90 + rp[4] * 180 / CV_PI);
	}
	if (box.angle < -180)
		box.angle += 360;
	if (box.angle > 360)
		box.angle -= 360;

	return box;
}

int main(int argc, char ** argv)
{
	PeronaMalikArgs pm_args;
	struct ChanVeseArgs cv_args;

	std::vector<double> point;
	std::string input_filename;
	bool object_selection = false;
	bool invert = false;
	bool segment = false;
	bool rectangle_contour = false;
	bool circle_contour = false;
	bool show_windows = false;
	ChanVese::TextPosition pos = ChanVese::TextPosition::TopLeft;
	cv::Scalar contour_color = ChanVese::Colors::blue;

	//-- Parse command line arguments
	//   Negative values in multitoken are not an issue, b/c it doesn't make much sense
	//   to use negative values for lambda1 and lambda2
	try {
		namespace po = boost::program_options;
		po::options_description desc("Allowed options", 120);
		desc.add_options()
			("help,h", "this message")
			("input,i", po::value<std::string>(&input_filename), "input image")
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
			("invert-selection,I", po::bool_switch(&invert), "invert selected region (see: select)")
			("select,s", po::bool_switch(&object_selection), "separate the region encolosed by the contour (adds suffix '_selection')")
			("show", po::bool_switch(&show_windows), "")
			("point,p", po::value<std::vector<double>>(&point)->multitoken(), "select seed point for segmentation")
			("rectangle,R", po::bool_switch(&rectangle_contour), "select rectangular contour interactively")
			("circle,C", po::bool_switch(&circle_contour), "select circular contour interactively");
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return EXIT_SUCCESS;
		}
		if (!vm.count("input")) msg_exit("Error: you have to specify input file name!");
		if (vm.count("input") && !boost::filesystem::exists(input_filename)) msg_exit("Error: file \"" + input_filename + "\" does not exists!");
		if (vm.count("dt") && cv_args.dt <= 0) msg_exit("Cannot have negative or zero timestep: " + std::to_string(cv_args.dt) + ".");
		if (vm.count("mu") && cv_args.mu < 0) msg_exit("Length penalty parameter cannot be negative: " + std::to_string(cv_args.mu) + ".");
		if (vm.count("lambda1") && cv_args.lambda1 < 0) msg_exit("Any value of lambda1 cannot be negative.");
		if (vm.count("lambda2") && cv_args.lambda2 < 0) msg_exit("Any value of lambda2 cannot be negative.");
		if (vm.count("eps") && cv_args.eps < 0) msg_exit("Cannot have negative smoothing parameter: " + std::to_string(cv_args.eps) + ".");
		if (vm.count("tol") && cv_args.tol < 0) msg_exit("Cannot have negative tolerance: " + std::to_string(cv_args.tol) + ".");
		if (vm.count("laplacian-coef") && (pm_args.L > 0.25 || pm_args.L < 0)) msg_exit("The Laplacian coefficient in Perona-Malik segmentation must be between 0 and 0.25.");
		if (vm.count("segment-time") && (pm_args.T < pm_args.L)) msg_exit("The segmentation duration must exceed the value of Laplacian coefficient, " + std::to_string(pm_args.L) + ".");
		if (rectangle_contour && circle_contour) msg_exit("Cannot initialize with both rectangular and circular contour");
	}
	catch (std::exception & e) {
		msg_exit("error: " + std::string(e.what()));
	}

	//-- Read the image (grayscale or BGR? RGB? BGR? help)
	Slice slice = read_dcm(input_filename);
	cv::Mat1d img = slice.image;

	if (!img.data)
		msg_exit("Error on opening \"" + input_filename + "\" (probably not an image)!");

	//-- Determine the constants and define functionals
	cv_args.max_steps = cv_args.max_steps < 0 ? std::numeric_limits<int>::max() : cv_args.max_steps;
	double max_size(std::max(img.cols, img.rows));
	double pixel_scale = 1.0;
	if (max_size > 256) {
		pixel_scale = 256. / max_size;
		cv::resize(img, img, cv::Size(), pixel_scale, pixel_scale, cv::INTER_CUBIC);
		max_size = std::max(img.cols, img.rows);
	}
	const int h = img.rows;
	const int w = img.cols;

	cv::Point seed(point[0] * pixel_scale, point[1] * pixel_scale);
	if (true) {
		cv::Mat1d gb_img, gb_img_draw;
		cv::GaussianBlur(img, gb_img, cv::Size(0.03*max_size, 0.03*max_size), 0, 0);
		gb_img_draw = gb_img.clone();
		cv::Point2d cur_pose(seed), prev_pose(0, 0);
		while (cv::norm(cur_pose - prev_pose) > 1) {
			cv::circle(gb_img_draw, cur_pose, 1, cv::Scalar(0., 0., 255.), -1);
			prev_pose = cur_pose;
			const size_t sz = (8. / 256.) * max_size;
			cv::Rect roi(cur_pose - cv::Point2d(sz, sz), cv::Size(2 * sz, 2 * sz));
			cv::Moments moments = cv::moments(gb_img(roi));
			cv::Point2d diff(moments.m10 / moments.m00, moments.m01 / moments.m00);
			cur_pose = cur_pose + diff - cv::Point2d(sz, sz);
		}
		seed = cur_pose;
	}

	//-- Construct the level set
	cv::Mat1d cv_init;
	if (point.size() >= 2) {
		cv_init = cv::Mat1d::zeros(h, w);
		cv::circle(cv_init, seed, 5, cv::Scalar::all(1), 1);
	} else if (rectangle_contour || circle_contour) {
		std::unique_ptr<InteractiveData> id;
		cv::startWindowThread();
		cv::namedWindow(WINDOW_TITLE, cv::WINDOW_NORMAL);

		double min, max;
		cv::minMaxLoc(img, &min, &max);
		cv::Mat scaled_img = img / max;
		if (rectangle_contour)
			id = std::unique_ptr<InteractiveDataRect>(new InteractiveDataRect(&scaled_img, contour_color));
		else if (circle_contour)
			id = std::unique_ptr<InteractiveDataCirc>(new InteractiveDataCirc(&scaled_img, contour_color));

		if (id) cv::setMouseCallback(WINDOW_TITLE, on_mouse, id.get());

		cv::imshow(WINDOW_TITLE, scaled_img);
		cv::waitKey();
		cv::destroyWindow(WINDOW_TITLE);

		if (id) {
			if (!id->is_ok())
				msg_exit("You must specify the contour with non-zero dimensions");
			cv_init = id->get_levelset(h, w);
		}
	} else {
		msg_exit("unknown method for initialization. Use point or rect or circle");
	}

	//-- Smooth the image with Perona-Malik
	cv::Mat smoothed_img;
	if (segment) {
		cv::Mat1d abs_img;

		abs_img = cv::abs(img - cv::mean(img(cv::Rect(seed - cv::Point(2, 2), cv::Size(5, 5))))[0]);

		smoothed_img = perona_malik(abs_img, pm_args);

		double min, max;
		cv::minMaxLoc(smoothed_img, &min, &max);
		cv::imwrite(add_suffix(input_filename, "pm") + ".png", smoothed_img);
	}


	// Actual Chen-Vese segmentation
	cv::Mat1d u = segmentation_chan_vese(segment ? smoothed_img : img, cv_init, cv_args);


	//-- Select the region enclosed by the contour and save it to the disk
	cv::Mat separated = separate(img, u, invert);
	if (object_selection) {
		cv::imwrite(add_suffix(input_filename, "selection") + ".png", separated);
	}


	cv::Mat3d imgc;
	cv::merge(std::vector<cv::Mat1d>{ img, img, img }, imgc);

	std::vector<std::vector<cv::Point> > contours;
	cv::Mat1b separated8uc;
	separated.convertTo(separated8uc, CV_8UC1, 255);
	separated8uc = separated8uc != 0;
	findContours(separated8uc, contours, {}, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	size_t target_idx = std::distance(contours.begin(), std::find_if(contours.begin(), contours.end(), [&](std::vector<cv::Point>& contour) { return 0 < cv::pointPolygonTest(contour, seed, false); }));

	const auto rectify_lv_segment = [] (cv::Mat1d img, cv::Point seed, std::vector<std::vector<cv::Point> > contours, const int target_idx) {
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

		cv::Mat lv_mask = ws_markers == ws_markers.at<int>(seed);
		cv::dilate(lv_mask, lv_mask, cv::getStructuringElement(cv::MORPH_RECT, { 3,3 }));
		return lv_mask;
	};
	cv::Mat1b final_mask = rectify_lv_segment(img, seed, contours, target_idx);

	findContours(final_mask, contours, {}, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	target_idx = 0;

	cv::circle(imgc, seed, 2, cv::Scalar(0., 0., 255.), -1);
	
	if (false && contours[target_idx].size() > 20) {
		cv::RotatedRect box = ::fitEllipse(contours[target_idx], seed, img.size());
		cv::ellipse(imgc, box, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
	}

	cv::drawContours(imgc, contours, target_idx, cv::Scalar(0, 255, 0));

	cv::imwrite(add_suffix(input_filename, "contour") + ".png", imgc);

	std::ofstream output(add_suffix(input_filename, "contour") + ".csv");
	std::transform(contours[target_idx].begin(), contours[target_idx].end(), std::ostream_iterator<std::string>(output, ","),
	[&] (const cv::Point& p) {
		return std::to_string(p.x / pixel_scale) + "," + std::to_string(p.y / pixel_scale);
	});

	if (show_windows) {
		double min, max;
		cv::minMaxLoc(img, &min, &max);
		max *= 0.5;
		separated = (separated - min) / (max - min);
		imgc = (imgc - min) / (max - min);
		cv::minMaxLoc(smoothed_img, &min, &max);
		smoothed_img = (smoothed_img - min) / (max - min);
		cv::imshow("input", imgc);
		cv::imshow("smoothed", smoothed_img);
		cv::imshow("separated", separated);
	}

	if (show_windows) cv::waitKey();
	return EXIT_SUCCESS;
}
