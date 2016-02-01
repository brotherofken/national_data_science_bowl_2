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

#include <boost/math/special_functions/sign.hpp> // boost::math::sign()
#include <boost/algorithm/string/predicate.hpp> // boost::iequals()
#include <boost/algorithm/string/join.hpp> // boost::algorithm::join()

#if defined(__gnu_linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <boost/math/constants/constants.hpp> // boost::math::constants::pi<>()
#include <boost/program_options/options_description.hpp> // boost::program_options::options_description,
// boost::program_options::value<>
#include <boost/program_options/variables_map.hpp> // boost::program_options::variables_map,
												   // boost::program_options::store(),
												   // boost::program_options::notify()
#include <boost/program_options/parsers.hpp> // boost::program_options::cmd_line::parser
#include <boost/filesystem/operations.hpp> // boost::filesystem::exists()
#include <boost/filesystem/convenience.hpp> // boost::filesystem::change_extension()

#if defined(__gnu_linux__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(_WIN32)
#include <windows.h> // CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo, GetStdHandle, STD_OUTPUT_HANDLE
#elif defined(__unix__)
#include <sys/ioctl.h> // struct winsize, ioctl(), TIOCGWINSZ
#endif

#include "chan_vese/ChanVeseCommon.hpp"        // ChanVese::
#include "chan_vese/InteractiveDataCirc.hpp"   // InteractiveDataCirc
#include "chan_vese/InteractiveDataRect.hpp"   // InteractiveDataRect
#include "chan_vese/VideoWriterManager.hpp"    // VideoWriterManager
#include "chan_vese/ParallelPixelFunction.hpp" // ParallelPixelFunction

									 // Everything above comes with cv::

#include <gdcmReader.h>
#include <gdcmImageReader.h>
#include <gdcmAttribute.h>

cv::Mat1d read_dcm(const std::string& filename)
{
	// Read Imageimage_position
	gdcm::ImageReader ir;
	ir.SetFileName(filename.c_str());
	if (!ir.Read()) {
		std::cerr << "Could not read: " << filename << std::endl;
		return cv::Mat1d();
	}

	const gdcm::Image &gimage = ir.GetImage();

	std::vector<short> vbuffer(gimage.GetBufferLength());
	gimage.GetBuffer((char*)&vbuffer[0]);

	const unsigned int size_x = gimage.GetDimensions()[0];
	const unsigned int size_y = gimage.GetDimensions()[1];
	cv::Mat1d image(size_y, size_x);
	std::copy(vbuffer.begin(), vbuffer.end(), image.begin());

	return image;
}


/**
 * @file
 * @todo
 *       - add level set reinitialization
 * @mainpage
 * @section intro_sec Introduction
 * This is the implementation of Perona-Malik + Chan-Sandberg-Vese segmentation algorithm in C++.
 * The premise of this code is that CSV segmentation is very sensitive to noise.
 * In order to get rid of the noise, we use Perona-Malik (which is optional) to smooth noisy
 * regions in the image. PM is optimal, because it preserves edges (unlike ordinary Gaussian blur).
 * The contour is calculated
 *
 * The article @cite Getreuer2012 is taken as a starting point in implementing CSV segmentation.
 * However, the text was short of describing vector-valued, i.e. multi-channel (RGB) images.
 * Fortunately, the original paper @cite Chan2000 proved to be useful.
 * PM segmentation is entirely based on the seminal paper @cite Perona1990.
 *
 * The code works for both grayscale and RGB images (any popular format is supported).
 * It mostly relies on OpenCV routines @cite Bradski2000. Some parallelization is done
 * across the channels with OpenMP preprocessors; one method is parallelized with TBB via
 * OpenCV interface.
 */

const cv::Scalar ChanVese::Colors::white = CV_RGB(255, 255, 255);
const cv::Scalar ChanVese::Colors::black = CV_RGB(0, 0, 0);
const cv::Scalar ChanVese::Colors::red = CV_RGB(255, 0, 0);
const cv::Scalar ChanVese::Colors::green = CV_RGB(0, 255, 0);
const cv::Scalar ChanVese::Colors::blue = CV_RGB(0, 0, 255);
const cv::Scalar ChanVese::Colors::magenta = CV_RGB(255, 0, 255);
const cv::Scalar ChanVese::Colors::yellow = CV_RGB(255, 255, 0);
const cv::Scalar ChanVese::Colors::cyan = CV_RGB(0, 255, 255);

const cv::Mat ChanVese::Kernel::fwd_x = (cv::Mat_<double>(1, 3) << 0, -1, 1);
const cv::Mat ChanVese::Kernel::fwd_y = (cv::Mat_<double>(3, 1) << 0, -1, 1);
const cv::Mat ChanVese::Kernel::bwd_x = (cv::Mat_<double>(1, 3) << -1, 1, 0);
const cv::Mat ChanVese::Kernel::bwd_y = (cv::Mat_<double>(3, 1) << -1, 1, 0);
const cv::Mat ChanVese::Kernel::ctr_x = (cv::Mat_<double>(1, 3) << -0.5, 0, 0.5);
const cv::Mat ChanVese::Kernel::ctr_y = (cv::Mat_<double>(3, 1) << -0.5, 0, 0.5);

/**
 * @brief Calculates the terminal/console width.
 *        Should work on all popular platforms.
 * @return Terminal width
 * @note Untested on Windows and MacOS.
 *       Credit to user 'quantum': http://stackoverflow.com/q/6812224/4056193
 */
int
get_terminal_width()
{
	int terminal_width = 80; // default
#if defined(_WIN32)
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	terminal_width = static_cast<int>(csbi.srWindow.Right -
		csbi.srWindow.Left + 1);
#elif defined(unix) || defined(__unix__) || defined(__unix)
	struct winsize max;
	ioctl(0, TIOCGWINSZ, &max);
	terminal_width = static_cast<int>(max.ws_col);
#endif
	return terminal_width;
}

/**
 * @brief Adds suffix to the file name
 * @param path   Path to the file
 * @param suffix Suffix
 * @param delim  String separating the original base name and the suffix
 * @return New file name with the suffix
 */
std::string
add_suffix(const std::string & path,
	const std::string & suffix,
	const std::string & delim = "_")
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
[[noreturn]] void
msg_exit(const std::string & msg)
{
	std::cerr << "\n" << msg << "\n\n";
	std::exit(EXIT_FAILURE);
}

/**
 * @brief Regularized (smoothed) Heaviside step function
 * @f[ H_\epsilon(x)=\frac{1}{2}\Big[1+\frac{2}{\pi}\arctan\Big(\frac{x}{\epsilon}\Big)\Big] @f]
 * where @f$x@f$ is the argument and @f$\epsilon@f$ the smoothing parameter
 * @param x   Argument of the step function, @f$x@f$
 * @param eps Smoothing parameter, @f$\epsilon@f$
 * @return Value of the step function at @f$x@f$
 */
double
regularized_heaviside(double x,
	double eps = 1)
{
	const double pi = boost::math::constants::pi<double>();
	return (1 + 2 / pi * std::atan(x / eps)) / 2;
}

/**
 * @brief Regularized (smoothed) Dirac delta function
 * @f[ \delta_\epsilon(x)=\frac{\epsilon}{\pi(\epsilon^2+x^2)}\,, @f]
 * where @f$x@f$ is the argument and @f$\epsilon@f$ the smoothing parameter
 * @param x   Argument of the delta function, @f$x@f$
 * @param eps Smoothing parameter, @f$\epsilon@f$
 * @return Value of the delta function at @f$x@f$
 */
/*constexpr*/ double
regularized_delta(double x,
	double eps = 1)
{
	const double pi = boost::math::constants::pi<double>();
	return eps / (pi * (std::pow(eps, 2) + std::pow(x, 2)));
}

/**
 * @brief Creates a level set with a checkerboard pattern at zero level
 *        The zero level set is found via the formula
 *        @f[ \mathrm{sign}\Big[\sin\Big(\frac{x}{5}\Big)\sin\Big(\frac{y}{5}\Big)\Big]\,, @f]
 *        where @f$x@f$ and @f$y@f$ are the positions in the image
 * @param w Width of the level set matrix
 * @param h Height of the level set matrix
 * @return The levelset
 */
cv::Mat levelset_checkerboard(int h, int w)
{
	cv::Mat u(h, w, CV_64FC1);
	const double pi = boost::math::constants::pi<double>();
	double * const u_ptr = reinterpret_cast<double *>(u.data);
	for (int i = 0; i < h; ++i)
		for (int j = 0; j < w; ++j)
			u_ptr[i * w + j] = boost::math::sign(std::sin(pi * i / 5) * std::sin(pi * j / 5));
	return u;
}

/**
 * @brief Calculates average regional variance
 * @f[ c_i = \frac{\int_\Omega I_i(x,y)g(u(x,y))\mathrm{d}x\mathrm{d}y}{
				   \int_\Omega g(u(x,y))\mathrm{d}x\mathrm{d}y}\,, @f]
 * where @f$u(x,y)@f$ is the level set function,
 * @f$I_i@f$ is the @f$i@f$-th channel in the image and
 * @f$g@f$ is either the Heaviside function @f$H(x)@f$
 * (for region encolosed by the contour) or @f$1-H(x)@f$ (for region outside
 * the contour).
 * @param img       Input image (channel), @f$I_i(x,y)@f$
 * @param u         Level set, @f$u(x,y)@f$
 * @param h         Height of the image
 * @param w         Width of the image
 * @param region    Region either inside or outside the contour
 * @param heaviside Heaviside function, @f$H(x)@f$
 *                  One might also try different regularized heaviside functions
 *                  or even a non-smoothed one; that's why we've left it as a parameter
 * @return          Average variance of the given region in the image
 * @sa variance_penalty, Region
 */
double region_variance(const cv::Mat1d& img, const cv::Mat1d& u,
	const int h, const int w,
	ChanVese::Region region, std::function<double(double)> heaviside)
{
	double nom = 0.0, denom = 0.0; 
	const auto H = (region == ChanVese::Region::Inside) ? heaviside : [&heaviside](double x) -> double { return 1. - heaviside(x); };

	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			const double heaviside = H(u(i, j));
			nom += img(i, j) * heaviside;
			denom += heaviside;
		}
	}

	return nom / denom;
}

/**
 * @brief Calculates variance penalty matrix,
 * @f[ \lambda_i\int_\Omega|I_i(x,y)-c_i|^2 g(u(x,y))\,\mathrm{d}x\mathrm{d}y\,, @f]
 * where @f$u(x,y)@f$ is the level set function,
 * @f$I_i@f$ is the @f$i@f$-th channel in the image and
 * @f$g@f$ is either the Heaviside function @f$H(x)@f$
 * (for region encolosed by the contour) or @f$1-H(x)@f$ (for region outside
 * the contour).
 * @param channel Channel of the input image, @f$I_i(x,y)@f$
 * @param h       Height of the image
 * @param w       Width of the image
 * @param c       Variance of particular region in the image, @f$c_i@f$
 * @param lambda  Penalty parameter, @f$\lambda_i@f$
 * @return Variance penalty matrix
 * @sa region_variance
 */
cv::Mat1d variance_penalty(const cv::Mat1d & channel,
	int h, int w,
	double c, double lambda)
{
	cv::Mat1d channel_term = channel - c;
	cv::pow(channel_term, 2, channel_term);
	channel_term *= lambda;
	return channel_term;
}

/**
 * @brief Calculates the curvature (divergence of normalized gradient)
 *        of the level set:
 *        @f[
 *        \kappa=
 * \Delta_x^-\left(\frac{\Delta_x^+u_{i,j}}
 * {\sqrt{\eta^2+(\Delta_x^+u_{i,j})^2+(\Delta_y^0u_{i,j})^2}}\right)+
 * \Delta_y^-\left(\frac{\Delta_y^+u_{i,j}}
 * {\sqrt{\eta^2+(\Delta_x^0u_{i,j})^2+(\Delta_y^+u_{i,j})^2}}\right)\,,
 *        @f]
 * where
 *   - @f$ \Delta_x^{\pm} @f$ and @f$ \Delta_y^{\pm} @f$ correspond to forward (@f$+@f$)
 *     and backward (@f$-@f$) difference in @f$x@f$ and @f$y@f$ direction, respectively
 *   - @f$\Delta_x^0@f$ and @f$\Delta_y^0@f$ correspond to central differences in
 *     @f$x@f$ and @f$y@f$ direction, respectively
 *   - @f$\eta@f$ is a small parameter to avoid division by zero
 *   - @f$u_{i,j}@f$ is the level set for @f$m\times n@f$ image
 * The curvature is calculated by convoluting forward, backward and central difference
 * kernels with the level set. The method assumes duplicating the pixels near the border:
 * @f[ u_{-1,j}=u_{0,j}\,,\quad u_{m,j}=u_{m-1,j}\,,\quad
 *     u_{i,-1}=u_{i,0}\,,\quad u_{i,n}=u_{n-1,j}\,. @f]
 * This method ensures that the curvature is centered at a given point and only one
 * extra pixel is needed per calculation.
 * @param u       The level set, @f$u_{i,j}@f$
 * @param h       Height of the level set matrix
 * @param w       Width of the level set matrix
 * @return Curvature
 */
cv::Mat1d curvature(const cv::Mat1d& u, int h, int w)
{
	const double eta = 1E-8;
	const double eta2 = std::pow(eta, 2);

	cv::Mat1d upx, upy, ucx, ucy;
	cv::filter2D(u, upx, CV_64FC1, ChanVese::Kernel::fwd_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
	cv::filter2D(u, upy, CV_64FC1, ChanVese::Kernel::fwd_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
	cv::filter2D(u, ucx, CV_64FC1, ChanVese::Kernel::ctr_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
	cv::filter2D(u, ucy, CV_64FC1, ChanVese::Kernel::ctr_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			upx(i, j) = upx(i, j) / std::sqrt(std::pow(upx(i, j), 2) + std::pow(ucx(i, j), 2) + eta2);
			upy(i, j) = upy(i, j) / std::sqrt(std::pow(upy(i, j), 2) + std::pow(ucy(i, j), 2) + eta2);
		}
	}

	cv::filter2D(upx, upx, CV_64FC1, ChanVese::Kernel::bwd_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
	cv::filter2D(upy, upy, CV_64FC1, ChanVese::Kernel::bwd_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
	
	const cv::Mat1d div = upx + upy;
	return div;
}

/**
 * @brief Separates the region enclosed by the contour in the image
 * @param img    Original image
 * @param u      Level set (the zero level of which gives us the region)
 * @param h      Height of the image
 * @param w      Width of the image
 * @param invert Invert the selected region
 * @return Image with a white background and the selected object(s) in the foreground
 */
cv::Mat separate(const cv::Mat & img, const cv::Mat & u, int h, int w, bool invert = false)
{
	cv::Mat selection(h, w, img.type());
	cv::Mat mask(h, w, CV_8U);
	cv::Mat u_cp(h, w, CV_32F); // for some reason cv::threshold() works only with 32-bit floats

	u.convertTo(u_cp, u_cp.type());
	cv::threshold(u_cp, mask, 0, 1, cv::THRESH_BINARY);
	mask.convertTo(mask, CV_8U);
	if (invert) mask = ~mask;

	selection = cv::Mat::zeros(h, w, img.type());// cv::Scalar(255, 255, 255));
	img.copyTo(selection, mask);
	return selection;
}

cv::Mat1d perona_malik(const cv::Mat1d& image,
	const int h, const int w,
	const double K, const double L, const double T)
{
	cv::Mat smoothed_img;

	cv::Mat1d I_prev = image.clone(); // image at previous time step
	cv::Mat1d I_res(h, w);          // the resulting image

	for (double t = 0; t < T; t += L) {
		cv::Mat1d g(h, w);
		cv::Mat1d dx, dy;

		cv::Sobel(I_prev, dx, CV_64FC1, 1, 0, 3);
		cv::Sobel(I_prev, dy, CV_64FC1, 0, 1, 3);
		cv::Mat1d I_curr = cv::Mat1d::zeros(h, w); // image at current time step

		const double * const I_prev_ptr = reinterpret_cast<double *>(I_prev.data);
		double * const I_curr_ptr = reinterpret_cast<double *>(I_curr.data);
		double * const g_ptr = reinterpret_cast<double *>(g.data);

		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				const double gx = dx(i, j);
				const double gy = dy(i, j);
				if (i == 0 || i == h - 1 || j == 0 || j == w - 1) {
					g(i, j) = 1;
				} else {
					g(i, j) = std::pow(1.0 + (std::pow(gx, 2) + std::pow(gy, 2)) / (std::pow(K, 2)), -1);
				}
			}
		}

		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j)
			{
				const int in = i == h - 1 ? i : i + 1;
				const int ip = i == 0 ? i : i - 1;
				const int jn = j == w - 1 ? j : j + 1;
				const int jp = j == 0 ? j : j - 1;

				const double Is = I_prev_ptr[in * w + j];
				const double Ie = I_prev_ptr[i  * w + jn];
				const double In = I_prev_ptr[ip * w + j];
				const double Iw = I_prev_ptr[i  * w + jp];
				const double I0 = I_prev_ptr[i  * w + j];

				const double cs = g_ptr[in * w + j];
				const double ce = g_ptr[i  * w + jn];
				const double cn = g_ptr[ip * w + j];
				const double cw = g_ptr[i  * w + jp];
				const double c0 = g_ptr[i  * w + j];

				I_curr_ptr[i * w + j] = I0 + L * ((cs + c0) * (Is - I0) +
					(ce + c0) * (Ie - I0) +
					(cn + c0) * (In - I0) +
					(cw + c0) * (Iw - I0)) / 4;
			}
		}

		I_curr.copyTo(I_prev);
		I_prev.convertTo(I_res, CV_64FC1);
	}

	smoothed_img = I_res;

	return smoothed_img;
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
		if (cv::norm(seed - _points[i]) < 0.1 * double(std::max(image_sz.width, image_sz.height)))
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
	double mu, nu, eps, tol, dt, fps, K, L, T;
	int max_steps;
	double lambda1, lambda2;
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
		po::options_description desc("Allowed options", get_terminal_width());
		desc.add_options()
			("help,h", "this message")
			("input,i", po::value<std::string>(&input_filename), "input image")
			("mu", po::value<double>(&mu)->default_value(0.5), "length penalty parameter (must be positive or zero)")
			("nu", po::value<double>(&nu)->default_value(0), "area penalty parameter")
			("dt", po::value<double>(&dt)->default_value(1), "timestep")
			("lambda2", po::value<double>(&lambda2)->default_value(1.), "penalty of variance outside the contour (default: 1's)")
			("lambda1", po::value<double>(&lambda1)->default_value(1.), "penalty of variance inside the contour (default: 1's)")
			("epsilon,e", po::value<double>(&eps)->default_value(1), "smoothing parameter in Heaviside/delta")
			("tolerance,t", po::value<double>(&tol)->default_value(0.001), "tolerance in stopping condition")
			("max-steps,N", po::value<int>(&max_steps)->default_value(1000), "maximum nof iterations (negative means unlimited)")
			("fps,f", po::value<double>(&fps)->default_value(10), "video fps")
			("edge-coef,K", po::value<double>(&K)->default_value(10), "coefficient for enhancing edge detection in Perona-Malik")
			("laplacian-coef,L", po::value<double>(&L)->default_value(0.25), "coefficient in the gradient FD scheme of Perona-Malik (must be [0, 1/4])")
			("segment-time,T", po::value<double>(&T)->default_value(20), "number of smoothing steps in Perona-Malik")
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
		if (vm.count("dt") && dt <= 0) msg_exit("Cannot have negative or zero timestep: " + std::to_string(dt) + ".");
		if (vm.count("mu") && mu < 0) msg_exit("Length penalty parameter cannot be negative: " + std::to_string(mu) + ".");
		if (vm.count("lambda1") && lambda1 < 0) msg_exit("Any value of lambda1 cannot be negative.");
		if (vm.count("lambda2") && lambda2 < 0) msg_exit("Any value of lambda2 cannot be negative.");
		if (vm.count("eps") && eps < 0) msg_exit("Cannot have negative smoothing parameter: " + std::to_string(eps) + ".");
		if (vm.count("tol") && tol < 0) msg_exit("Cannot have negative tolerance: " + std::to_string(tol) + ".");
		if (vm.count("laplacian-coef") && (L > 0.25 || L < 0)) msg_exit("The Laplacian coefficient in Perona-Malik segmentation must be between 0 and 0.25.");
		if (vm.count("segment-time") && (T < L)) msg_exit("The segmentation duration must exceed the value of Laplacian coefficient, " + std::to_string(L) + ".");
		if (rectangle_contour && circle_contour) msg_exit("Cannot initialize with both rectangular and circular contour");
	}
	catch (std::exception & e) {
		msg_exit("error: " + std::string(e.what()));
	}

	//-- Read the image (grayscale or BGR? RGB? BGR? help)
	cv::Mat1d img = read_dcm(input_filename);
	if (!img.data)
		msg_exit("Error on opening \"" + input_filename + "\" (probably not an image)!");

	//-- Determine the constants and define functionals
	max_steps = max_steps < 0 ? std::numeric_limits<int>::max() : max_steps;
	double max_size(std::max(img.cols, img.rows));
	double pixel_scale = 1.0;
	if (max_size > 256) {
		pixel_scale = 256. / max_size;
		cv::resize(img, img, cv::Size(), pixel_scale, pixel_scale, cv::INTER_CUBIC);
		max_size = std::max(img.cols, img.rows);
	}
	const int h = img.rows;
	const int w = img.cols;

	const auto heaviside = std::bind(regularized_heaviside, std::placeholders::_1, eps);
	const auto delta = std::bind(regularized_delta, std::placeholders::_1, eps);

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
	cv::Mat1d u;
	if (rectangle_contour || circle_contour) {
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
			u = id->get_levelset(h, w);
		}
	}
	else if (point.size() >= 2) {
		u = cv::Mat1d::zeros(h, w);
		cv::circle(u, seed, 5, cv::Scalar::all(1), 1);
	}
	else {
		u = levelset_checkerboard(h, w);
	}

	//-- Smooth the image with Perona-Malik
	cv::Mat smoothed_img;
	if (segment) {
		cv::Mat1d abs_img;

		abs_img = cv::abs(img - cv::mean(img(cv::Rect(seed - cv::Point(2, 2), cv::Size(5, 5))))[0]);

		smoothed_img = perona_malik(abs_img, h, w, K, L, T);

		double min, max;
		cv::minMaxLoc(smoothed_img, &min, &max);
		cv::imwrite(add_suffix(input_filename, "pm") + ".png", smoothed_img);
	}

	//-- Find intensity sum and derive the stopping condition
	double stop_cond = tol * cv::norm(img, cv::NORM_L2);

	//double min, max;
	//cv::minMaxLoc(channels[0], &min, &max);
	//channels[0] = (channels[0] - min) / (max - min);

	//-- Timestep loop
	for (int t = 1; t <= max_steps; ++t) {
		cv::Mat1d u_diff = cv::Mat1d::zeros(h, w);

		//-- Channel loop

		cv::Mat1d channel = segment ? smoothed_img : img;
		//-- Find the average regional variances
		const double c1 = region_variance(channel, u, h, w, ChanVese::Region::Inside, heaviside);
		const double c2 = region_variance(channel, u, h, w, ChanVese::Region::Outside, heaviside);

		//-- Calculate the contribution of one channel to the level set
		const cv::Mat1d variance_inside = variance_penalty(channel, h, w, c1, lambda1);
		const cv::Mat1d variance_outside = variance_penalty(channel, h, w, c2, lambda2);
		u_diff += -variance_inside + variance_outside;

		//-- Calculate the curvature (divergence of normalized gradient)
		const cv::Mat1d kappa = curvature(u, h, w);

		//-- Mash the terms together
		u_diff = dt * (mu * kappa - nu + u_diff);

		//-- Run delta function on the level set
		cv::Mat1d u_cp = u.clone();

		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				u_cp(i, j) = delta(u_cp(i, j));
			}
		}
		//cv::parallel_for_(cv::Range(0, h * w), ParallelPixelFunction(u_cp, w, delta));

		//-- Shift the level set
		cv::multiply(u_diff, u_cp, u_diff);
		const double u_diff_norm = cv::norm(u_diff, cv::NORM_L2);
		u += u_diff;

		//-- Check if we have achieved the desired precision
		if (u_diff_norm <= stop_cond) break;
	}

	//-- Select the region enclosed by the contour and save it to the disk
	cv::Mat separated = separate(img, u, h, w, invert);
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
	//size_t target_idx = std::distance(contours.begin(), std::find_if(contours.begin(), contours.end(), [&](std::vector<cv::Point>& contour) { return 0 < cv::pointPolygonTest(contour, seed, false); }));

	const auto rectify_lv_segment = [] (cv::Mat1d img, cv::Point seed, std::vector<std::vector<cv::Point> > contours) {
		cv::Mat1b watershed_contours(img.size(), 0);
		cv::drawContours(watershed_contours, contours, -1, cv::Scalar(255, 255, 255), -1);

		cv::Mat1b eroded_contours;
		cv::erode(watershed_contours, eroded_contours, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));

		cv::Mat1i markers(img.size(), 0);
		{
			std::vector<std::vector<cv::Point>> ws_contours;
			cv::findContours(eroded_contours, ws_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for (size_t i = 0; i < ws_contours.size(); ++i) {
				cv::drawContours(markers, ws_contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i) + 1), -1);
			}
		}
		cv::Mat3b watershed_contours_3b;
		cv::merge(std::vector<cv::Mat1b>{watershed_contours, watershed_contours, watershed_contours}, watershed_contours_3b);
		cv::watershed(watershed_contours_3b, markers);
		return markers == markers(seed);
	};
	cv::Mat1b final_mask = rectify_lv_segment(img, seed, contours);
	findContours(final_mask, contours, {}, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	const size_t target_idx = 0;
	cv::drawContours(imgc, contours, target_idx, cv::Scalar(0, 255, 0));

	cv::circle(imgc, seed, 2, cv::Scalar(0., 0., 255.), -1);
	
	cv::RotatedRect box = ::fitEllipse(contours[target_idx], seed, img.size());
	cv::ellipse(imgc, box, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

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
