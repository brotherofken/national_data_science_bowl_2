#include "contour_extraction.hpp"

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

double region_variance(const cv::Mat1d& img, const cv::Mat1d& u, const int h, const int w, ChanVese::Region region, std::function<double(double)> heaviside)
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

cv::Mat1d variance_penalty(const cv::Mat1d & channel, int h, int w, double c, double lambda)
{
	cv::Mat1d channel_term = channel - c;
	cv::pow(channel_term, 2, channel_term);
	channel_term *= lambda;
	return channel_term;
}

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

cv::Mat1d perona_malik(const cv::Mat1d& image, const PeronaMalikArgs& args)
{
	cv::Mat smoothed_img;

	const size_t w = image.cols;
	const size_t h = image.rows;

	cv::Mat1d I_prev = image.clone(); // image at previous time step
	cv::Mat1d I_res(h, w);          // the resulting image

	for (double t = 0; t < args.T; t += args.L) {
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
				}
				else {
					g(i, j) = std::pow(1.0 + (std::pow(gx, 2) + std::pow(gy, 2)) / (std::pow(args.K, 2)), -1);
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

				I_curr_ptr[i * w + j] = I0 + args.L * ((cs + c0) * (Is - I0) +
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


cv::Mat1d segmentation_chan_vese(const cv::Mat1d& img, const cv::Mat1d& init, const ChanVeseArgs& args)
{
	const auto heaviside = std::bind(regularized_heaviside, std::placeholders::_1, args.eps);
	const auto delta = std::bind(regularized_delta, std::placeholders::_1, args.eps);

	cv::Mat1d u = init.clone();

	const size_t w = img.cols;
	const size_t h = img.rows;

	//-- Find intensity sum and derive the stopping condition
	double stop_cond = args.tol * cv::norm(img, cv::NORM_L2);

	for (int t = 1; t <= args.max_steps; ++t) {
		cv::Mat1d u_diff = cv::Mat1d::zeros(h, w);

		//-- Channel loop

		cv::Mat1d channel = img;
		//-- Find the average regional variances
		const double c1 = region_variance(channel, u, h, w, ChanVese::Region::Inside, heaviside);
		const double c2 = region_variance(channel, u, h, w, ChanVese::Region::Outside, heaviside);

		//-- Calculate the contribution of one channel to the level set
		const cv::Mat1d variance_inside = variance_penalty(channel, h, w, c1, args.lambda1);
		const cv::Mat1d variance_outside = variance_penalty(channel, h, w, c2, args.lambda2);
		u_diff += -variance_inside + variance_outside;

		//-- Calculate the curvature (divergence of normalized gradient)
		const cv::Mat1d kappa = curvature(u, h, w);

		//-- Mash the terms together
		u_diff = args.dt * (args.mu * kappa - args.nu + u_diff);

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

	return u;
}

cv::Mat separate(const cv::Mat & img, const cv::Mat & u, bool invert /*= false*/)
{
	const size_t w = img.cols;
	const size_t h = img.rows;

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

cv::RotatedRect fitEllipse(const std::vector<cv::Point>& _points, const std::vector<double>& weights)
{
	using namespace cv;

	Mat points = cv::Mat(_points);
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


	cv::Mat1d W = cv::Mat1d::eye(A.rows, A.rows);
	for (size_t i{}; i < _points.size(); ++i) {
		W(i, i) = weights[i];
	}
	double wmin, wmax;
	cv::minMaxLoc(W, &wmin, &wmax);
	W *= 1 / wmax;

	//cv::pow(W, 0.5, W);

	double l = 1.;

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



cv::RotatedRect fitEllipseToCenter(const std::vector<cv::Point>& _points, const std::vector<double>& weights, const cv::Point& center)
{
	using namespace cv;

	Mat points = cv::Mat(_points);
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

	const Point* ptsi = points.ptr<Point>();

	AutoBuffer<double> _Ad(n * 5), _bd(n);
	double *Ad = _Ad, *bd = _bd;


	cv::Mat1d W = cv::Mat1d::eye(n, n);
	for (size_t i{}; i < _points.size(); ++i) {
		W(i, i) = weights[i];
	}

	for (i = 0; i < n; i++)
	{
		Point2f p = Point2f((float)ptsi[i].x, (float)ptsi[i].y);
		p -= c;
		bd[i] = 10000.0; // 1.0?
		Ad[i * 5] = -(double)p.x * p.x; // A - C signs inverted as proposed by APP
		Ad[i * 5 + 1] = -(double)p.y * p.y;
		Ad[i * 5 + 2] = -(double)p.x * p.y;
		Ad[i * 5 + 3] = p.x;
		Ad[i * 5 + 4] = p.y;
	}

	// re-fit for parameters A - C with those center coordinates
	rp[0] = center.x;
	rp[1] = center.x;
	Mat A = Mat(n, 3, CV_64F, Ad);
	Mat b = Mat(n, 1, CV_64F, bd);
	Mat x = Mat(3, 1, CV_64F, gfp);

	double l = 10000.0;
	cv::Mat1d Reg = cv::Mat1d::eye(A.cols, A.cols);

	for (i = 0; i < n; i++)
	{
		Point2f p = Point2f((float)ptsi[i].x, (float)ptsi[i].y);
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
