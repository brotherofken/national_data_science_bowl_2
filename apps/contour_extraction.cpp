
#include "contour_extraction.hpp"

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

