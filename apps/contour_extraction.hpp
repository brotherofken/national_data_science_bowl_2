#pragma once

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

#include "chan_vese/ChanVeseCommon.hpp"        // ChanVese::
#include "chan_vese/InteractiveDataCirc.hpp"   // InteractiveDataCirc
#include "chan_vese/InteractiveDataRect.hpp"   // InteractiveDataRect
#include "chan_vese/VideoWriterManager.hpp"    // VideoWriterManager
#include "chan_vese/ParallelPixelFunction.hpp" // ParallelPixelFunction

#include <opencv2/opencv.hpp>

#include <boost/math/special_functions/sign.hpp> // boost::math::sign()
#include <boost/math/constants/constants.hpp>    // boost::math::constants::pi<>()

/**
* @brief Regularized (smoothed) Heaviside step function
* @f[ H_\epsilon(x)=\frac{1}{2}\Big[1+\frac{2}{\pi}\arctan\Big(\frac{x}{\epsilon}\Big)\Big] @f]
* where @f$x@f$ is the argument and @f$\epsilon@f$ the smoothing parameter
* @param x   Argument of the step function, @f$x@f$
* @param eps Smoothing parameter, @f$\epsilon@f$
* @return Value of the step function at @f$x@f$
*/
double regularized_heaviside(double x, double eps = 1)
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
double regularized_delta(double x, double eps = 1)
{
	const double pi = boost::math::constants::pi<double>();
	return eps / (pi * (std::pow(eps, 2) + std::pow(x, 2)));
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
double region_variance(const cv::Mat1d& img, const cv::Mat1d& u, const int h, const int w, ChanVese::Region region, std::function<double(double)> heaviside);


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
cv::Mat1d variance_penalty(const cv::Mat1d & channel, int h, int w, double c, double lambda);


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
cv::Mat1d curvature(const cv::Mat1d& u, int h, int w);


/**
* @brief Separates the region enclosed by the contour in the image
* @param img    Original image
* @param u      Level set (the zero level of which gives us the region)
* @param h      Height of the image
* @param w      Width of the image
* @param invert Invert the selected region
* @return Image with a white background and the selected object(s) in the foreground
*/
cv::Mat separate(const cv::Mat & img, const cv::Mat & u, bool invert = false);

struct PeronaMalikArgs
{
	double K;
	double L;
	double T;
};

cv::Mat1d perona_malik(const cv::Mat1d& image, const PeronaMalikArgs& args);

struct ChanVeseArgs
{
	int max_steps;
	double eps;
	double lambda1;
	double lambda2;
	double tol;
	double dt;
	double mu;
	double nu;
};

cv::Mat1d segmentation_chan_vese(const cv::Mat1d& img, const cv::Mat1d& init, const ChanVeseArgs& args);

cv::RotatedRect fitEllipse(const std::vector<cv::Point>& _points, const std::vector<double>& weights);
