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

#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor(), CV_BGR2RGB cv::threshold(),
                                       // cv::findContours(), cv::drawContours(),
                                       // cv::THRESH_BINARY, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE,
                                       // cv::BORDER_REPLICATE, cv::filter2D()
#include <opencv2/highgui/highgui.hpp> // cv::imread(), CV_LOAD_IMAGE_COLOR, cv::WINDOW_NORMAL,
                                       // cv::imshow(), cv::waitKey(), cv::namedWindow()

                                       // cv::Mat, cv::Scalar, cv::Vec4i, cv::Point, cv::norm(),
                                       // cv::NORM_L2, CV_64FC1, CV_64FC1, cv::Mat_<>, ParallelLoopBody

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
#include <curses.h>
#elif defined(__unix__)
#include <sys/ioctl.h> // struct winsize, ioctl(), TIOCGWINSZ
#include <ncurses.h> // initscr(), printw(), refresh(), endwin()
#endif

/**
 * @file
 * @todo
 *       - add an option to specify the initial contour from command line OR
 *         add capability to set the contour by clicking twice on the image (rectangle/circle)
 *       - add level set reinitialization
 * @mainpage
 * @section intro_sec Introduction
 * This is the implementation of Chan-Sandberg-Vese segmentation algorithm in C++.
 *
 * The article @cite Getreuer2012 is taken as a starting point in this implementation.
 * However, the text was short of describing vector-valued, i.e. multi-channel (RGB) images.
 * Fortunately, the original paper @cite Chan2000 proved to be useful.
 */

typedef unsigned char uchar; ///< Short for unsigned char
typedef unsigned long ulong; ///< Short for unsigned long int

/**
 * @brief The Region enum
 * @sa region_variance
 */
enum Region { Inside, Outside };

/**
 * @brief Enum for specifying overlay text in the image
 * @sa overlay_color
 */
enum TextPosition { TopLeft, TopRight, BottomLeft, BottomRight };

/**
 * @brief The Colors struct
 */
struct Colors
{
  static const cv::Scalar white; ///< White
  static const cv::Scalar black; ///< Black
  static const cv::Scalar red;   ///< Red
  static const cv::Scalar green; ///< Green
  static const cv::Scalar blue;  ///< Blue
};

const cv::Scalar Colors::white = CV_RGB(255, 255, 255);
const cv::Scalar Colors::black = CV_RGB(  0,   0,   0);
const cv::Scalar Colors::red   = CV_RGB(255,   0,   0);
const cv::Scalar Colors::green = CV_RGB(  0, 255,   0);
const cv::Scalar Colors::blue  = CV_RGB(  0,   0, 255);

/**
 * @brief Class for holding basic parameters of a font
 */
class FontParameters
{
public:
  FontParameters()
    : face(CV_FONT_HERSHEY_PLAIN)
    , scale(0.8)
    , thickness(1)
    , type(CV_AA)
    , baseline(0)
  {}
  /**
   * @brief FontParameters constructor
   * @param font_face      Font (type)face, expecting CV_FONT_HERSHEY_*
   * @param font_scale     Font size (relative; multiplied with base font size)
   * @param font_thickness Font thickness
   * @param font_linetype  Font line type, e.g. 8 (8-connected), 4 (4-connected)
   *                       or CV_AA (antialiased font)
   * @param font_baseline  Bottom padding in y-direction?
   */
  FontParameters(int font_face,
                 double font_scale,
                 int font_thickness,
                 int font_linetype,
                 int font_baseline)
    : face(font_face)
    , scale(font_scale)
    , thickness(font_thickness)
    , type(font_linetype)
    , baseline(font_baseline)
  {}
  int face;      ///< Font (type)face
  double scale;  ///< Font size (relative; multiplied with base font size)
  int thickness; ///< Font thickness
  int type;      ///< Font line type
  int baseline;  ///< Bottom padding in y-direction?
};

/**
 * @brief Finite difference kernels
 * @sa curvature
 */
struct Kernel
{
  static const cv::Mat fwd_x; ///< Forward difference in the x direction, @f$\Delta_x^+=f_{i+1,j}-f_{i,j}@f$
  static const cv::Mat fwd_y; ///< Forward difference in the y direction, @f$\Delta_y^+=f_{i,j+1}-f_{i,j}@f$
  static const cv::Mat bwd_x; ///< Backward difference in the x direction, @f$\Delta_x^-=f_{i,j}-f_{i-1,j}@f$
  static const cv::Mat bwd_y; ///< Backward difference in the y direction, @f$\Delta_y^-=f_{i,j}-f_{i,j-1}@f$
  static const cv::Mat ctr_x; ///< Central difference in the x direction, @f$\Delta_x^0=\frac{f_{i+1,j}-f_{i-1,j}}{2}@f$
  static const cv::Mat ctr_y; ///< Central difference in the y direction, @f$\Delta_y^0=\frac{f_{i,j+1}-f_{i,j-1}}{2}@f$
};

const cv::Mat Kernel::fwd_x = (cv::Mat_<double>(1, 3) << 0,-1,1);
const cv::Mat Kernel::fwd_y = (cv::Mat_<double>(3, 1) << 0,-1,1);
const cv::Mat Kernel::bwd_x = (cv::Mat_<double>(1, 3) << -1,1,0);
const cv::Mat Kernel::bwd_y = (cv::Mat_<double>(3, 1) << -1,1,0);
const cv::Mat Kernel::ctr_x = (cv::Mat_<double>(1, 3) << -0.5,0,0.5);
const cv::Mat Kernel::ctr_y = (cv::Mat_<double>(3, 1) << -0.5,0,0.5);

/**
 * @brief Class for calculating function values on the level set in parallel.
 *        Specifically, meant for taking regularized delta function on the level set.
 *
 * Credit to 'maythe4thbewithu' for the idea: http://goo.gl/jPtLI2
 */
class ParallelPixelFunction : public cv::ParallelLoopBody
{
public:
  /**
   * @brief Constructor
   * @param _data  Level set
   * @param _w     Width of the level set matrix
   * @param _func  Any function
   */
  ParallelPixelFunction(cv::Mat & _data,
                        int _w,
                        std::function<double(double)> _func)
    : data(_data)
    , w(_w)
    , func(_func)
  {}
  /**
   * @brief Needed by cv::parallel_for_
   * @param r Range of all indices (as if the level set is flatten)
   */
  virtual void operator () (const cv::Range & r) const
  {
    for(int i = r.start; i != r.end; ++i)
      data.at<double>(i / w, i % w) = func(data.at<double>(i / w, i % w));
  }

private:
  cv::Mat & data;
  const int w;
  const std::function<double(double)> func;
};

/**
 * @brief A wrapper for cv::VideoWriter
 *        Holds information about underlying image, optional overlay text position,
 *        contour colors and frame rate
 */
class VideoWriterManager
{
public:
  VideoWriterManager() = default;
  /**
   * @brief VideoWriterManager constructor
   *        Needs minimum information to create a video file
   * @param input_filename  File name of the original image, the file extension of which
   *                        will be renamed to '.avi'
   * @param _img            Underlying image
   * @param _contour_color  Color of the contour
   * @param fps             Frame rate (frames per second)
   * @param _pos            Overlay text position
   * @param _enable_overlay Enables text overlay
   * @sa TextPosition
   */
  VideoWriterManager(const std::string & input_filename,
                     const cv::Mat & _img,
                     const cv::Scalar & _contour_color,
                     double fps,
                     TextPosition _pos,
                     bool _enable_overlay)
    : img(_img)
    , contour_color(_contour_color)
    , font({CV_FONT_HERSHEY_PLAIN, 0.8, 1, 0, CV_AA})
    , pos(_pos)
    , enable_overlay(_enable_overlay)
  {
    const std::string video_filename = boost::filesystem::change_extension(input_filename, "avi").string();
    vw = cv::VideoWriter(video_filename, CV_FOURCC('X','V','I','D'), fps, img.size());
  }
  /**
   * @brief Writes the frame with a given zero level set to a video file
   * @param u            Level set
   * @param overlay_text Optional overlay text
   */
  void
  write_frame(const cv::Mat & u,
              const std::string & overlay_text = "")
  {
    cv::Mat nw_img = img.clone();
    draw_contour(nw_img, u);
    if(enable_overlay)
    {
      cv::Scalar color;
      cv::Point p;
      overlay_color(overlay_text, color, p);
      cv::putText(nw_img, overlay_text, p, font.face, font.scale, color, font.thickness, font.type);
    }
    vw.write(nw_img);
  }

private:
  cv::VideoWriter vw;
  cv::Mat img;
  cv::Scalar contour_color;
  FontParameters font;
  TextPosition pos;
  bool enable_overlay;

  /**
   * @brief Draws the zero level set on a given image
   * @param dst        The image where the contour is placed.
   * @param u          The level set, the zero level of which is plotted.
   * @param line_color Contour line color
   * @return 0
   * @sa levelset2contour
   */
  int
  draw_contour(cv::Mat & dst,
               const cv::Mat & u)
  {
    cv::Mat mask(img.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> cs;
    std::vector<cv::Vec4i> hier;

    cv::Mat u_cp(u.size(), CV_8UC1);
    u.convertTo(u_cp, u_cp.type());
    cv::threshold(u_cp, mask, 0, 1, cv::THRESH_BINARY);
    cv::findContours(mask, cs, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    int idx = 0;
    for(; idx >= 0; idx = hier[idx][0])
      cv::drawContours(dst, cs, idx, contour_color, 1, 8, hier);

    return 0;
  }
  /**
   * @brief Finds proper font color for overlay text
   *        The color is determined by the average intensity of the ROI where
   *        the text is placed. The function also finds correct bottom left point
   *        of the text area
   * @param img    3-channel image where the text is placed
   * @param txt    The text itself
   * @param pos    Position in the image (for possibilities: top left corner,
   *               top right, bottom left or bottom right)
   * @param fparam Font parameters that help to determine the dimensions of ROI
   * @param color  Reference to the color variable
   * @param p      Reference to the bottom left point of the text area
   * @return Black color, if the background is white enough; otherwise white color
   * @todo add some check if the text width/height exceeds image dimensions
   * @sa TextPosition, FontParameters
   */
  int
  overlay_color(const std::string txt,
                cv::Scalar & color,
                cv::Point & p)
  {

    const int threshold = 105; // bias towards black font

    const cv::Size txt_sz = cv::getTextSize(txt, font.face, font.scale, font.thickness, &font.baseline);
    const int padding = 5;
    cv::Point q;

    if(pos == TextPosition::TopLeft)
    {
      p = cv::Point(padding, padding + txt_sz.height);
      q = cv::Point(padding, padding);
    }
    else if(pos == TextPosition::TopRight)
    {
      p = cv::Point(img.cols - padding - txt_sz.width, padding + txt_sz.height);
      q = cv::Point(img.cols - padding - txt_sz.width, padding);
    }
    else if(pos == TextPosition::BottomLeft)
    {
      p = cv::Point(padding, img.rows - padding);
      q = cv::Point(padding, img.rows - padding - txt_sz.height);
    }
    else if(pos == TextPosition::BottomRight)
    {
      p = cv::Point(img.cols - padding - txt_sz.width, img.rows - padding);
      q = cv::Point(img.cols - padding - txt_sz.width, img.rows - padding - txt_sz.height);
    }

    cv::Scalar avgs = cv::mean(img(cv::Rect(q, txt_sz)));
    const double intensity_avg = 0.114*avgs[0] + 0.587*avgs[1] + 0.299*avgs[2];
    color = 255 - intensity_avg < threshold ? Colors::black : Colors::white;
    return 0;
  }
};

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
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
  return static_cast<int>(csbi.srWindow.Right - csbi.srWindow.Left + 1);
#elif defined(__unix__)
  struct winsize max;
  ioctl(0, TIOCGWINSZ, &max);
  return static_cast<int>(max.ws_col);
#endif
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
[[ noreturn ]] void
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
constexpr double
regularized_delta(double x,
                  double eps = 1)
{
  const double pi = boost::math::constants::pi<double>();
  return eps / (pi * (std::pow(eps, 2) + std::pow(x, 2)));
}

/**
 * @brief Creates a level set with rectangular zero level set
 * @param w Width of the level set matrix
 * @param h Height of the level set matrix
 * @param l Offset in pixels from the underlying image borders
 * @return The levelset
 * @todo Add support for offsets from all borders
 */
cv::Mat
levelset_rect(int h,
              int w,
              int l)
{
  cv::Mat u(h, w, CV_64FC1);
  u.setTo(cv::Scalar(1));
  for(int i = 0; i < l; ++i)
  {
    u.row(i) = cv::Scalar(-1);
    u.row(h - i - 1) = cv::Scalar(-1);
  }
  for(int j = 0; j < l; ++j)
  {
    u.col(j) = cv::Scalar(-1);
    u.col(w - j - 1) = cv::Scalar(-1);
  }

  return u;
}

/**
 * @brief Creates a level set with circular zero level set
 * @param w Width of the level set matrix
 * @param h Height of the level set matrix
 * @param d Diameter of the circle in relative units;
 *          its value must be within (0, 1); 1 indicates that
 *          the diameter is minimum of the image dimensions
 * @return The level set
 */
cv::Mat
levelset_circ(int h,
              int w,
              double d)
{
  cv::Mat u(h, w, CV_64FC1);
  double * const u_ptr = reinterpret_cast<double *>(u.data);

  const int r = std::min(w, h) * d / 2;
  const int mid_x = w / 2;
  const int mid_y = h / 2;

  for(int i = 0; i < h; ++i)
    for(int j = 0; j < w; ++j)
    {
      const double d = std::sqrt(std::pow(mid_x - i, 2) +
                                 std::pow(mid_y - j, 2));
      if(d < r) u_ptr[i * w + j] = 1;
      else      u_ptr[i * w + j] = -1;
    }

  return u;
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
cv::Mat
levelset_checkerboard(int h,
                      int w)
{
  cv::Mat u(h, w, CV_64FC1);
  const double pi = boost::math::constants::pi<double>();
  double * const u_ptr = reinterpret_cast<double *>(u.data);
  for(int i = 0; i < h; ++i)
    for(int j = 0; j < w; ++j)
      u_ptr[i * w + j] = (boost::math::sign(std::sin(pi * i / 5) *
                                            std::sin(pi * j / 5)));
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
double
region_variance(const cv::Mat & img,
                const cv::Mat & u,
                int h,
                int w,
                Region region,
                std::function<double(double)> heaviside)
{
  double nom = 0.0,
         denom = 0.0;
  const auto H = (region == Region::Inside)
                  ? heaviside
                  : [&heaviside](double x) -> double { return 1 - heaviside(x); };

  const double * const u_ptr = reinterpret_cast<double *>(u.data);
  const uchar * const img_ptr = img.data;

  for(int i = 0; i < h; ++i)
    for(int j = 0; j < w; ++j)
    {
      const double h = H(u_ptr[i * w + j]);
      nom += img_ptr[i * w + j] * h;
      denom += h;
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
cv::Mat
variance_penalty(const cv::Mat & channel,
                 int h,
                 int w,
                 double c,
                 double lambda)
{
  cv::Mat channel_term(cv::Mat::zeros(h, w, CV_64FC1));
  channel.convertTo(channel_term, channel_term.type());
  channel_term -= c;
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
cv::Mat
curvature(const cv::Mat & u,
          int h,
          int w)
{
  const double eta = 1E-8;
  const double eta2 = std::pow(eta, 2);
  cv::Mat upx (h, w, CV_64FC1), upy (h, w, CV_64FC1),
          ucx (h, w, CV_64FC1), ucy (h, w, CV_64FC1);
  cv::filter2D(u, upx, CV_64FC1, Kernel::fwd_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(u, upy, CV_64FC1, Kernel::fwd_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(u, ucx, CV_64FC1, Kernel::ctr_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(u, ucy, CV_64FC1, Kernel::ctr_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

  double * const upx_ptr = reinterpret_cast<double *>(upx.data);
  double * const upy_ptr = reinterpret_cast<double *>(upy.data);
  const double * const ucx_ptr = reinterpret_cast<double *>(ucx.data);
  const double * const ucy_ptr = reinterpret_cast<double *>(ucy.data);

#pragma omp parallel for num_threads(NUM_THREADS)
  for(int i = 0; i < h; ++i)
    for(int j = 0; j < w; ++j)
      {
        upx_ptr[i * w + j] = upx_ptr[i * w + j] /
                             std::sqrt(std::pow(upx_ptr[i * w + j], 2) + std::pow(ucx_ptr[i * w + j], 2) + eta2);
        upy_ptr[i * w + j] = upy_ptr[i * w + j] /
                             std::sqrt(std::pow(upy_ptr[i * w + j], 2) + std::pow(ucy_ptr[i * w + j], 2) + eta2);
      }

  cv::filter2D(upx, upx, CV_64FC1, Kernel::bwd_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(upy, upy, CV_64FC1, Kernel::bwd_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  upx += upy;
  return upx;
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
cv::Mat
separate(const cv::Mat & img,
         const cv::Mat & u,
         int h,
         int w,
         bool invert = false)
{
  cv::Mat selection(h, w, CV_8UC3);
  cv::Mat mask(h, w, CV_8U);
  cv::Mat u_cp(h, w, CV_32F); // for some reason cv::threshold() works only with 32-bit floats

  u.convertTo(u_cp, u_cp.type());
  cv::threshold(u_cp, mask, 0, 1, cv::THRESH_BINARY);
  mask.convertTo(mask, CV_8U);
  if(invert) mask = 1 - mask;

  selection.setTo(cv::Scalar(255, 255, 255));
  img.copyTo(selection, mask);
  return selection;
}

/**
 * @brief perona_malik
 * @param channels
 * @param h
 * @param w
 * @param K
 * @param L
 * @param T
 * @return
 */
cv::Mat
perona_malik(const std::vector<cv::Mat> & channels,
             int h,
             int w,
             double K,
             double L,
             double T)
{
  const int nof_channels = channels.size();
  std::vector<cv::Mat> smoothed_channels(nof_channels);

#pragma omp parallel for num_threads(nof_channels)
  for(int k = 0; k < nof_channels; ++k)
  {
    cv::Mat x0(h, w, CV_64FC1),
            x1(h, w, CV_64FC1),
            xc(h, w, CV_8UC1);
    channels[k].copyTo(x0);
    x0.convertTo(x0, CV_64FC1);

    for (double t = 0; t < T; t += L)
    {
      cv::Mat g(h, w, CV_64FC1),
              dx(h, w, CV_64FC1),
              dy(h, w, CV_64FC1);
      cv::Sobel(x0, dx, CV_64FC1, 1, 0, 3);
      cv::Sobel(x0, dy, CV_64FC1, 0, 1, 3);
      x1 = cv::Mat::zeros(h, w, CV_64F);

      const double * const x0_ptr = reinterpret_cast<double *>(x0.data);
      const double * const dx_ptr = reinterpret_cast<double *>(dx.data);
      const double * const dy_ptr = reinterpret_cast<double *>(dy.data);
      double * const x1_ptr = reinterpret_cast<double *>(x1.data);
      double * const g_ptr = reinterpret_cast<double *>(g.data);

      for(int i = 0; i < h; ++i)
        for(int j = 0; j < w; ++j)
        {
            const double gx = dx_ptr[i * w + j];
            const double gy = dy_ptr[i * w + j];
            const double d = i == 0 || i == h - 1 || j == 0 || j == w - 1 ?
                             1 :
                             std::pow(1.0 + (std::pow(gx, 2) + std::pow(gy, 2)) / (std::pow(K, 2)), -1);
            g_ptr[i * w + j] = d;
       }

      for(int i = 0; i < h; ++i)
        for(int j = 0; j < w; ++j)
        {
          const int in = i == h - 1 ? i : i + 1;
          const int ip = i == 0     ? i : i - 1;
          const int jn = j == w - 1 ? j : j + 1;
          const int jp = j == 0     ? j : j - 1;

          const double Is = x0_ptr[in * w + j ];
          const double Ie = x0_ptr[i  * w + jn];
          const double In = x0_ptr[ip * w + j ];
          const double Iw = x0_ptr[i  * w + jp];
          const double I0 = x0_ptr[i  * w + j ];

          const double cs = g_ptr[in * w + j ];
          const double ce = g_ptr[i  * w + jn];
          const double cn = g_ptr[ip * w + j ];
          const double cw = g_ptr[i  * w + jp];
          const double c0 = g_ptr[i  * w + j ];

          x1_ptr[i * w + j] = I0 + L * ((cs + c0) * (Is - I0) +
                                        (ce + c0) * (Ie - I0) +
                                        (cn + c0) * (In - I0) +
                                        (cw + c0) * (Iw - I0) ) / 4;
        }

      x1.copyTo(x0);
      x0.convertTo(xc, CV_8UC1);
    }

    smoothed_channels[k] = xc;
  }
  cv::Mat smoothed_img;
  cv::merge(smoothed_channels, smoothed_img);

  return smoothed_img;
}

int
main(int argc,
     char ** argv)
{
/// Performs Chan-Vese segmentation on a given input image

  double mu, nu, eps, tol, dt, fps, K, L, T;
  int max_steps;
  std::vector<double> lambda1,
                      lambda2;
  std::string input_filename,
              text_position,
              line_color_str;
  bool grayscale        = false,
       write_video      = false,
       overlay_text     = false,
       object_selection = false,
       invert           = false,
       segment          = false,
       verbose          = false;
  TextPosition pos = TextPosition::TopLeft;
  cv::Scalar contour_color = Colors::blue;

//-- Parse command line arguments
//   Negative values in multitoken are not an issue, b/c it doesn't make much sense
//   to use negative values for lambda1 and lambda2
  try
  {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options", get_terminal_width());
    desc.add_options()
      ("help,h",                                                                               "this message")
      ("input,i",            po::value<std::string>(&input_filename),                          "input image")
      ("mu",                 po::value<double>(&mu) -> default_value(0.5),                     "length penalty parameter")
      ("nu",                 po::value<double>(&nu) -> default_value(0),                       "area penalty parameter")
      ("dt",                 po::value<double>(&dt) -> default_value(1),                       "timestep")
      ("lambda1",            po::value<std::vector<double>>(&lambda1) -> multitoken(),         "penalty of variance inside the contour (default: 1's)")
      ("lambda2",            po::value<std::vector<double>>(&lambda2) -> multitoken(),         "penalty of variance outside the contour (default: 1's)")
      ("epsilon,e",          po::value<double>(&eps) -> default_value(1),                      "smoothing parameter in Heaviside/delta")
      ("tolerance,t",        po::value<double>(&tol) -> default_value(0.001),                  "tolerance in stopping condition")
      ("max-steps,N",        po::value<int>(&max_steps) -> default_value(-1),                  "maximum nof iterations (negative means unlimited)")
      ("fps,f",              po::value<double>(&fps) -> default_value(10),                     "video fps")
      ("overlay-pos,P",      po::value<std::string>(&text_position) -> default_value("TL"),    "overlay tex position; allowed only: TL, BL, TR, BR")
      ("line-color,l",       po::value<std::string>(&line_color_str) -> default_value("blue"), "contour color (allowed only: red, green, blue, black, white)")
      ("edge-coef,K",        po::value<double>(&K) -> default_value(10),                       "coefficient for enhancing edge detection in Perona-Malik")
      ("laplacian-coef,L",   po::value<double>(&L) -> default_value(0.25),                     "coefficient in the Laplacian FD scheme of Perona-Malik (must be [0, 1/4])")
      ("segment-time,T",     po::value<double>(&T) -> default_value(20),                       "number of smoothing steps in Perona-Malik")
      ("segment,S",          po::bool_switch(&segment),                                        "segment the image with Perona-Malik beforehand")
      ("grayscale,g",        po::bool_switch(&grayscale),                                      "read in as grayscale")
      ("video,V",            po::bool_switch(&write_video),                                    "enable video output (changes the extension to '.avi')")
      ("overlay-text,O",     po::bool_switch(&overlay_text),                                   "add overlay text")
      ("invert-selection,I", po::bool_switch(&invert),                                         "invert selected region (see: select)")
      ("select,s",           po::bool_switch(&object_selection),                               "separate the region encolosed by the contour (adds suffix '_selection')")
      ("verbose,v",          po::bool_switch(&verbose),                                        "verbose mode")
    ;
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
      std::cout << desc << "\n";
      return EXIT_SUCCESS;
    }
    if(! vm.count("input"))
      msg_exit("Error: you have to specify input file name!");
    else if(vm.count("input") && ! boost::filesystem::exists(input_filename))
      msg_exit("Error: file \"" + input_filename + "\" does not exists!");
    if(vm.count("dt") && dt <= 0)
      msg_exit("Cannot have negative or zero timestep: " + std::to_string(dt) + ".");
    if(vm.count("lambda1"))
    {
      if(grayscale && lambda1.size() != 1)
        msg_exit("Too many lambda1 values for a grayscale image.");
      else if(! grayscale && lambda1.size() != 3)
        msg_exit("Number of lambda1 values must be 3 for a colored input image.");
      else if(grayscale && lambda1[0] < 0)
        msg_exit("The value of lambda1 cannot be negative.");
      else if(! grayscale && (lambda1[0] < 0 || lambda1[1] < 0 || lambda1[2] < 0))
        msg_exit("Any value of lambda1 cannot be negative.");
    }
    else if(! vm.count("lambda1"))
    {
      if(grayscale) lambda1 = {1};
      else          lambda1 = {1, 1, 1};
    }
    if(vm.count("lambda2"))
    {
      if(grayscale && lambda2.size() != 1)
        msg_exit("Too many lambda2 values for a grayscale image.");
      else if(! grayscale && lambda2.size() != 3)
        msg_exit("Number of lambda2 values must be 3 for a colored input image.");
      else if(grayscale && lambda2[0] < 0)
        msg_exit("The value of lambda2 cannot be negative.");
      else if(! grayscale && (lambda2[0] < 0 || lambda2[1] < 0 || lambda2[2] < 0))
        msg_exit("Any value of lambda2 cannot be negative.");
    }
    else if(! vm.count("lambda2"))
    {
      if(grayscale) lambda2 = {1};
      else          lambda2 = {1, 1, 1};
    }
    if(vm.count("eps") && eps < 0)
      msg_exit("Cannot have negative smoothing parameter: " + std::to_string(eps) + ".");
    if(vm.count("tol") && tol < 0)
      msg_exit("Cannot have negative tolerance: " + std::to_string(tol) + ".");
    if(vm.count("overlay-pos"))
    {
      if     (boost::iequals(text_position, "TL")) pos = TextPosition::TopLeft;
      else if(boost::iequals(text_position, "BL")) pos = TextPosition::BottomLeft;
      else if(boost::iequals(text_position, "TR")) pos = TextPosition::TopRight;
      else if(boost::iequals(text_position, "BR")) pos = TextPosition::BottomRight;
      else
        msg_exit("Invalid text position requested.\n"\
                 "Correct values are: TL -- top left\n"\
                 "                    BL -- bottom left\n"\
                 "                    TR -- top right\n"\
                 "                    BR -- bottom right"\
                );
    }
    if(vm.count("line-color"))
    {
      if     (boost::iequals(line_color_str, "red"))   contour_color = Colors::red;
      else if(boost::iequals(line_color_str, "green")) contour_color = Colors::green;
      else if(boost::iequals(line_color_str, "blue"))  contour_color = Colors::blue;
      else if(boost::iequals(line_color_str, "black")) contour_color = Colors::black;
      else if(boost::iequals(line_color_str, "white")) contour_color = Colors::white;
      else
        msg_exit("Invalid contour color requested.\n"\
                 "Correct values are: red, green, blue, black, white.");
    }
    if(vm.count("laplacian-coef") && (L > 0.25 || L < 0))
      msg_exit("The Laplacian coefficient in Perona-Malik segmentation must be between 0 and 0.25.");
    if(vm.count("segment-time") && (T < L))
      msg_exit("The segmentation duration must exceed the value of Laplacian coefficient, " +
               std::to_string(L) + ".");
  }
  catch(std::exception & e)
  {
    msg_exit("error: " + std::string(e.what()));
  }

//-- Read the image (grayscale or BGR? RGB? BGR? help)
  cv::Mat _img;
  if(grayscale) _img = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);
  else          _img = cv::imread(input_filename, CV_LOAD_IMAGE_COLOR);
  if(! _img.data)
    msg_exit("Error on opening \"" + input_filename + "\" (probably not an image)!");

//-- Second conversion needed since we want to display a colored contour on a grayscale image
  cv::Mat img;
  if(grayscale) cv::cvtColor(_img, img, CV_GRAY2RGB);
  else          img = _img;
  _img.release();

//-- Determine the constants and define functionals
  max_steps = max_steps < 0 ? std::numeric_limits<int>::max() : max_steps;
  const int h = img.rows;
  const int w = img.cols;
  const int nof_channels = grayscale ? 1 : img.channels();
  const auto heaviside = std::bind(regularized_heaviside, std::placeholders::_1, eps);
  const auto delta = std::bind(regularized_delta, std::placeholders::_1, eps);

//-- Set up the video writer
  VideoWriterManager vwm;
  if(write_video)
    vwm = VideoWriterManager(input_filename, img, contour_color, fps, pos, overlay_text);

//-- Construct the level set
  cv::Mat u = levelset_checkerboard(h, w);

//-- Split the channels
  std::vector<cv::Mat> channels;
  channels.reserve(nof_channels);
  cv::split(img, channels);
  if(grayscale) channels.erase(channels.begin() + 1, channels.end());

//-- Smooth the image with Perona-Malik
  cv::Mat smoothed_img;
  if(segment)
  {
    smoothed_img = perona_malik(channels, h, w, K, L, T);
    channels.clear();
    cv::split(smoothed_img, channels);
    cv::imwrite(add_suffix(input_filename, "pm"), smoothed_img);
  }

//-- Find intensity sum and derive the stopping condition
  cv::Mat intensity_avg = cv::Mat(h, w, CV_64FC1);
#pragma omp parallel for num_threads(nof_channels)
  for(int k = 0; k < nof_channels; ++k)
  {
    cv::Mat channel(h, w, intensity_avg.type());
    channels[k].convertTo(channel, channel.type());
    intensity_avg += channel;
  }
  intensity_avg /= nof_channels;
  double stop_cond = tol * cv::norm(intensity_avg, cv::NORM_L2);
  intensity_avg.release();
  double u_diff_avg = 0;

//-- Open ncurses
  if(verbose) initscr();

//-- Timestep loop
  for(int t = 1; t <= max_steps; ++t)
  {
    cv::Mat u_diff(cv::Mat::zeros(h, w, CV_64FC1));

//-- For statistics
    std::vector<std::string> c1s, c2s;
    c1s.reserve(nof_channels);
    c2s.reserve(nof_channels);

//-- Channel loop
#pragma omp parallel for num_threads(nof_channels)
    for(int k = 0; k < nof_channels; ++k)
    {
      cv::Mat channel = channels[k];
//-- Find the average regional variances
      const double c1 = region_variance(channel, u, h, w, Region::Inside, heaviside);
      const double c2 = region_variance(channel, u, h, w, Region::Outside, heaviside);

      c1s.push_back(std::to_string(c1));
      c2s.push_back(std::to_string(c2));

//-- Calculate the contribution of one channel to the level set
      const cv::Mat variance_inside = variance_penalty(channel, h, w, c1, lambda1[k]);
      const cv::Mat variance_outside = variance_penalty(channel, h, w, c2, lambda2[k]);
      u_diff += -variance_inside + variance_outside;
    }
//-- Calculate the curvature (divergence of normalized gradient)
    const cv::Mat kappa = curvature(u, h, w);

//-- Mash the terms together
    u_diff = dt * (kappa + mu * (u_diff / nof_channels - nu));

//-- Run delta function on the level set
    cv::Mat u_cp = u.clone();
    cv::parallel_for_(cv::Range(0, h * w), ParallelPixelFunction(u_cp, w, delta));

//-- Shift the level set
    cv::multiply(u_diff, u_cp, u_diff);
    const double u_diff_norm = cv::norm(u_diff, cv::NORM_L2);
    u += u_diff;

//-- Print statistics
    if(verbose)
    {
      u_diff_avg += (u_diff_norm - u_diff_avg) / t;
      clear();
      if(max_steps != std::numeric_limits<int>::max())
        printw("\n\n\tChan-Vese completed: %f%%\n", static_cast<double>(t) / max_steps * 100);
      printw("\ti (i * dt)   = %d (%f)\n", t, t * dt);
      if(grayscale)
      {
        printw("\tc1           = %f\n", c1s[0]);
        printw("\tc2           = %f\n", c2s[0]);
      }
      else
      {
        printw("\tc1 (R, G, B) = %s\n", boost::algorithm::join(c1s, ", ").c_str());
        printw("\tc2 (R, G, B) = %s\n", boost::algorithm::join(c2s, ", ").c_str());
      }
      printw("\tu_diff (avg) = %f (%f)\n", u_diff_norm, u_diff_avg);
      refresh();
    }

//-- Save the frame
    if(write_video) vwm.write_frame(u, overlay_text ? "t = " + std::to_string(t) : "");

//-- Check if we have achieved the desired precision
    if(u_diff_norm <= stop_cond) break;
  }

//-- Close ncurses
  if(verbose) endwin();

//-- Select the region enclosed by the contour and save it to the disk
  if(object_selection)
    cv::imwrite(add_suffix(input_filename, "selection"), separate(img, u, h, w, invert));

  return EXIT_SUCCESS;
}
