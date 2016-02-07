#ifndef SLIC_H
#define SLIC_H

/* slic.h.
 *
 * Written by: Pascal Mettes.
 *
 * This file contains the class elements of the class Slic. This class is an
 * implementation of the SLIC Superpixel algorithm by Achanta et al. [PAMI'12,
 * vol. 34, num. 11, pp. 2274-2282].
 *
 * This implementation is created for the specific purpose of creating
 * over-segmentations in an OpenCV-based environment.
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>

/*
 * class Slic.
 *
 * In this class, an over-segmentation is created of an image, provided by the
 * step-size (distance between initial cluster locations) and the colour
 * distance parameter.
 */
class Slic {
	public:
		/* The number of iterations run by the clustering algorithm. */
		static const unsigned int NR_ITERATIONS = 25;

		const cv::Mat1i& get_clusters() const { return clusters; }
		const cv::Rect& get_roi() const { return roi; }

    private:
        /* The cluster assignments and distance values for each pixel. */
        cv::Mat1i clusters;
        cv::Mat1d distances;
        
		cv::Rect roi;

        /* The LAB and xy values of the centers. */
		struct center_t {
			size_t id;
			double color;
			cv::Point2i coord;
		};
		std::vector<center_t> centers;
        /* The number of occurrences of each center. */
        std::vector<int> center_counts;
        
        /* The step size per cluster, and the color (nc) and distance (ns) parameters. */
        int step, ns;
		double nc;
        
        /* Compute the distance between a center and an individual pixel. */
		inline double compute_dist(const center_t c, cv::Point pixel, const double& colour);
        /* Find the pixel with the lowest gradient in a 3x3 surrounding. */
		cv::Point find_local_minimum(cv::Mat1d& image, const cv::Point2i center, const int r = 5);
        
		// Clear the data as saved by the algorithm.
        void clear_data()
		{
			clusters.release();
			distances.release();
			centers.clear();
			center_counts.clear();
		}

		/*
		* Initialize the cluster centers and initial values of the pixel-wise cluster
		* assignment and distance values.
		*
		* Input : The image (IplImage*).
		* Output: -
		*/
        void init_data(cv::Mat1d& image);


    public:
        // Class constructors and destructors.
		Slic(){}
		~Slic(){}
        
        // Generate an over-segmentation for an image.
		void generate_superpixels(cv::Mat1d& image, const int superpixel_num, const double nc, cv::Rect roi = cv::Rect());
        // Enforce connectivity for an image.
		void create_connectivity(cv::Mat1d& image);
        
        // Draw functions.
		void display_center_grid(cv::Mat3d& image, const cv::Scalar& colour);
		void display_contours(cv::Mat3d& image, const cv::Scalar& colour, const double scale = 1.0);
        void colour_with_cluster_means(cv::Mat3d& image);
};

#endif
