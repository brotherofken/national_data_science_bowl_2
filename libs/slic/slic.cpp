#include "slic.h"

#include <opencv2/opencv.hpp>

void Slic::init_data(cv::Mat3d& image)
{
	/* Initialize the cluster and distance matrices. */
	clusters = cv::Mat1i(image.size(), -1);
	distances = cv::Mat1d(image.size(), std::numeric_limits<double>::max());

	/* Initialize the centers and counters. */
	for (int i = step; i < image.cols - step / 2; i += step) {
		for (int j = step; j < image.rows - step / 2; j += step) {
			/* Find the local minimum (gradient-wise). */
			cv::Point2i nc = find_local_minimum(image, cvPoint(i, j));
			cv::Vec3d colour = image(nc.y, nc.x);

			/* Append to vector of centers. */
			centers.push_back({
				{ colour[0], colour[1], colour[2] },
				{ nc.x, nc.y }
			});
			center_counts.push_back(0);
		}
	}
}

/*
 * Compute the distance between a cluster center and an individual pixel.
 * Input : The cluster index (int), the pixel (CvPoint), and the Lab values of
 *         the pixel (CvScalar).
 * Output: The distance (double).
 */
double Slic::compute_dist(int ci, cv::Point pixel, cv::Vec3d color) {
	const double dc = cv::norm(centers[ci].color - color); 
	const double ds = cv::norm(centers[ci].coord - pixel);
	return std::sqrt(std::pow(dc / nc, 2) + std::pow(ds / ns, 2));
}

/*
 * Find a local gradient minimum of a pixel in a 3x3 neighborhood. This
 * method is called upon initialization of the cluster centers.
 *
 * Input : The image (IplImage*) and the pixel center (CvPoint).
 * Output: The local gradient minimum (cv::Point2i).
 */
CvPoint Slic::find_local_minimum(cv::Mat3d& image, const cv::Point2i center) {
    double min_grad = std::numeric_limits<double>::max();
    cv::Point2i loc_min = center;
    
    for (int x = center.x-1; x < center.x+2; x++) {
        for (int y = center.y-1; y < center.y+2; y++) {
            cv::Vec3d c1 = image(y + 1, x);
            cv::Vec3d c2 = image(y, x + 1);
            cv::Vec3d c3 = image(y, x);
            // Convert colour values to grayscale values.
            double i1 = c1[0]; // Lab - first channel
            double i2 = c2[0]; // Lab - first channel
            double i3 = c3[0]; // Lab - first channel
            //double i1 = c1.val[0] * 0.11 + c1.val[1] * 0.59 + c1.val[2] * 0.3;
            //double i2 = c2.val[0] * 0.11 + c2.val[1] * 0.59 + c2.val[2] * 0.3;
            //double i3 = c3.val[0] * 0.11 + c3.val[1] * 0.59 + c3.val[2] * 0.3;
            
            // Compute horizontal and vertical gradients and keep track of the minimum.
            if (std::sqrt(std::pow(i1 - i3, 2)) + std::sqrt(std::pow(i2 - i3,2)) < min_grad) {
				min_grad = std::abs(i1 - i3) + std::abs(i2 - i3);
				loc_min = cv::Point(x, y);
            }
        }
    }
    
    return loc_min;
}


/*
 * Compute the over-segmentation based on the step-size and relative weighting
 * of the pixel and colour values.
 *
 * Input : The Lab image (IplImage*), the stepsize (int), and the weight (int).
 * Output: -
 */

void Slic::generate_superpixels(cv::Mat3d& image, int step, int nc) {
    this->step = step;
    this->nc = nc;
    this->ns = step;
    
    /* Clear previous data (if any), and re-initialize it. */
    clear_data();
    init_data(image);
    
    /* Run EM for 10 iterations (as prescribed by the algorithm). */
    for (int iter = 0; iter < NR_ITERATIONS; iter++) {
        /* Reset distance values. */
		distances = cv::Mat1d(image.size(), std::numeric_limits<double>::max());

        for (int ci = 0; ci < int(centers.size()); ci++) {
            /* Only compare to pixels in a 2 x step by 2 x step region. */
            for (int x = centers[ci].coord.x - step; x < centers[ci].coord.x + step; ++x) {
				for (int y = centers[ci].coord.y - step; y < centers[ci].coord.y + step; ++y) {
                
                    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                        const cv::Vec3d color = image(y, x);
                        const double d = compute_dist(ci, cvPoint(x, y), color);
                        
                        // Update cluster allocation if the cluster minimizes the distance.
                        if (d < distances(y, x)) {
                            distances(y, x) = d;
                            clusters(y, x) = ci;
                        }
                    }
                }
            }
        }
        
        // Clear the center values.
        for (int ci = 0; ci < int(centers.size()); ++ci) {
			centers[ci] = { { 0, 0, 0 }, { 0, 0 } };// [0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
            center_counts[ci] = 0;
        }
        
        /* Compute the new cluster centers. */
        for (int x = 0; x < image.cols; ++x) {
            for (int y = 0; y < image.rows; ++y) {
                int c_id = clusters(y, x);
                if (c_id != -1) {
					centers[c_id].color += image(y, x);
                    centers[c_id].coord += cv::Point(x, y);
                    center_counts[c_id]++;
                }
            }
        }

        /* Normalize the clusters. */
        for (int ci = 0; ci < int(centers.size()); ci++) {
			centers[ci].color[0] /= center_counts[ci];
			centers[ci].color[1] /= center_counts[ci];
			centers[ci].color[2] /= center_counts[ci];
			centers[ci].coord.x  /= center_counts[ci];
			centers[ci].coord.y  /= center_counts[ci];
        }
    }
}

/*
 * Enforce connectivity of the superpixels. This part is not actively discussed
 * in the paper, but forms an active part of the implementation of the authors
 * of the paper.
 *
 * Input : The image (IplImage*).
 * Output: -
 */
void Slic::create_connectivity(cv::Mat3d& image) {
	int label = 0;
	int adjlabel = 0;
    const int lims = (image.cols * image.rows) / int(centers.size());
    
    const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};
    
    /* Initialize the new cluster matrix. */
    cv::Mat1i new_clusters(image.size(), -1);

    for (int mx = 0; mx < image.cols; ++mx) {
        for (int my = 0; my < image.rows; ++my) {
            if (new_clusters(my, mx) == -1) {
                std::vector<cv::Point> elements;
                elements.push_back(cv::Point(mx, my));
            
                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) {
                    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
                    
                    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                        if (new_clusters(y, x) >= 0) {
                            adjlabel = new_clusters(y, x);
                        }
                    }
                }
                
                int count = 1;
                for (int c = 0; c < count; c++) {
                    for (int k = 0; k < 4; k++) {
                        int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
                        
                        if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                            if (new_clusters(y, x) == -1 && clusters(my, mx) == clusters(y,x)) {
                                elements.push_back(cv::Point(x, y));
                                new_clusters(y, x) = label;
                                count += 1;
                            }
                        }
                    }
                }
                
                /* Use the earlier found adjacent label if a segment size is
                   smaller than a limit. */
                if (count <= lims >> 2) {
                    for (int c = 0; c < count; c++) {
						new_clusters(elements[c].y, elements[c].x) = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }
        }
    }
}

/*
 * Display the cluster centers.
 *
 * Input : The image to display upon (IplImage*) and the colour (CvScalar).
 * Output: -
 */
void Slic::display_center_grid(cv::Mat3d& image, CvScalar colour) {
    for (int i = 0; i < (int) centers.size(); i++) {
        cv::circle(image, cv::Point(centers[i].coord.x, centers[i].coord.y), 2, colour, 2);
    }
}

/*
 * Display a single pixel wide contour around the clusters.
 *
 * Input : The target image (IplImage*) and contour colour (CvScalar).
 * Output: -
 */
void Slic::display_contours(cv::Mat3d& image, cv::Vec3d colour, const double scale /*= 1.0*/) {
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	
	/* Initialize the contour vector and the matrix detailing whether a pixel
	 * is already taken to be a contour. */
	std::vector<cv::Point> contours;
	cv::Mat1b istaken(image.size(), 0);
	//for (int i = 0; i < image.cols; i++) { 
	//	std::vector<bool> nb;
    //    for (int j = 0; j < image.rows; j++) {
    //        nb.push_back(false);
    //    }
    //    istaken.push_back(nb);
    //}
    
    /* Go through all the pixels. */
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            int nr_p = 0;
            
            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k], y = j + dy8[k];
                
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                    if (istaken(y, x) == 0 && clusters(j, i) != clusters(y, x)) {
                        nr_p += 1;
                    }
                }
            }
            
            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                contours.push_back(cv::Point(i,j));
                istaken(j, i) = 255;
            }
        }
    }

	/* Draw the contour pixels. */
    for (int i = 0; i < (int)contours.size(); i++) {
		image(contours[i].y, contours[i].x) = colour;
    }
}

/*
 * Give the pixels of each cluster the same colour values. The specified colour
 * is the mean RGB colour per cluster.
 *
 * Input : The target image (IplImage*).
 * Output: -
 */
void Slic::colour_with_cluster_means(cv::Mat3d& image)
{
	std::vector<cv::Vec3d> colours(centers.size());
    
    /* Gather the colour values per cluster. */
    for (int x = 0; x < image.cols; ++x) {
        for (int y = 0; y < image.rows; ++y) {
            const int index = clusters(y, x);
			colours[index] += image(y, x);
        }
    }
    
    /* Divide by the number of pixels per cluster to get the mean colour. */
    for (size_t i = 0; i < colours.size(); ++i) {
        colours[i][0] /= center_counts[i];
        colours[i][1] /= center_counts[i];
        colours[i][2] /= center_counts[i];
    }
    
    /* Fill in. */
    for (int x = 0; x < image.cols; x++) {
        for (int y = 0; y < image.rows; y++) {
			image(y, x) = colours[clusters(y, x)];
        }
    }
}
