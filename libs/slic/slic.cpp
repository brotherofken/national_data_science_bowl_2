#include "slic.h"

#include <array>

#include <opencv2/opencv.hpp>

#if 0
// Intersection of 3 planes, Graphics Gems 1 pg 305
static Vector3f getIntersection(const Plane& plane1, const Plane& plane2, const Plane& plane3)
{
	float det = Matrix3f::det(plane1.normal, plane2.normal, plane3.normal);

	// If the determinant is 0, that means parallel planes, no intn.
	if (det == 0.f) return 0; //could return inf or whatever

	return (plane2.normal.cross(plane3.normal)*-plane1.d +
		plane3.normal.cross(plane1.normal)*-plane2.d +
		plane1.normal.cross(plane2.normal)*-plane3.d) / det;
}
#endif

void Slic::init_data(cv::Mat3d& image)
{
	// Initialize the cluster and distance matrices.
	clusters = cv::Mat1i(image.size(), -1);
	distances = cv::Mat1d(image.size(), std::numeric_limits<double>::max());
	
	// Calculate gradient magnitude
	cv::Mat1d image_grad;
	{
		cv::Mat1d image_gray;
		cv::extractChannel(image, image_gray, 0);
		cv::Mat1d image_dx, image_dy;
		cv::Sobel(image_gray, image_dx, CV_64F, 1, 0);
		cv::Sobel(image_gray, image_dy, CV_64F, 0, 1);
		cv::magnitude(image_dx, image_dy, image_grad);
	}
	
	// Initialize the centers and counters.
	for (int i = step; i < image.cols - step / 2; i += step) {
		for (int j = step; j < image.rows - step / 2; j += step) {
			// Find the local minimum (gradient-wise).
			cv::Point2i nc = find_local_minimum(image_grad, cvPoint(i, j));
			cv::Vec3d colour = image(nc.y, nc.x);

			// Append to vector of centers.
			centers.push_back({
				centers.size(),
				{ colour[0], colour[1], colour[2] },
				{ nc.x, nc.y }
			});
			center_counts.push_back(0);
		}
	}

}


// Compute the distance between a cluster center and an individual pixel.
double Slic::compute_dist(const center_t c, cv::Point pixel, cv::Vec3d color) {
	const double color_dist = cv::norm(c.color - color); 
	const double plane_dist = cv::norm(c.coord - pixel);
	// TODO: formula slightly differs from paper, check later
	return std::sqrt(std::pow(color_dist / nc, 2) + std::pow(plane_dist / ns, 2));
}


// Find a local gradient minimum of a pixel neighborhood. 
// This method is called during initialization of cluster centers.
cv::Point Slic::find_local_minimum(cv::Mat1d& image, const cv::Point2i center, const int r /*= 5*/) {
    double min_value = std::numeric_limits<double>::max();
    cv::Point min_location = center;

	const cv::Rect roi(center.x - r, center.y - r, 2 * r, 2 * r);
	cv::minMaxLoc(image(roi), &min_value, nullptr, &min_location);
	return min_location + roi.tl();
}


// Compute the over-segmentation based on the step-size and relative weighting
// of the pixel and colour values.
void Slic::generate_superpixels(cv::Mat3d& image, const int superpixel_num, const int nc) {
	this->step = int(std::sqrt(image.total() / double(superpixel_num)));
    this->nc = nc;
    this->ns = step;
    
    // Clear previous data (if any), and re-initialize it.
    clear_data();
    init_data(image);

	const cv::Rect image_rect(cv::Point(0, 0), image.size());
    
    // Run EM iterations
    for (int iter = 0; iter < NR_ITERATIONS; iter++) {

#if defined(_DEBUG)
		// Visualization
		{
			cv::Mat1d gray;
			cv::extractChannel(image, gray, 0);
			cv::Mat3d img;
			cv::merge(std::vector<cv::Mat1d>(3, gray), img);

			this->display_contours(img, cv::Vec3d(0, 0, 255), 3.0);
			for (const auto& c : centers) {
				cv::circle(img, c.coord, 1, cv::Scalar(0, 255, 0), 1);
			}
			cv::imshow("init", img / 255);
			cv::waitKey(1);
		}
#endif

        // Reset distance values
		distances = cv::Mat1d(image.size(), std::numeric_limits<double>::max());

        for (const center_t center : centers) {
            // Compare to pixels in a 2 x step by 2 x step region
			for (int x = center.coord.x - step*1.5; x < center.coord.x + step*1.5; ++x) {
				for (int y = center.coord.y - step*1.5; y < center.coord.y + step*1.5; ++y) {
					if (image_rect.contains(cv::Point(x, y))) {
                        const cv::Vec3d color = image(y, x);
                        const double d = compute_dist(center, cv::Point(x, y), color);                        
                        // Update cluster allocation if the cluster minimizes the distance
                        if (d < distances(y, x)) {
                            distances(y, x) = d;
							clusters(y, x) = center.id;
                        }
                    }
                }
            }
        }
        
        // Clear the center values
		centers = std::vector<center_t>(centers.size(), { 0, { 0, 0, 0 }, { 0, 0 } });
		center_counts = std::vector<int>(centers.size(), 0);
        
        // Compute new cluster centers
		for (auto it = image.begin(); it != image.end(); ++it) {
			const int c_id = clusters(it.pos());
			if (c_id != -1) {
				centers[c_id].color += image(it.pos());
				centers[c_id].coord += it.pos();
				center_counts[c_id]++;
			}
		}

        // Normalize the cluster
        for (size_t ci = 0; ci < centers.size(); ++ci) {
			centers[ci].id = ci;
			centers[ci].color /= center_counts[ci]; // TODO: division by zero?
			centers[ci].coord /= center_counts[ci];
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


// Display a single pixel wide contour around the clusters.
void Slic::display_contours(cv::Mat3d& image, cv::Vec3d colour, const double scale /*= 1.0*/) {
    const std::array<int, 8> dx8 = {-1, -1,  0,  1, 1, 1, 0, -1};
	const std::array<int, 8> dy8 = { 0, -1, -1, -1, 0, 1, 1,  1};
	
	// Initialize the contour vector and the matrix detailing whether a pixel is already taken to be a contour
	std::vector<cv::Point> contours;
	cv::Mat1b istaken(image.size(), 0);
    
    for (int mx = 0; mx < image.cols; ++mx) {
        for (int my = 0; my < image.rows; ++my) {
            int nr_p = 0;
            
            // Compare the pixel to its 8 neighbors
            for (int k = 0; k < 8; k++) {
				const int x = mx + dx8[k];
				const int y = my + dy8[k];
                
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (clusters(my, mx) != clusters(y, x) && 0 == istaken(y, x)) {
                        nr_p += 1;
                    }
                }
            }
            
            // Add the pixel to the contour list if desired
            if (nr_p >= 2) {
                contours.push_back(cv::Point(mx,my));
                istaken(my, mx) = 255;
            }
        }
    }

	// Draw contour pixels
    for (size_t i = 0; i < contours.size(); ++i) {
		image(contours[i].y, contours[i].x) = colour;
    }
}


// Give the pixels of each cluster the same colour values. The specified colour is the mean RGB colour per cluster.
void Slic::colour_with_cluster_means(cv::Mat3d& image)
{
	std::vector<cv::Vec3d> colours(centers.size());
    
    // Gather the colour values per cluster.
    for (int x = 0; x < image.cols; ++x) {
        for (int y = 0; y < image.rows; ++y) {
            const int index = clusters(y, x);
			colours[index] += image(y, x);
        }
    }
    
    // Divide by the number of pixels per cluster to get the mean colour.
    for (size_t i = 0; i < colours.size(); ++i) {
        colours[i] /= center_counts[i];
    }
    
    // Fill
    for (int x = 0; x < image.cols; x++) {
        for (int y = 0; y < image.rows; y++) {
			image(y, x) = colours[clusters(y, x)];
        }
    }
}
