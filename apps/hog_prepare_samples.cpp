#include <opencv2/opencv.hpp>
#include "dicom_reader.hpp"

int main() {
	srand(42);
	const int img_num = 500;
	const int landmark_num = 16;
	const int negative_samples_per_image = 25;
	const int positive_samples_per_image = 0;
	cv::Size sample_size(32, 32);

	std::ifstream fin("dataset/lv_keypointse16_train.txt");
	int ns_cnt = 0; // negative samples counter
	int ps_cnt = 0; // positive samples counter
	for (int i = 0; i < img_num; i++) {
		std::cout << "Read image" << i << '\r' << std::flush;
		std::string image_name;
		cv::Rect2d bbox;
		fin >> image_name >> bbox.x >> bbox.y >> bbox.width >> bbox.height;
		if (bbox.width > bbox.height) {
			bbox.x -= (bbox.width - bbox.height) / 2;
			bbox.height = bbox.width;
		} else {
			bbox.y -= (bbox.height - bbox.width) / 2;
			bbox.width = bbox.height;
		}
		bbox.x -= 0.2 * bbox.width;
		bbox.y -= 0.2 * bbox.height;
		bbox.width *= 1.4;
		bbox.height *= 1.4;

		Slice slice(image_name);
		cv::Mat1d imaged = slice.image.clone();

		// Get positive ROI and write sample
		cv::Mat1d sample;
		cv::resize(imaged(bbox) * 255, sample, sample_size, 0, 0, cv::INTER_CUBIC);
		cv::imwrite("hog_pos_samples/" + std::to_string(ps_cnt++) + ".png", sample);

		for (size_t j{}; j < positive_samples_per_image; ++j) {
			cv::Mat1d sample;
			cv::Rect2d new_bbox = bbox + cv::Point2d(rand() % 6 - 3, rand() % 6 - 3); // shift augmentation
			int size_increment = rand() % std::max(int(bbox.width * 0.1), 1);
			new_bbox = new_bbox + cv::Size2d(size_increment, size_increment) - cv::Point2d(size_increment/2, size_increment/2);
			cv::resize(imaged(new_bbox) * 255, sample, sample_size, 0, 0, cv::INTER_CUBIC);
			cv::imwrite("hog_pos_samples/" + std::to_string(ps_cnt++) + ".png", sample);
		}
		// Prepare negative samples
		for (size_t j{}; j < negative_samples_per_image; ++j) {
			const size_t size = rand() % 64 + 24;

			cv::Rect2d roi = bbox;
			while ((roi & bbox).area() > bbox.area()*0.2) { // 20% intersection
				roi = cv::Rect(rand() % (imaged.cols - size), rand() % (imaged.rows - size), size, size);
			}
			cv::Mat1d sample;
			cv::resize(imaged(roi) * 255, sample, sample_size, 0, 0, cv::INTER_CUBIC);
			cv::imwrite("hog_neg_samples/" + std::to_string(ns_cnt) + ".png", sample);
			ns_cnt++;
		}

		cv::Mat1d landmarks(landmark_num, 2);
		for (int j = 0; j < landmark_num; j++) {
			fin >> landmarks(j, 0) >> landmarks(j, 1);
		}
	}
	fin.close();
	std::cout << std::endl << "Done" << std::endl;
	return 0;
}

