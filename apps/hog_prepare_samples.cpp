#include <string>
#include <fstream>
#include <iterator>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/regex/regex_traits.hpp>

#include <opencv2/opencv.hpp>
#include "dicom_reader.hpp"

#include "hog_parameters.hpp"

namespace bfs = boost::filesystem;

struct bbox_annotation {
	std::string path;
	cv::Rect roi;
};

std::vector< std::string > find_files_by_mask(const bfs::path& path, const boost::regex& filter)
{
	std::vector< std::string > result;
	for (bfs::directory_iterator i(path); i != bfs::directory_iterator(); ++i) {
		// Skip if not a file
		if (!bfs::is_regular_file(i->status())) continue;

		// Skip if no match
		boost::smatch what;
		const std::string filename = i->path().filename().string();
		if (!boost::regex_match(filename, what, filter)) continue;

		// File matches, store it
		result.push_back(filename);
	}
	return result;
}

struct csv_reader : std::ctype<char> {
	csv_reader() : std::ctype<char>(get_table()) {}
	static std::ctype_base::mask const* get_table() {
		static std::vector<std::ctype_base::mask> rc(table_size, std::ctype_base::mask());

		rc[','] = std::ctype_base::space;
		rc['\n'] = std::ctype_base::space;
		rc[' '] = std::ctype_base::space;
		return &rc[0];
	}
};

std::vector<bbox_annotation> load_annotation(const bfs::path& annotations_path, const std::vector<std::string>& positive_files)
{
	std::vector<bbox_annotation> result;
	for (const std::string& filename : positive_files) {
		std::string full_name = (annotations_path / bfs::path(filename)).string();
		std::ifstream fin(full_name);
		std::vector<std::string> lines ;
		std::copy(std::istream_iterator<std::string>(fin), std::istream_iterator<std::string>(), std::back_inserter(lines));

		for (const std::string& l : lines) {

			std::stringstream ss(l);
			ss.imbue(std::locale(ss.getloc(), new csv_reader()));
			bbox_annotation annotation;
			ss >> annotation.path >> annotation.roi.x >> annotation.roi.y >> annotation.roi.width >> annotation.roi.height;
			result.push_back(annotation);
		}
	}
	return result;
}


int main(int argc, char** argv) {
	srand(42);

	bfs::path annotations_path(argv[1]);
	bfs::path data_path(argv[2]);

	const boost::regex filter_positives(".*NDSB_[\\d]*_positive\.csv");
	const boost::regex filter_negatives(".*NDSB_[\\d]*_negative\.csv");

	std::vector< std::string > positive_files = find_files_by_mask(annotations_path, filter_positives);
	std::vector< std::string > negative_files = find_files_by_mask(annotations_path, filter_negatives);

	std::vector<bbox_annotation> positive_samples = load_annotation(annotations_path, positive_files);
	std::vector<bbox_annotation> negative_samples = load_annotation(annotations_path, negative_files);

	const int negative_samples_per_image = 5;

	int ps_cnt = 0;
	int ns_cnt = 0;

	for (auto sample : negative_samples) {
		std::cout << sample.path << std::endl;

		const std::string image_name = (data_path / "train" / sample.path).string();

		Slice slice(image_name);
		cv::Mat1d imaged = slice.image.clone();

		cv::Rect bbox = sample.roi;
		bbox = bbox & cv::Rect({ 0,0 }, imaged.size());
		if (bbox.size() != sample.roi.size()) continue;

		cv::Mat1d sample;
		cv::resize(imaged(bbox), sample, hog::SAMPLE_SIZE, 0, 0, cv::INTER_CUBIC);
		cv::imwrite("hog_neg_samples/" + std::to_string(ns_cnt++) + ".png", sample);
	}

	for (auto sample : positive_samples) {
		std::cout << sample.path << std::endl;

		const std::string image_name = (data_path / "train" / sample.path).string();

		Slice slice(image_name);
		cv::Mat1d imaged = slice.image.clone();

		cv::Rect bbox = sample.roi;
		bbox = bbox & cv::Rect({ 0,0 }, imaged.size());
		if (bbox.size() != sample.roi.size()) continue;

		cv::Mat1d sample;
		cv::resize(imaged(bbox), sample, hog::SAMPLE_SIZE, 0, 0, cv::INTER_CUBIC);
		cv::imwrite("hog_pos_samples/" + std::to_string(ps_cnt++) + ".png", sample);

		for (size_t j{}; j < negative_samples_per_image; ++j) {
			const size_t size = rand() % 64 + 24;

			cv::Rect roi = bbox;
			while ((roi & bbox).area() > bbox.area()*0.2) { // 20% intersection
				roi = cv::Rect(rand() % (imaged.cols - size), rand() % (imaged.rows - size), size, size);
			}
			cv::Mat1d sample;
			cv::resize(imaged(roi), sample, hog::SAMPLE_SIZE, 0, 0, cv::INTER_CUBIC);
			cv::imwrite("hog_neg_samples/" + std::to_string(ns_cnt) + ".png", sample);
			ns_cnt++;
		}
	}



#if 0
	// Old code
	const int img_num = 500;
	const int landmark_num = 16;
	const int negative_samples_per_image = 25;
	const int positive_samples_per_image = 0;

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
#endif
}

