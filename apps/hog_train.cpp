#include <opencv2/opencv.hpp>
#include "dicom_reader.hpp"

#include "hog_parameters.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <time.h>

//using namespace cv;
//using namespace cv::ml;
//using namespace std;

void get_svm_detector(const cv::Ptr<cv::ml::SVM>& svm, std::vector< float > & hog_detector);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void load_images(const std::string & prefix, const size_t count, std::vector< cv::Mat > & img_lst);
void sample_neg(const std::vector< cv::Mat > & full_neg_lst, std::vector< cv::Mat > & neg_lst, const cv::Size & size);
cv::Mat get_hogdescriptor_visu(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size & size);
void compute_hog(const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, const cv::Size & size);
void train_svm(const std::vector< cv::Mat > & gradient_lst, const std::vector< int > & labels);
void draw_locations(cv::Mat & img, const std::vector< cv::Rect > & locations, const cv::Scalar & color);
void test_it(const cv::Size & size);

void get_svm_detector(const cv::Ptr<cv::ml::SVM>& svm, std::vector< float > & hog_detector)
{
	// get the support vectors
	cv::Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	cv::Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}


/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a cv::Matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	std::vector< cv::Mat >::const_iterator itr = train_samples.begin();
	std::vector< cv::Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}

void load_images(const std::string & prefix, const size_t count, std::vector< cv::Mat > & img_lst)
{
	std::string line;

	bool end_of_parsing = false;
	for (size_t i{}; i < count; ++i)
	{
		cv::Mat img = cv::imread((prefix + "/" + std::to_string(i) + ".png")); // load the image
		if (img.empty()) // invalid image, just skip it.
			continue;
#ifdef _DEBUG
		cv::imshow("image", img);
		cv::waitKey(10);
#endif
		img_lst.push_back(img.clone());
	}
}

void sample_neg(const std::vector< cv::Mat > & full_neg_lst, std::vector< cv::Mat > & neg_lst, const cv::Size & size)
{
	cv::Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	srand((unsigned int)time(NULL));

	std::vector< cv::Mat >::const_iterator img = full_neg_lst.begin();
	std::vector< cv::Mat >::const_iterator end = full_neg_lst.end();
	for (; img != end; ++img)
	{
		box.x = rand() % (img->cols - size_x);
		box.y = rand() % (img->rows - size_y);
		cv::Mat roi = (*img)(box);
		neg_lst.push_back(roi.clone());
#ifdef _DEBUG
		cv::imshow("img", roi.clone());
		cv::waitKey(10);
#endif
	}
}

// From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
cv::Mat get_hogdescriptor_visu(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	cv::Mat visu;
	resize(color_origImg, visu, cv::Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y < cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x < cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin < gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx < blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky < blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr < 4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin < gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (celly = 0; celly < cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx < cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin < gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly < cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx < cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, cv::Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), cv::Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), cv::Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin < gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, cv::Point((int)(x1*zoomFac), (int)(y1*zoomFac)), cv::Point((int)(x2*zoomFac), (int)(y2*zoomFac)), cv::Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y < cells_in_y_dir; y++)
	{
		for (int x = 0; x < cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu

void compute_hog(const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, const cv::Size & size)
{
	cv::HOGDescriptor hog(size, cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1, -1.0, cv::HOGDescriptor::L2Hys, 0.05, false, 64, false);
	cv::Mat gray;
	std::vector<cv::Point> location;
	std::vector<float> descriptors;

	std::vector< cv::Mat >::const_iterator img = img_lst.begin();
	std::vector< cv::Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img)
	{
		cvtColor(*img, gray, cv::COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, cv::Size(8, 8), cv::Size(0, 0), location);
		gradient_lst.push_back(cv::Mat(descriptors).clone());
#ifdef _DEBUG
		cv::imshow("gradient", get_hogdescriptor_visu(img->clone(), descriptors, size));
		cv::waitKey(10);
#endif
	}
}

void train_svm(const std::vector< cv::Mat > & gradient_lst, const std::vector< int > & labels)
{

	cv::Mat train_data_mat;
	convert_to_ml(gradient_lst, train_data_mat);

	std::clog << "Start training...";
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	/* Default values to train SVM */

	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.001); // From paper, soft classifier
	svm->setType(cv::ml::SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	//cv::Mat1d class_weights(1, 2);
	//class_weights(0) = 1.0;
	//class_weights(1) = 2.0;
	//svm->setClassWeights(class_weights);
	//svm->train(train_data_mat, cv::ml::ROW_SAMPLE, cv::Mat(labels));
	
	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(train_data_mat, cv::ml::ROW_SAMPLE, cv::Mat(labels));
	svm->trainAuto(train_data, 10
		, cv::ml::ParamGrid(0.001, 1.0, 10) //cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C)      // cv::ml::ParamGrid Cgrid =      
		, cv::ml::ParamGrid(0, 0, 0) //cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA) //cv::ml::ParamGrid(0, 0, 0)       // // cv::ml::ParamGrid gammaGrid =  
		, cv::ml::ParamGrid(0.01,1.0,10)     //cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P)      // cv::ml::ParamGrid pGrid =      
		, cv::ml::ParamGrid(0.1,0.1,0)     //cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU)     // cv::ml::ParamGrid nuGrid =     
		, cv::ml::ParamGrid(0, 0, 0)       //cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF)   // cv::ml::ParamGrid coeffGrid =  
		, cv::ml::ParamGrid(1, 1, 0)       //cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE) // cv::ml::ParamGrid degreeGrid = 
		);
	std::clog << "...[done]" << std::endl;
	svm->save("lv_detector.yml");
}

void draw_locations(cv::Mat & img, const std::vector< cv::Rect > & locations, const cv::Scalar & color, double scale)
{
	if (!locations.empty())
	{
		std::vector< cv::Rect >::const_iterator loc = locations.begin();
		std::vector< cv::Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc) {
			cv::rectangle(img, cv::Rect((*loc).tl() * scale, (*loc).br() * scale), color, 1);
		}
	}
}

void test_it(const cv::Size & size)
{
	char key = 27;
	cv::Scalar reference(0, 255, 0);
	cv::Scalar trained(0, 0, 255);
	cv::Mat img, draw;
	cv::Ptr<cv::ml::SVM> svm;
	cv::HOGDescriptor hog(size, cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1, -1.0, cv::HOGDescriptor::L2Hys, 0.05, false, 64, false);
	std::vector< cv::Rect > locations;

	// Load the trained SVM.

	svm = cv::ml::StatModel::load<cv::ml::SVM>("lv_detector.yml");
	// Set the trained svm to my_hog
	std::vector< float > hog_detector;
	get_svm_detector(svm, hog_detector);
	hog.setSVMDetector(hog_detector);

	// Open the camera.
	int test_img_num = 49;
	int landmark_num = 16;

	//std::ifstream fin("dataset/lv_keypointse16_test.txt");
	std::ifstream fin("dataset/lv_keypointse16_train.txt");
	for (int i = 0; i < test_img_num; i++) {
		std::string image_name;
		double skip;
		cv::Rect2d bbox;
		fin >> image_name >> bbox.x >> bbox.y >> bbox.width >> bbox.height;
		for (int j = 0; j < landmark_num; j++) { fin >> skip >> skip; }
		const double bbox_scale = 1.5;
		
		bbox.x -= bbox_scale * bbox.width;
		bbox.y -= bbox_scale * bbox.height;
		bbox.width +=  2. * bbox_scale*bbox.width;
		bbox.height += 2. * bbox_scale*bbox.height;


		Slice slice(image_name);
		bbox = bbox & cv::Rect2d({ 0., 0. }, slice.image.size());
		cv::Mat1d imaged = slice.image(bbox).clone();

		cv::resize(imaged, imaged, cv::Size(), slice.pixel_spacing[0], slice.pixel_spacing[1]);
		std::cout << i << " " << slice.pixel_spacing[0] << " " << slice.pixel_spacing[1] << std::endl;
		double scale = 1.0;
		//if (imaged.cols > 256 || imaged.rows > 256) {
		//	scale = 256. / imaged.cols;
		//	cv::resize(imaged, imaged, cv::Size(), scale, scale, cv::INTER_CUBIC);
		//}
		draw = imaged.clone();
		cv::Mat1b image;
		imaged.convertTo(image, image.type());
		locations.clear();

		hog.detectMultiScale(image, locations, 0.00, cv::Size(), cv::Size(), 1.05, 2.);// , -0.95, cv::Size(1, 1), cv::Size(0, 0), 1.2, 2.0);
		cv::merge(std::vector<cv::Mat1d>(3, draw), draw);

		draw_locations(draw, locations, trained, scale);

		imshow("Test", draw/255);
		key = char(cv::waitKey(0));
		std::cout << (key == ' ' ? "bad" : "") << std::endl;
	}
	fin.close();
}

int main(int argc, char** argv)
{
	std::vector<cv::Mat> pos_lst;
	std::vector<cv::Mat> full_neg_lst;
	std::vector<cv::Mat> neg_lst;
	std::vector<cv::Mat> gradient_lst;
	std::vector<int> labels;
	std::string pos_dir(argv[1]);
	size_t pos(std::stoi(argv[2]));
	std::string neg_dir(argv[3]);
	size_t neg(std::stoi(argv[4]));

	if (pos_dir.empty() || pos == 0 || neg_dir.empty() || neg == 0)
	{
		std::cout << "Wrong number of parameters." << std::endl
			<< "Usage: " << argv[0] << " --pd=pos_dir -p=pos.lst --nd=neg_dir -n=neg.lst" << std::endl
			<< "example: " << argv[0] << " --pd=/INRIA_dataset/ -p=Train/pos.lst --nd=/INRIA_dataset/ -n=Train/neg.lst" << std::endl;
		exit(-1);
	}

	const cv::Size sample_size = hog::SAMPLE_SIZE;
	bool skip_train = std::stoi(argv[5]);
	if (!skip_train) {
		std::cout << "Loading pos." << std::endl;
		load_images(pos_dir, pos, pos_lst);
		std::cout << pos_lst.size() << std::endl;
		labels.assign(pos_lst.size(), +1);
		const unsigned int old = (unsigned int)labels.size();
		std::cout << "Loading neg." << std::endl;
		load_images(neg_dir, neg, neg_lst);
		std::cout << neg_lst.size() << std::endl;
		//sample_neg(full_neg_lst, neg_lst, sample_size);
		labels.insert(labels.end(), neg_lst.size(), -1);
		CV_Assert(old < labels.size());

		std::cout << "Computing HOG." << std::endl;
		compute_hog(pos_lst, gradient_lst, sample_size);
		compute_hog(neg_lst, gradient_lst, sample_size);

		std::cout << "Train svm." << std::endl;
		train_svm(gradient_lst, labels);
	}

	test_it(sample_size); // change with your parameters

	return 0;
}
