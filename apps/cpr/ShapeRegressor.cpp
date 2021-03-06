/*
Author: Bi Sai 
Date: 2014/06/18
This program is a reimplementation of algorithms in "Face Alignment by Explicit 
Shape Regression" by Cao et al.
If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

Copyright (c) 2014 Bi Sai 
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include "FaceAlignment.h"
using namespace std;
using namespace cv;

ShapeRegressor::ShapeRegressor(){
    first_level_num_ = 0;
}

/**
 * @param images gray scale images
 * @param ground_truth_shapes a vector of N*2 matrix, where N is the number of landmarks
 * @param bounding_box BoundingBox of faces
 * @param first_level_num number of first level regressors
 * @param second_level_num number of second level regressors
 * @param candidate_pixel_num number of pixels to be selected as features
 * @param fern_pixel_num number of pixel pairs in a fern
 * @param initial_num number of initial shapes for each input image
 */
void ShapeRegressor::Train(const vector<Mat_<uchar> >& images, 
                   const vector<Mat_<double> >& ground_truth_shapes,
                   const vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num){
    cout<<"Start training..."<<endl;
    bounding_box_ = bounding_box;
    training_shapes_ = ground_truth_shapes;
    first_level_num_ = first_level_num;
    landmark_num_ = ground_truth_shapes[0].rows; 
    // data augmentation and multiple initialization 
    vector<Mat_<uchar> > augmented_images;
    vector<BoundingBox> augmented_bounding_box;
    vector<Mat_<double> > augmented_ground_truth_shapes;
    vector<Mat_<double> > current_shapes;
     
    RNG random_generator(getTickCount());
    for(int i = 0;i < images.size();i++){
        for(int j = 0;j < initial_num;j++){
            int index = 0;
            do{
                // index = (i+j+1) % (images.size()); 
                index = random_generator.uniform(0, images.size());
            }while(index == i);
            augmented_images.push_back(images[i]);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);

			BoundingBox aug_bbox = bounding_box[i];
			aug_bbox.height *= random_generator.uniform(0.5, 1.2);// random_generator.uniform(bounding_box[i].height*0.0, bounding_box[i].height * 1.0);
			aug_bbox.width *= aug_bbox.height / bounding_box[i].height; // random_generator.uniform(bounding_box[i].width*0.0, bounding_box[i].width * 0.5);
			aug_bbox.start_x -= (aug_bbox.height - bounding_box[i].height) / 2;
			aug_bbox.start_y -= (aug_bbox.width - bounding_box[i].width) / 2;
			aug_bbox.centroid_x = aug_bbox.start_x + aug_bbox.width / 2.0;
			aug_bbox.centroid_y = aug_bbox.start_y + aug_bbox.height / 2.0;

            augmented_bounding_box.push_back(aug_bbox);
            // 1. Select ground truth shapes of other images as initial shapes
            // 2. Project current shape to bounding box of ground truth shapes 
            Mat_<double> temp = ground_truth_shapes[index];
            temp = ProjectShape(temp, bounding_box[index]);
			temp = ReProjectShape(temp, aug_bbox/*bounding_box[i]*/);

#if _DEBUG
			{
				cv::Mat img = images[i].clone();
				cv::rectangle(img, cv::Rect(aug_bbox.start_x, aug_bbox.start_y, aug_bbox.width, aug_bbox.height), 255);

				for (int i = 0; i < 15; i++) {
					cv::circle(img, cv::Point2d(temp(i, 0), temp(i, 1)), 1, cv::Scalar(255), -1);
				}

				cv::imshow("augmented sample", img);
				cv::waitKey(0);
			}
#endif
            current_shapes.push_back(temp); 
        } 
    }
    
    // get mean shape from training shapes
    mean_shape_ = GetMeanShape(ground_truth_shapes,bounding_box); 
    
    // train fern cascades
    fern_cascades_.resize(first_level_num);
    vector<Mat_<double> > prediction;
    for(int i = 0;i < first_level_num;i++){
        cout<<"Training fern cascades: "<<i+1<<" out of "<<first_level_num<<endl;
        prediction = fern_cascades_[i].Train(augmented_images,current_shapes,
                augmented_ground_truth_shapes,augmented_bounding_box,mean_shape_,second_level_num,candidate_pixel_num,fern_pixel_num, i+1, first_level_num);
        
        // update current shapes 
        for(int j = 0;j < prediction.size();j++){
            current_shapes[j] = prediction[j] + ProjectShape(current_shapes[j], augmented_bounding_box[j]);
            current_shapes[j] = ReProjectShape(current_shapes[j],augmented_bounding_box[j]);
        }
    } 
    
}


void ShapeRegressor::Write(std::ostream& fout)
{
#if defined(BINARY_IO)
	io::write_scalar(fout, first_level_num_);
	io::write_scalar(fout, landmark_num_);

	io::write_mat(fout, mean_shape_);

	//fout << 
	long training_num_ = training_shapes_.size();
	fout.write(reinterpret_cast<char*>(&training_num_), 4);

	io::write_vector(fout, bounding_box_);
	for (int i = 0; i < training_shapes_.size(); i++) {
		io::write_mat(fout, training_shapes_[i]);
	}

	for (int i = 0; i < first_level_num_; i++) {
		fern_cascades_[i].Write(fout);
	}
#else
	fout << first_level_num_ << std::endl;
	fout << mean_shape_.rows << std::endl;
	for (int i = 0; i < landmark_num_; i++) {
		fout << mean_shape_(i, 0) << " " << mean_shape_(i, 1) << " ";
	}
	fout << std::endl;

	fout << training_shapes_.size() << std::endl;
	for (int i = 0; i < training_shapes_.size(); i++) {
		fout << bounding_box_[i].start_x << " " << bounding_box_[i].start_y << " "
			<< bounding_box_[i].width << " " << bounding_box_[i].height << " "
			<< bounding_box_[i].centroid_x << " " << bounding_box_[i].centroid_y << std::endl;
		for (int j = 0; j < training_shapes_[i].rows; j++) {
			fout << training_shapes_[i](j, 0) << " " << training_shapes_[i](j, 1) << " ";
		}
		fout << std::endl;
	}

	for (int i = 0; i < first_level_num_; i++) {
		fern_cascades_[i].Write(fout);
	}
#endif
}

void ShapeRegressor::Read(std::istream& fin)
{
#if defined(BINARY_IO)
	io::read_scalar(fin, first_level_num_);
	io::read_scalar(fin, landmark_num_);

	mean_shape_ = cv::Mat(landmark_num_, 2, CV_64FC1);
	io::read_mat(fin, mean_shape_);

	long training_num;
	fin.read(reinterpret_cast<char*>(&training_num), 4);
	//fin >> training_num;
	training_shapes_.resize(training_num);
	bounding_box_.resize(training_num);

	io::read_vector(fin, bounding_box_);
	for (int i = 0; i < training_num; i++) {
		cv::Mat1d temp1(landmark_num_, 2);

		io::read_mat(fin, temp1);
		training_shapes_[i] = temp1;
	}

	fern_cascades_.resize(first_level_num_);
	for (int i = 0; i < first_level_num_; i++) {
		fern_cascades_[i].Read(fin);
	}
#else
	fin >> first_level_num_;
	fin >> landmark_num_;
	mean_shape_ = Mat::zeros(landmark_num_, 2, CV_64FC1);
	for (int i = 0; i < landmark_num_; i++) {
		fin >> mean_shape_(i, 0) >> mean_shape_(i, 1);
	}

	int training_num;
	fin >> training_num;
	training_shapes_.resize(training_num);
	bounding_box_.resize(training_num);

	for (int i = 0; i < training_num; i++) {
		BoundingBox temp;
		fin >> temp.start_x >> temp.start_y >> temp.width >> temp.height >> temp.centroid_x >> temp.centroid_y;
		bounding_box_[i] = temp;

		cv::Mat1d temp1(landmark_num_, 2);
		for (int j = 0; j < landmark_num_; j++) {
			fin >> temp1(j, 0) >> temp1(j, 1);
		}
		training_shapes_[i] = temp1;
	}

	fern_cascades_.resize(first_level_num_);
	for (int i = 0; i < first_level_num_; i++) {
		fern_cascades_[i].Read(fin);
	}
#endif
}


Mat1d ShapeRegressor::Predict(const Mat1b& image, const BoundingBox& _bounding_box, int initial_num, const cv::Mat1d& initial_contour)
{
	// generate multiple initializations
	Mat1d result = Mat::zeros(landmark_num_, 2, CV_64FC1);
	RNG random_generator(getTickCount());

	//if (!initial_contour.empty()) initial_num = 1;
	BoundingBox orig_bbox = _bounding_box;
	for (int i = 0; i < initial_num; i++) {
		random_generator = RNG(i);
		//int index = random_generator.uniform(0, training_shapes_.size());
		Mat1d current_shape;
		current_shape = mean_shape_.clone();
		BoundingBox bounding_box = orig_bbox;
		if (i != 0) {
			for (size_t r{}; r < current_shape.rows; ++r) {
				for (size_t c{}; c < current_shape.cols; ++c) {
					current_shape(r, c) += random_generator.uniform(-0.125, 0.125); // Random jiggling
				}
			}
			//BoundingBox aug_bbox = bounding_box[i];
			bounding_box.height *= random_generator.uniform(0.5, 1.0);// random_generator.uniform(bounding_box[i].height*0.0, bounding_box[i].height * 1.0);
			bounding_box.width *= bounding_box.height / orig_bbox.height; // random_generator.uniform(bounding_box[i].width*0.0, bounding_box[i].width * 0.5);
			bounding_box.start_x -= (bounding_box.height - orig_bbox.height) / 2;
			bounding_box.start_y -= (bounding_box.width - orig_bbox.width) / 2;
			bounding_box.centroid_x = bounding_box.start_x + bounding_box.width / 2.0;
			bounding_box.centroid_y = bounding_box.start_y + bounding_box.height / 2.0;
		}

		//BoundingBox current_bounding_box = bounding_box_[index];
		//if (i != 0) current_shape = ProjectShape(current_shape, bounding_box);// current_bounding_box);
		current_shape = ReProjectShape(current_shape, bounding_box);

		for (int j = 0; j < first_level_num_; j++) {
			Mat1d prediction = fern_cascades_[j].Predict(image, bounding_box, mean_shape_, current_shape);
			// update current shape
			current_shape = prediction + ProjectShape(current_shape, bounding_box);
			current_shape = ReProjectShape(current_shape, bounding_box);
		}
		result = result + current_shape;
	}

	return 1.0 / initial_num * result;
}

void ShapeRegressor::Load(string path){
    cout<<"Loading model..."<<endl;
    ifstream fin;
    fin.open(path, ios::binary);
    this->Read(fin); 
    fin.close();
    cout<<"Model loaded successfully..."<<endl;
}

void ShapeRegressor::Save(string path){
    cout<<"Saving model..."<<endl;
    ofstream fout;
    fout.open(path, ios::binary);
    this->Write(fout);
    fout.close();
}


