// -*- c++ -*-
//
// Time-stamp: <2015-04-22 09:59:37 zophos>
//
#define DEBUG

#include <iostream>
#include <fstream>

#include <gdcmReader.h>
#include <gdcmImageReader.h>
#include <gdcmAttribute.h>


#include <opencv2/opencv.hpp>

inline bool ends_with(const std::string& s, const std::string& end)
{
	if (end.size() > s.size()) return false;
	return std::equal(end.rbegin(), end.rend(), s.rbegin());
}

struct Slice
{
	cv::Mat1d image;
	cv::Vec3d row_dc;
	cv::Vec3d col_dc;
	cv::Vec3d position;
	cv::Vec3d pixel_spacing;
	double intercept;

	cv::Mat1d rotation_matrix() const
	{
		cv::Mat1d rm;
		rm.push_back(row_dc * pixel_spacing[0]);
		rm.push_back(col_dc * pixel_spacing[0]);
		rm.push_back(normal() * pixel_spacing[0]);
		return rm.reshape(1, 3);
	}
	cv::Vec3d normal() const { return row_dc.cross(col_dc); };

	cv::Point2d project_point(const cv::Vec3d& p)
	{
		const cv::Mat1d projection = rotation_matrix().inv().t() * cv::Mat1d(p - position);
		intercept = projection(1, 0);

		return{ projection(0, 0), projection(1,0) };
	}
};

Slice read_dcm(const std::string& filename)
{
	//gdcm::Reader reader;
	//reader.SetFileName(filename.c_str());
	//if (!reader.Read()) {
	//	return Slice();
	//}
	//const gdcm::File &file = reader.GetFile();
	//const gdcm::DataSet &ds = file.GetDataSet();
	//std::cout << ds.GetDataElement(gdcm::PrivateTag(gdcm::Tag(0x0020, 0x1041))) << std::endl;

	// Read Imageimage_position
	gdcm::ImageReader ir;
	ir.SetFileName(filename.c_str());
	if (!ir.Read()) {
		std::cerr << "Could not read: " << filename << std::endl;
		return Slice();
	}

	const gdcm::Image &gimage = ir.GetImage();

	std::vector<short> vbuffer(gimage.GetBufferLength());
	gimage.GetBuffer((char*)&vbuffer[0]);

	//const unsigned int* const dimension = gimage.GetDimensions();
	const unsigned int size_x = gimage.GetDimensions()[0];
	const unsigned int size_y = gimage.GetDimensions()[1];
	cv::Mat1d image(size_y, size_x);
	std::copy(vbuffer.begin(), vbuffer.end(), image.begin());

	return Slice{
				image,
				cv::Vec3d(gimage.GetDirectionCosines()),
				cv::Vec3d(gimage.GetDirectionCosines() + 3),
				cv::Vec3d(gimage.GetOrigin()),
				cv::Vec3d(gimage.GetSpacing()),
				gimage.GetIntercept()
			};
}

cv::Vec3d slices_intersection(const Slice& s1, const Slice& s2, const Slice& s3)
{
	cv::Mat1d normals;
	normals.push_back(s1.normal());
	normals.push_back(s2.normal());
	normals.push_back(s3.normal());
	normals = normals.reshape(1, 3);

	cv::Mat1d d = (cv::Mat1d(3, 1) << 
		s1.normal().dot(s1.position), 
		s2.normal().dot(s2.position), 
		s3.normal().dot(s3.position)
		);
		
	cv::Mat1d intersection;
	cv::solve(normals, d, intersection, cv::DECOMP_SVD);
	return cv::Vec3d(intersection);
}

int main(int argc, char *argv[])
{
	if (argc < 4) {
		return EXIT_FAILURE;
	}
	const std::string slicecname_2ch = argv[1];
	const std::string slicecname_4ch = argv[2];
	const std::string slicecname_sax = argv[3];

	if (!ends_with(slicecname_2ch,"dcm") || !ends_with(slicecname_4ch, "dcm") || !ends_with(slicecname_sax, "dcm")) {
		std::cerr << "skip " << slicecname_sax << std::endl;
		return EXIT_FAILURE;
	}

	Slice slice_2ch = read_dcm(slicecname_2ch);
	Slice slice_4ch = read_dcm(slicecname_4ch);
	Slice slice_sax = read_dcm(slicecname_sax);

	cv::Vec3d intersection = slices_intersection(slice_2ch, slice_4ch, slice_sax);
	cv::Point2d intersection_sax = slice_sax.project_point(intersection);
	double vmin, vmax;
	cv::minMaxLoc(slice_sax.image, &vmin, &vmax);
	cv::Mat1d image = (slice_sax.image - vmin) / (vmax - vmin);

	std::cout
		<< slicecname_sax << "\t"
		<< intersection[1] << "\t" // Write Y coordinate in order to properly sort SAX slices
		<< intersection_sax.x << "\t"
		<< intersection_sax.y << "\t"
		<< intersection_sax.x / slice_sax.image.rows << "\t"
		<< intersection_sax.y / slice_sax.image.cols << std::endl;

#if _DEBUG
	cv::circle(image, intersection_sax, 2, cv::Scalar(255));
	cv::imshow("image", image);
	cv::waitKey(0);
	cv::resize(image, image, image.size() / 3);
	cv::imwrite("1.png", 255*image);
#endif

	return EXIT_SUCCESS;
}
