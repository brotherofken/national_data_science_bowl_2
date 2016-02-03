#include "dicom_reader.hpp"

#include <gdcmReader.h>
#include <gdcmStringFilter.h>
#include <gdcmImageReader.h>
#include <gdcmAttribute.h>

Slice read_dcm(const std::string& filename)
{
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

	const unsigned int size_x = gimage.GetDimensions()[0];
	const unsigned int size_y = gimage.GetDimensions()[1];
	cv::Mat1d image(size_y, size_x);
	std::copy(vbuffer.begin(), vbuffer.end(), image.begin());

	// Read non-image fields

	gdcm::Reader reader;
	reader.SetFileName(filename.c_str());
	if (!reader.Read()) {
		std::cerr << "Could not read: " << filename << std::endl;
		return Slice();
	}
	gdcm::File &file = reader.GetFile();

	gdcm::StringFilter sf;
	sf.SetFile(reader.GetFile());
	std::pair<std::string, std::string> slice_location = sf.ToStringPair(gdcm::Tag(0x0020, 0x1041));
	std::pair<std::string, std::string> slice_thickness = sf.ToStringPair(gdcm::Tag(0x0018, 0x0050));

#ifdef _DEBUG
	std::cout << "File meta: " << filename << std::endl;
	std::cout << slice_location.first << " " << slice_location.second << std::endl;
	std::cout << slice_thickness.first << " " << slice_thickness.second << std::endl;
#endif

	return Slice {
		image,
		cv::Vec3d(gimage.GetDirectionCosines()),
		cv::Vec3d(gimage.GetDirectionCosines() + 3),
		cv::Vec3d(gimage.GetOrigin()),
		cv::Vec3d(gimage.GetSpacing()),
		std::stod(slice_location.second),
		std::stod(slice_thickness.second),
	};
}


