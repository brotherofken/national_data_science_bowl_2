#include "dicom_reader.hpp"

#include <gdcmReader.h>
#include <gdcmStringFilter.h>
#include <gdcmImageReader.h>
#include <gdcmAttribute.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

namespace fs = ::boost::filesystem;

namespace {
	// return the filenames of all files that have the specified extension
	// in the specified directory and all subdirectories
	std::vector<fs::path> get_all(const fs::path& root, const std::string& ext)
	{
		if (!fs::exists(root) || !fs::is_directory(root))
			return{};

		fs::recursive_directory_iterator it(root);
		fs::recursive_directory_iterator endit;

		std::vector<fs::path> ret;

		const bool get_directories = (ext == "~");
		while (it != endit) {
			auto fn = *it;
			const bool is_reg = get_directories ? fs::is_directory(fn) :fs::is_regular_file(fn);
			const auto fn_ext = get_directories ? true : (fn.path().extension() == ext);
			if (is_reg && fn_ext) ret.push_back(it->path().filename());
			++it;
		}

		return ret;
	}

	std::vector<std::string> string_split(const std::string& str, const std::string& delimeters)
	{
		std::vector<std::string> strs;
		boost::split(strs, str, boost::is_any_of(delimeters));
		return strs;
	}
}

size_t get_frame_number(const std::string& dcm_filename)
{
	const auto strs = string_split(fs::path(dcm_filename).stem().string(), "-");

	// expect one of two patterns for filename
	// IM-6720-0004.dcm
	// IM-2774-0006-0001.dcm
	//          ^    ^ This one is slice index in one position
	//           This is always slice frame number (I hope)
	if (strs.size() == 3 || strs.size() == 4) {
		return std::stoul(strs[2]);
	} else {
		std::cerr << "Error get_frame_number: " << dcm_filename << std::endl;
		throw std::runtime_error("get_frame_number error");
	}
}

// Bad style, no time for refactoring
Slice::Slice(const std::string& filename)
	: filename(filename)
	, frame_number(get_frame_number(filename))
{
	{
		// Read to image
		gdcm::ImageReader ir;
		ir.SetFileName(filename.c_str());
		if (!ir.Read()) {
			std::cerr << "Could not read: " << filename << std::endl;
			throw std::runtime_error("Shit happened");
		}

		const gdcm::Image &gimage = ir.GetImage();

		std::vector<short> vbuffer(gimage.GetBufferLength());
		gimage.GetBuffer((char*)&vbuffer[0]);

		const unsigned int size_x = gimage.GetDimensions()[0];
		const unsigned int size_y = gimage.GetDimensions()[1];
		image = cv::Mat1d(size_y, size_x);
		std::copy(vbuffer.begin(), vbuffer.end(), image.begin());

		// Read non-image fields

		row_dc = cv::Vec3d(gimage.GetDirectionCosines());
		col_dc = cv::Vec3d(gimage.GetDirectionCosines() + 3);
		position = cv::Vec3d(gimage.GetOrigin());
		pixel_spacing = cv::Vec3d(gimage.GetSpacing());
		// Rotation matrix
		rm = cv::Mat1d(0, 3);
		rm.push_back(row_dc * pixel_spacing[0]);
		rm.push_back(col_dc * pixel_spacing[1]);
		rm.push_back(normal() * pixel_spacing[0]);
		rm = rm.reshape(1, 3);
	}
	{
		gdcm::Reader reader;
		reader.SetFileName(filename.c_str());
		if (!reader.Read()) {
			std::cerr << "Could not read: " << filename << std::endl;
			throw std::runtime_error("Crap happened");
		}
		gdcm::File &file = reader.GetFile();

		gdcm::StringFilter sf;
		sf.SetFile(reader.GetFile());
		std::pair<std::string, std::string> slice_location_p = sf.ToStringPair(gdcm::Tag(0x0020, 0x1041));
		std::pair<std::string, std::string> slice_thickness_p = sf.ToStringPair(gdcm::Tag(0x0018, 0x0050));

#if 0 && defined(_DEBUG)
		std::cout << "File meta: " << filename << std::endl;
		std::cout << slice_location_p.first << " " << slice_location_p.second << std::endl;
		std::cout << slice_thickness_p.first << " " << slice_thickness_p.second << std::endl;
#endif


		slice_location = std::stod(slice_location_p.second);
		slice_thickness = std::stod(slice_thickness_p.second);
	}
}

Sequence::Sequence(const std::string& directory)
	: empty(false)
{
	name = fs::path(directory).stem().string();
	std::clog << " sequence " << name << " : ";
	const auto strs = string_split(name, "_");
	assert(strs.size() == 2);
	number = std::stoul(strs[1]);

	if (strs[0] == "2ch") { 
		type = Type::ch2;
	} else if (strs[0] == "4ch") {
		type = Type::ch4;
	} else if (strs[0] == "sax") {
		type = Type::sax;
	} else {
		std::cerr << "Sequence::Sequence : somthing wrong with directory name " << directory << std::endl;
		throw std::runtime_error("Crap happened");
	}

	// for *dcm read
	std::vector<fs::path> dcm_files = get_all(directory, ".dcm");

	for (const auto& file : dcm_files) {
		slices.push_back(Slice((directory / file).string()));
		std::clog << slices.back().frame_number << " ";
	}
	std::clog << std::endl;

	// I'm sure that all DCM in sax contain same information.
	if (slices.size()) {
		row_dc = slices[0].row_dc;
		col_dc = slices[0].col_dc;
		position = slices[0].position;
		slice_location = slices[0].slice_location;
		slice_thickness = slices[0].slice_thickness;
		rm = slices[0].rm.reshape(1, 3).clone();
	}
}

PatientData::PatientData(const std::string& directory)
{
	std::clog << "Reading patient " << directory << std::endl;
	number = std::stoul(fs::path(directory).stem().string());

	// for *dcm read
	fs::path sequences_location = fs::path(directory) / "study";
	std::vector<fs::path> slice_directories = get_all(sequences_location, "~");

	for (const auto& dir : slice_directories) {
		const Sequence s((sequences_location / dir).string());
		if (s.type == Sequence::Type::sax) sax_seqs.push_back(s);
		if (s.type == Sequence::Type::ch2) ch2_seq = s;
		if (s.type == Sequence::Type::ch4) ch4_seq = s;
	}
}
