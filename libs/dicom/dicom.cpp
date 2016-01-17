#include "dicom.hpp"

namespace VVV {

	Dicom::Element & Dicom::Element::_read_element_data_string(std::istream &ist, const size_t len)
	{
		char *buf = new char[len];
		ist.read(buf, len);

		if (ist.eof() || !ist.good())
			throw StreamError("");

		this->_value = std::string(buf, len);
		this->_is_vector = false;

		delete[] buf;
		return *this;
	}

	Dicom::Element & Dicom::Element::_read_element_data_sequence(std::istream &ist, size_t len)
	{
		//
		// when size was known
		//
		if (len != 0xFFFFFFFF)
			return this->_read_element_data<unsigned char>(ist, len);

		//
		// when unknown size gaven
		//
		unsigned char eos[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

		std::vector<unsigned char> value;
		unsigned char c;
		while (true){
			ist.read((char *)&c, 1);
			if (ist.eof() || !ist.good())
				throw StreamError("");

			value.push_back(c);

			//for(int i=0;i<7;i++)
			//    eos[i]=eos[i+1];
			memmove(eos, eos + 1, 7);
			eos[7] = c;

			// check end of sequence 0xFF FE E0 DD 00 00 00 00
			if (eos[0] == 0xFF &&
				eos[1] == 0xFE &&
				eos[2] == 0xE0 &&
				eos[3] == 0xDD &&
				eos[4] == 0x00 &&
				eos[5] == 0x00 &&
				eos[6] == 0x00 &&
				eos[7] == 0x00)
				break;
		}

		// erase end of sequence
		if (value.size() >= 8){
			for (int i = 0; i < 8; i++)
				value.pop_back();
		}

		// erase start of sequence 0xFE FF E0 00
		if (value.size() >= 4 &&
			value[0] == 0xFE &&
			value[1] == 0xFF &&
			value[2] == 0xE0 &&
			value[3] == 0x00){
			std::vector<unsigned char>::iterator itr = value.begin();
			value.erase(itr, itr + 4);
		}

		this->_value = value;
		this->_is_vector = true;

		return *this;
	}


	const uint16_t VVV::Dicom::TAG_GROUP_META = 0x0002;
	const uint16_t VVV::Dicom::TAG_GROUP_DIRECTORY = 0x0004;
	
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_TRANSFER_SYNTAX_UID = VVV::Dicom::TypeTag(0x0002, 0x0010);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_IMG_POSITION = VVV::Dicom::TypeTag(0x0020, 0x0032);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_PHOTO_INTERPRET = VVV::Dicom::TypeTag(0x0028, 0x0004);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_ROWS = VVV::Dicom::TypeTag(0x0028, 0x0010);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_COLS = VVV::Dicom::TypeTag(0x0028, 0x0011);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_PX_SPACING = VVV::Dicom::TypeTag(0x0028, 0x0030);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_BIT_ALLOC = VVV::Dicom::TypeTag(0x0028, 0x0100);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_BIT_STORED = VVV::Dicom::TypeTag(0x0028, 0x0101);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_HI_BIT = VVV::Dicom::TypeTag(0x0028, 0x0102);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_PX_REP = VVV::Dicom::TypeTag(0x0028, 0x0103);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_RESCALE_INT = VVV::Dicom::TypeTag(0x0028, 0x1052);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_RESCALE_SLP = VVV::Dicom::TypeTag(0x0028, 0x1053);
	const VVV::Dicom::TypeTag VVV::Dicom::TAG_FRAME_DATA = VVV::Dicom::TypeTag(0x7fe0, 0x0010);


}