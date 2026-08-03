//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#include "AppraisalSensitivityMatrix.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
	const long long HEADER_SIZE_BYTES = 2LL * static_cast<long long>(sizeof(int));
	const long long VALUE_SIZE_BYTES = static_cast<long long>(sizeof(double));

	std::string baseName(const std::string& fileName) {
		const std::string::size_type pos = fileName.find_last_of("/\\");
		if (pos == std::string::npos) {
			return fileName;
		}
		return fileName.substr(pos + 1);
	}

	std::string buildPrefixMessage(const std::string& fileName) {
		std::ostringstream msg;
		msg << "Invalid appraisal sensitivity matrix file '" << fileName << "': ";
		return msg.str();
	}

	void requireMaintainedBinaryTypes(const std::string& fileName) {
		if (sizeof(int) != 4 || sizeof(double) != 8) {
			std::ostringstream msg;
			msg << buildPrefixMessage(fileName)
				<< "unsupported native binary layout. Expected sizeof(int)=4 and sizeof(double)=8, but got "
				<< sizeof(int) << " and " << sizeof(double) << ".";
			throw std::runtime_error(msg.str());
		}
	}
}

AppraisalSensitivityMatrix::MatrixNameInfo::MatrixNameInfo()
	: frequencyID(-1),
	  matrixKind(MATRIX_KIND_UNKNOWN),
	  hasValidName(false)
{
}

AppraisalSensitivityMatrix::MatrixData::MatrixData()
	: numData(0),
	  numModel(0),
	  valueCount(0),
	  expectedSizeBytes(0),
	  actualSizeBytes(0)
{
}

AppraisalSensitivityMatrix::MatrixNameInfo
AppraisalSensitivityMatrix::parseSensitivityMatrixFileName(const std::string& fileName)
{
	MatrixNameInfo info;
	const std::string name = baseName(fileName);
	const std::string prefix = "sensMatFreq";

	if (name.compare(0, prefix.size(), prefix) != 0) {
		return info;
	}

	std::string::size_type pos = prefix.size();
	if (pos >= name.size() || !std::isdigit(static_cast<unsigned char>(name[pos]))) {
		return info;
	}

	long long frequencyID = 0;
	while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
		frequencyID = frequencyID * 10 + static_cast<long long>(name[pos] - '0');
		if (frequencyID > static_cast<long long>(std::numeric_limits<int>::max())) {
			return info;
		}
		++pos;
	}

	if (pos == name.size()) {
		info.matrixKind = MATRIX_KIND_RAW;
	} else if (name.substr(pos) == "Mod") {
		info.matrixKind = MATRIX_KIND_MODIFIED;
	} else {
		return info;
	}

	info.frequencyID = static_cast<int>(frequencyID);
	info.hasValidName = true;
	return info;
}

long long
AppraisalSensitivityMatrix::calculateExpectedFileSizeBytes(const int numData, const int numModel)
{
	if (numData <= 0 || numModel <= 0) {
		throw std::runtime_error("Invalid appraisal sensitivity matrix dimensions: numData and numModel must be positive.");
	}

	const long long rows = static_cast<long long>(numData);
	const long long columns = static_cast<long long>(numModel);
	if (rows > std::numeric_limits<long long>::max() / columns) {
		throw std::runtime_error("Invalid appraisal sensitivity matrix dimensions: value-count overflow.");
	}

	const long long valueCount = rows * columns;
	if (valueCount > (std::numeric_limits<long long>::max() - HEADER_SIZE_BYTES) / VALUE_SIZE_BYTES) {
		throw std::runtime_error("Invalid appraisal sensitivity matrix dimensions: file-size overflow.");
	}

	return HEADER_SIZE_BYTES + valueCount * VALUE_SIZE_BYTES;
}

AppraisalSensitivityMatrix::MatrixData
AppraisalSensitivityMatrix::readAndValidate(
	const std::string& fileName,
	const int expectedNumData,
	const int expectedNumModel,
	const bool requireKnownName)
{
	requireMaintainedBinaryTypes(fileName);

	MatrixData matrix;
	matrix.nameInfo = parseSensitivityMatrixFileName(fileName);
	if (requireKnownName && !matrix.nameInfo.hasValidName) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName)
			<< "expected name pattern sensMatFreq<ID> or sensMatFreq<ID>Mod.";
		throw std::runtime_error(msg.str());
	}

	std::ifstream input(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	if (!input) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName) << "cannot open the file.";
		throw std::runtime_error(msg.str());
	}

	const std::streamoff endPosition = static_cast<std::streamoff>(input.tellg());
	if (endPosition < 0) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName) << "cannot determine file size.";
		throw std::runtime_error(msg.str());
	}
	matrix.actualSizeBytes = static_cast<long long>(endPosition);
	input.seekg(0, std::ios::beg);

	input.read(reinterpret_cast<char*>(&matrix.numData), sizeof(int));
	input.read(reinterpret_cast<char*>(&matrix.numModel), sizeof(int));
	if (!input) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName) << "cannot read the binary header.";
		throw std::runtime_error(msg.str());
	}

	if (matrix.numData <= 0 || matrix.numModel <= 0) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName)
			<< "non-positive dimensions in header: numData=" << matrix.numData
			<< ", numModel=" << matrix.numModel << ".";
		throw std::runtime_error(msg.str());
	}

	if (expectedNumData >= 0 && matrix.numData != expectedNumData) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName)
			<< "numData mismatch. Expected " << expectedNumData
			<< ", got " << matrix.numData << ".";
		throw std::runtime_error(msg.str());
	}
	if (expectedNumModel >= 0 && matrix.numModel != expectedNumModel) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName)
			<< "numModel mismatch. Expected " << expectedNumModel
			<< ", got " << matrix.numModel << ".";
		throw std::runtime_error(msg.str());
	}

	matrix.expectedSizeBytes = calculateExpectedFileSizeBytes(matrix.numData, matrix.numModel);
	if (matrix.actualSizeBytes != matrix.expectedSizeBytes) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName)
			<< "file-size mismatch. Expected " << matrix.expectedSizeBytes
			<< " bytes, got " << matrix.actualSizeBytes << " bytes.";
		throw std::runtime_error(msg.str());
	}

	matrix.valueCount = static_cast<long long>(matrix.numData) * static_cast<long long>(matrix.numModel);
	if (static_cast<unsigned long long>(matrix.valueCount) > static_cast<unsigned long long>(matrix.values.max_size())) {
		std::ostringstream msg;
		msg << buildPrefixMessage(fileName)
			<< "matrix is too large to allocate in memory: valueCount=" << matrix.valueCount << ".";
		throw std::runtime_error(msg.str());
	}

	matrix.values.resize(static_cast<std::size_t>(matrix.valueCount));
	if (!matrix.values.empty()) {
		const long long valueSizeBytes = matrix.valueCount * VALUE_SIZE_BYTES;
		if (valueSizeBytes > static_cast<long long>(std::numeric_limits<std::streamsize>::max())) {
			std::ostringstream msg;
			msg << buildPrefixMessage(fileName)
				<< "matrix is too large for one binary read: valueBytes=" << valueSizeBytes << ".";
			throw std::runtime_error(msg.str());
		}
		input.read(reinterpret_cast<char*>(&matrix.values[0]), static_cast<std::streamsize>(valueSizeBytes));
		if (!input) {
			std::ostringstream msg;
			msg << buildPrefixMessage(fileName) << "cannot read all matrix values.";
			throw std::runtime_error(msg.str());
		}
	}

	for (std::size_t i = 0; i < matrix.values.size(); ++i) {
		if (!std::isfinite(matrix.values[i])) {
			std::ostringstream msg;
			msg << buildPrefixMessage(fileName)
				<< "non-finite matrix value at flattened index " << i << ".";
			throw std::runtime_error(msg.str());
		}
	}

	return matrix;
}
