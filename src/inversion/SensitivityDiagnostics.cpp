//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#include "SensitivityDiagnostics.h"

#include "AppraisalSensitivityMatrix.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

SensitivityDiagnostics::SensitivityEnergyDiagonalResult::SensitivityEnergyDiagonalResult()
	: numModel(0),
	  fileCount(0),
	  totalDataRows(0)
{
}

SensitivityDiagnostics::SensitivityEnergyDiagonalResult
SensitivityDiagnostics::calculateSensitivityEnergyDiagonalFromSensitivityMatrices(
	const std::vector<std::string>& sensitivityMatrixFiles,
	const int expectedNumModel)
{
	if (sensitivityMatrixFiles.empty()) {
		throw std::runtime_error("No sensitivity matrix files were supplied for sensitivity-energy diagonal diagnostics.");
	}
	if (expectedNumModel == 0 || expectedNumModel < -1) {
		std::ostringstream msg;
		msg << "Invalid expectedNumModel for sensitivity-energy diagonal diagnostics: " << expectedNumModel << ".";
		throw std::runtime_error(msg.str());
	}

	SensitivityEnergyDiagonalResult result;
	int numModel = expectedNumModel;

	for (std::vector<std::string>::const_iterator itr = sensitivityMatrixFiles.begin();
		 itr != sensitivityMatrixFiles.end(); ++itr) {
		const AppraisalSensitivityMatrix::MatrixData matrix =
			AppraisalSensitivityMatrix::readAndValidate(*itr, -1, numModel, true);

		if (numModel < 0) {
			numModel = matrix.numModel;
			result.numModel = numModel;
			result.diagonal.assign(static_cast<std::size_t>(numModel), 0.0);
		} else if (result.diagonal.empty()) {
			result.numModel = numModel;
			result.diagonal.assign(static_cast<std::size_t>(numModel), 0.0);
		}

		if (matrix.numModel != result.numModel) {
			std::ostringstream msg;
			msg << "Inconsistent sensitivity matrix model dimension in " << *itr
				<< ": expected " << result.numModel << ", got " << matrix.numModel << ".";
			throw std::runtime_error(msg.str());
		}

		for (int iData = 0; iData < matrix.numData; ++iData) {
			const long long rowOffset = static_cast<long long>(iData) * static_cast<long long>(matrix.numModel);
			for (int iModel = 0; iModel < matrix.numModel; ++iModel) {
				const double value = matrix.values[static_cast<std::size_t>(rowOffset + iModel)];
				result.diagonal[static_cast<std::size_t>(iModel)] += value * value;
			}
		}

		++result.fileCount;
		result.totalDataRows += matrix.numData;
	}

	if (result.diagonal.empty()) {
		throw std::runtime_error("Sensitivity-energy diagonal diagnostics produced no model values.");
	}

	return result;
}

void
SensitivityDiagnostics::writeSensitivityEnergyDiagonalCsv(
	const std::string& fileName,
	const SensitivityEnergyDiagonalResult& result)
{
	if (result.numModel <= 0 || result.diagonal.size() != static_cast<std::size_t>(result.numModel)) {
		throw std::runtime_error("Invalid sensitivity-energy diagonal result supplied for CSV output.");
	}

	std::ofstream output(fileName.c_str());
	if (!output) {
		std::ostringstream msg;
		msg << "Cannot open sensitivity-energy diagonal CSV file: " << fileName;
		throw std::runtime_error(msg.str());
	}

	output << "model_index,sensitivity_energy_diagonal,sensitivity_l2\n";
	output << std::setprecision(17);
	for (int iModel = 0; iModel < result.numModel; ++iModel) {
		const double value = result.diagonal[static_cast<std::size_t>(iModel)];
		if (!std::isfinite(value)) {
			std::ostringstream msg;
			msg << "Non-finite sensitivity-energy diagonal value at model_index=" << iModel << ".";
			throw std::runtime_error(msg.str());
		}
		output << iModel << "," << value << "," << std::sqrt(value) << "\n";
	}
}
