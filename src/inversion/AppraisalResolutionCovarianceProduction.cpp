//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#include "AppraisalResolutionCovarianceProduction.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "AppraisalRougheningState.h"
#include "AppraisalSensitivityMatrix.h"
#include "DoubleSparseSquareSymmetricMatrix.h"
#include "OutputFiles.h"
#include "ResistivityBlock.h"

namespace {

const int APPRAISAL_MODE_RESOLUTION_AND_COVARIANCE_DIAGONALS = 0;
const int APPRAISAL_MODE_RESOLUTION_DIAGONAL = 1;
const int APPRAISAL_MODE_COVARIANCE_DIAGONAL = 2;

struct MatrixBundle {
	MatrixBundle();

	int numModel;
	int numData;
	int fileCount;
	long long totalBytes;
	std::vector<double> sensitivity;
};

struct DiagonalStats {
	DiagonalStats();

	double numeratorMin;
	double numeratorMax;
	double numeratorMean;
	double denominatorMin;
	double denominatorMax;
	double denominatorMean;
	double diagonalMin;
	double diagonalMax;
	double diagonalMean;
	long long diagonalNonfiniteCount;
	long long zeroDenominatorCount;
};

std::string joinPath(const std::string& directory, const std::string& fileName)
{
	if (directory.empty() || directory == ".") {
		return fileName;
	}
	const char last = directory[directory.size() - 1];
	if (last == '/' || last == '\\') {
		return directory + fileName;
	}
	return directory + "/" + fileName;
}

void ensureDirectoryExists(const std::string& directory)
{
	if (directory.empty() || directory == ".") {
		return;
	}
	if (mkdir(directory.c_str(), 0777) == 0) {
		return;
	}
	if (errno == EEXIST) {
		struct stat status;
		if (stat(directory.c_str(), &status) == 0 && S_ISDIR(status.st_mode)) {
			return;
		}
	}
	std::ostringstream msg;
	msg << "Cannot create appraisal output directory: " << directory;
	throw std::runtime_error(msg.str());
}

double randomSign(const int modelIndex, const int randomVectorIndex)
{
	std::uint32_t value = static_cast<std::uint32_t>(modelIndex + 1);
	value ^= static_cast<std::uint32_t>(randomVectorIndex + 1) * 0x9e3779b9U;
	value ^= value >> 16;
	value *= 0x7feb352dU;
	value ^= value >> 15;
	value *= 0x846ca68bU;
	value ^= value >> 16;
	return (value & 1U) == 0U ? -1.0 : 1.0;
}

std::vector<double> buildRandomVectors(const int numModel, const int numRandomVectors)
{
	std::vector<double> randomVectors(static_cast<std::size_t>(numModel) * numRandomVectors, 0.0);
	for (int iVector = 0; iVector < numRandomVectors; ++iVector) {
		for (int iModel = 0; iModel < numModel; ++iModel) {
			randomVectors[static_cast<std::size_t>(iVector) * numModel + iModel] =
				randomSign(iModel, iVector);
		}
	}
	return randomVectors;
}

MatrixBundle::MatrixBundle():
	numModel(-1),
	numData(0),
	fileCount(0),
	totalBytes(0)
{
}

DiagonalStats::DiagonalStats():
	numeratorMin(std::numeric_limits<double>::max()),
	numeratorMax(-std::numeric_limits<double>::max()),
	numeratorMean(0.0),
	denominatorMin(std::numeric_limits<double>::max()),
	denominatorMax(-std::numeric_limits<double>::max()),
	denominatorMean(0.0),
	diagonalMin(std::numeric_limits<double>::max()),
	diagonalMax(-std::numeric_limits<double>::max()),
	diagonalMean(0.0),
	diagonalNonfiniteCount(0),
	zeroDenominatorCount(0)
{
}

MatrixBundle readSensitivityMatrices(
	const std::vector<std::string>& sensitivityMatrixFiles,
	const int expectedNumModel)
{
	if (sensitivityMatrixFiles.empty()) {
		throw std::runtime_error("No sensMatFreq* files were supplied for production appraisal.");
	}

	MatrixBundle bundle;
	std::vector<AppraisalSensitivityMatrix::MatrixData> matrices;
	for (std::vector<std::string>::const_iterator itr = sensitivityMatrixFiles.begin();
		 itr != sensitivityMatrixFiles.end(); ++itr) {
		const AppraisalSensitivityMatrix::MatrixData matrix =
			AppraisalSensitivityMatrix::readAndValidate(*itr, -1, bundle.numModel, true);
		if (!matrix.nameInfo.hasValidName ||
			matrix.nameInfo.matrixKind != AppraisalSensitivityMatrix::MATRIX_KIND_RAW) {
			throw std::runtime_error("Production appraisal requires raw sensMatFreq<ID> files.");
		}
		if (bundle.numModel < 0) {
			bundle.numModel = matrix.numModel;
		}
		bundle.numData += matrix.numData;
		bundle.totalBytes += matrix.actualSizeBytes;
		matrices.push_back(matrix);
	}

	if (bundle.numModel <= 0 || bundle.numData <= 0) {
		throw std::runtime_error("Invalid combined sensitivity-matrix dimensions for production appraisal.");
	}
	if (expectedNumModel > 0 && bundle.numModel != expectedNumModel) {
		std::ostringstream msg;
		msg << "Production appraisal sensitivity numModel mismatch: expected "
			<< expectedNumModel << ", got " << bundle.numModel << ".";
		throw std::runtime_error(msg.str());
	}

	const long long totalValues = static_cast<long long>(bundle.numData) * bundle.numModel;
	if (static_cast<unsigned long long>(totalValues) >
		static_cast<unsigned long long>(bundle.sensitivity.max_size())) {
		throw std::runtime_error("Combined production appraisal sensitivity matrix is too large to allocate.");
	}

	bundle.sensitivity.reserve(static_cast<std::size_t>(totalValues));
	for (std::vector<AppraisalSensitivityMatrix::MatrixData>::const_iterator itr = matrices.begin();
		 itr != matrices.end(); ++itr) {
		bundle.sensitivity.insert(bundle.sensitivity.end(), itr->values.begin(), itr->values.end());
	}
	bundle.fileCount = static_cast<int>(matrices.size());
	return bundle;
}

std::vector<double> solveSparseRtr(
	DoubleSparseSquareSymmetricMatrix& rtrMatrix,
	const int numRightHandSide,
	const std::vector<double>& rhs)
{
	if (numRightHandSide <= 0) {
		throw std::runtime_error("Production appraisal sparse solve received a non-positive RHS count.");
	}
	std::vector<double> rhsCopy(rhs);
	std::vector<double> solution(rhs.size(), 0.0);
	rtrMatrix.solvePhaseMatrixSolver(
		numRightHandSide,
		&rhsCopy[0],
		&solution[0]);
	return solution;
}

std::vector<double> multiplyJModel(
	const std::vector<double>& sensitivity,
	const int numData,
	const int numModel,
	const std::vector<double>& modelVectors,
	const int numVectors)
{
	std::vector<double> result(static_cast<std::size_t>(numVectors) * numData, 0.0);
	for (int iVector = 0; iVector < numVectors; ++iVector) {
		const std::size_t vectorOffset = static_cast<std::size_t>(iVector) * numModel;
		const std::size_t resultOffset = static_cast<std::size_t>(iVector) * numData;
		for (int iData = 0; iData < numData; ++iData) {
			const std::size_t rowOffset = static_cast<std::size_t>(iData) * numModel;
			double sum = 0.0;
			for (int iModel = 0; iModel < numModel; ++iModel) {
				sum += sensitivity[rowOffset + iModel] * modelVectors[vectorOffset + iModel];
			}
			result[resultOffset + iData] = sum;
		}
	}
	return result;
}

std::vector<double> multiplyJTData(
	const std::vector<double>& sensitivity,
	const int numData,
	const int numModel,
	const std::vector<double>& dataVectors,
	const int numVectors)
{
	std::vector<double> result(static_cast<std::size_t>(numVectors) * numModel, 0.0);
	for (int iVector = 0; iVector < numVectors; ++iVector) {
		const std::size_t dataOffset = static_cast<std::size_t>(iVector) * numData;
		const std::size_t resultOffset = static_cast<std::size_t>(iVector) * numModel;
		for (int iData = 0; iData < numData; ++iData) {
			const std::size_t rowOffset = static_cast<std::size_t>(iData) * numModel;
			const double dataValue = dataVectors[dataOffset + iData];
			for (int iModel = 0; iModel < numModel; ++iModel) {
				result[resultOffset + iModel] += sensitivity[rowOffset + iModel] * dataValue;
			}
		}
	}
	return result;
}

std::vector<double> subtractVectors(
	const std::vector<double>& left,
	const std::vector<double>& right)
{
	if (left.size() != right.size()) {
		throw std::runtime_error("Production appraisal vector subtraction size mismatch.");
	}
	std::vector<double> result(left.size(), 0.0);
	for (std::size_t i = 0; i < left.size(); ++i) {
		result[i] = left[i] - right[i];
	}
	return result;
}

std::vector<double> buildDataMatrix(
	const std::vector<double>& sensitivity,
	const std::vector<double>& modifiedSensitivity,
	const int numData,
	const int numModel)
{
	std::vector<double> matrix(static_cast<std::size_t>(numData) * numData, 0.0);
	for (int iData = 0; iData < numData; ++iData) {
		matrix[static_cast<std::size_t>(iData) * numData + iData] = 1.0;
	}
	for (int iLeft = 0; iLeft < numData; ++iLeft) {
		const std::size_t leftOffset = static_cast<std::size_t>(iLeft) * numModel;
		for (int iRight = 0; iRight < numData; ++iRight) {
			const std::size_t rightOffset = static_cast<std::size_t>(iRight) * numModel;
			double sum = 0.0;
			for (int iModel = 0; iModel < numModel; ++iModel) {
				sum += sensitivity[leftOffset + iModel] * modifiedSensitivity[rightOffset + iModel];
			}
			matrix[static_cast<std::size_t>(iLeft) * numData + iRight] += sum;
		}
	}
	return matrix;
}

std::vector<double> solveDenseLinearSystem(
	const std::vector<double>& matrix,
	const int size,
	const std::vector<double>& rightHandSide,
	const int numRightHandSide)
{
	std::vector<double> work(static_cast<std::size_t>(size) * (size + numRightHandSide), 0.0);
	for (int iRow = 0; iRow < size; ++iRow) {
		for (int iCol = 0; iCol < size; ++iCol) {
			work[static_cast<std::size_t>(iRow) * (size + numRightHandSide) + iCol] =
				matrix[static_cast<std::size_t>(iRow) * size + iCol];
		}
		for (int iRhs = 0; iRhs < numRightHandSide; ++iRhs) {
			work[static_cast<std::size_t>(iRow) * (size + numRightHandSide) + size + iRhs] =
				rightHandSide[static_cast<std::size_t>(iRhs) * size + iRow];
		}
	}

	for (int iPivot = 0; iPivot < size; ++iPivot) {
		int pivotRow = iPivot;
		double pivotAbs = std::fabs(work[static_cast<std::size_t>(iPivot) * (size + numRightHandSide) + iPivot]);
		for (int iRow = iPivot + 1; iRow < size; ++iRow) {
			const double valueAbs = std::fabs(work[static_cast<std::size_t>(iRow) * (size + numRightHandSide) + iPivot]);
			if (valueAbs > pivotAbs) {
				pivotAbs = valueAbs;
				pivotRow = iRow;
			}
		}
		if (pivotAbs < 1.0e-14) {
			throw std::runtime_error("Singular data-space matrix in production appraisal.");
		}
		if (pivotRow != iPivot) {
			for (int iCol = iPivot; iCol < size + numRightHandSide; ++iCol) {
				const std::size_t index0 = static_cast<std::size_t>(iPivot) * (size + numRightHandSide) + iCol;
				const std::size_t index1 = static_cast<std::size_t>(pivotRow) * (size + numRightHandSide) + iCol;
				const double tmp = work[index0];
				work[index0] = work[index1];
				work[index1] = tmp;
			}
		}

		const double pivot = work[static_cast<std::size_t>(iPivot) * (size + numRightHandSide) + iPivot];
		for (int iCol = iPivot; iCol < size + numRightHandSide; ++iCol) {
			work[static_cast<std::size_t>(iPivot) * (size + numRightHandSide) + iCol] /= pivot;
		}
		for (int iRow = 0; iRow < size; ++iRow) {
			if (iRow == iPivot) {
				continue;
			}
			const double factor = work[static_cast<std::size_t>(iRow) * (size + numRightHandSide) + iPivot];
			for (int iCol = iPivot; iCol < size + numRightHandSide; ++iCol) {
				work[static_cast<std::size_t>(iRow) * (size + numRightHandSide) + iCol] -=
					factor * work[static_cast<std::size_t>(iPivot) * (size + numRightHandSide) + iCol];
			}
		}
	}

	std::vector<double> result(static_cast<std::size_t>(numRightHandSide) * size, 0.0);
	for (int iRow = 0; iRow < size; ++iRow) {
		for (int iRhs = 0; iRhs < numRightHandSide; ++iRhs) {
			result[static_cast<std::size_t>(iRhs) * size + iRow] =
				work[static_cast<std::size_t>(iRow) * (size + numRightHandSide) + size + iRhs];
		}
	}
	return result;
}

DiagonalStats calculateStats(
	const std::vector<double>& resultVector,
	const std::vector<double>& randomVectors,
	const int numModel,
	const int numRandomVectors,
	const int checkpoint)
{
	if (checkpoint <= 0 || checkpoint > numRandomVectors) {
		throw std::runtime_error("Invalid checkpoint for production appraisal.");
	}

	DiagonalStats stats;
	for (int iModel = 0; iModel < numModel; ++iModel) {
		double numerator = 0.0;
		double denominator = 0.0;
		for (int iVector = 0; iVector < checkpoint; ++iVector) {
			const std::size_t index = static_cast<std::size_t>(iVector) * numModel + iModel;
			numerator += resultVector[index] * randomVectors[index];
			denominator += randomVectors[index] * randomVectors[index];
		}
		double diagonal = 0.0;
		if (denominator < 1.0e-12) {
			++stats.zeroDenominatorCount;
		} else {
			diagonal = std::sqrt(std::fabs(numerator / denominator));
		}
		if (!std::isfinite(diagonal)) {
			++stats.diagonalNonfiniteCount;
			continue;
		}

		stats.numeratorMin = std::min(stats.numeratorMin, numerator);
		stats.numeratorMax = std::max(stats.numeratorMax, numerator);
		stats.numeratorMean += numerator;
		stats.denominatorMin = std::min(stats.denominatorMin, denominator);
		stats.denominatorMax = std::max(stats.denominatorMax, denominator);
		stats.denominatorMean += denominator;
		stats.diagonalMin = std::min(stats.diagonalMin, diagonal);
		stats.diagonalMax = std::max(stats.diagonalMax, diagonal);
		stats.diagonalMean += diagonal;
	}

	const double denominator = static_cast<double>(numModel);
	stats.numeratorMean /= denominator;
	stats.denominatorMean /= denominator;
	stats.diagonalMean /= denominator;
	if (stats.diagonalNonfiniteCount == numModel) {
		stats.numeratorMin = 0.0;
		stats.numeratorMax = 0.0;
		stats.denominatorMin = 0.0;
		stats.denominatorMax = 0.0;
		stats.diagonalMin = 0.0;
		stats.diagonalMax = 0.0;
	}
	return stats;
}

std::vector<double> calculateDiagonalValues(
	const std::vector<double>& resultVector,
	const std::vector<double>& randomVectors,
	const int numModel,
	const int numRandomVectors,
	const int checkpoint)
{
	if (checkpoint <= 0 || checkpoint > numRandomVectors) {
		throw std::runtime_error("Invalid checkpoint for production appraisal block output.");
	}

	std::vector<double> diagonalValues(static_cast<std::size_t>(numModel), 0.0);
	for (int iModel = 0; iModel < numModel; ++iModel) {
		double numerator = 0.0;
		double denominator = 0.0;
		for (int iVector = 0; iVector < checkpoint; ++iVector) {
			const std::size_t index = static_cast<std::size_t>(iVector) * numModel + iModel;
			numerator += resultVector[index] * randomVectors[index];
			denominator += randomVectors[index] * randomVectors[index];
		}
		if (denominator >= 1.0e-12) {
			diagonalValues[static_cast<std::size_t>(iModel)] =
				std::sqrt(std::fabs(numerator / denominator));
		}
	}
	return diagonalValues;
}

std::string appraisalDiagonalBlockPath(
	const std::string& outputDirectory,
	const char* const fileName)
{
	return joinPath(outputDirectory, fileName);
}

std::string writeDiagonalBlockValues(
	const AppraisalResolutionCovarianceProduction::RunConfig& config,
	const char* const family,
	const ResistivityBlock::AppraisalOutputFamily familyID,
	const int checkpoint,
	const std::vector<double>& resultVector,
	const std::vector<double>& randomVectors,
	const int numModel)
{
	const std::vector<double> modelDiagonalValues = calculateDiagonalValues(
		resultVector,
		randomVectors,
		numModel,
		config.numRandomVectors,
		checkpoint);
	const ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();
	const int numBlocks = ptrResistivityBlock->getNumResistivityBlockTotal();
	std::vector<double> blockValues(static_cast<std::size_t>(numBlocks), 0.0);
	ptrResistivityBlock->mapNotFixedModelValuesToResistivityBlockValues(
		&modelDiagonalValues[0],
		numModel,
		0.0,
		&blockValues[0],
		numBlocks);
	ptrResistivityBlock->outputAppraisalResistivityBlock(
		config.outputDirectory,
		familyID,
		ResistivityBlock::APPRAISAL_QUANTITY_DIAGONAL_VALUE,
		checkpoint,
		&blockValues[0],
		numBlocks);

	std::ostringstream fileName;
	fileName << "appraisal_" << family << "_diagonal_value_checkpoint" << checkpoint << ".dat";
	return appraisalDiagonalBlockPath(config.outputDirectory, fileName.str().c_str());
}

std::vector<double> calculateResolutionResult(
	DoubleSparseSquareSymmetricMatrix& rtrMatrix,
	const std::vector<double>& sensitivity,
	const std::vector<double>& dataMatrix,
	const std::vector<double>& randomVectors,
	const int numData,
	const int numModel,
	const int numRandomVectors)
{
	const std::vector<double> jTimesRandomVectors =
		multiplyJModel(sensitivity, numData, numModel, randomVectors, numRandomVectors);
	const std::vector<double> resolutionInitial =
		multiplyJTData(sensitivity, numData, numModel, jTimesRandomVectors, numRandomVectors);
	const std::vector<double> resolutionPreconditioned =
		solveSparseRtr(rtrMatrix, numRandomVectors, resolutionInitial);
	const std::vector<double> resolutionDataRhs =
		multiplyJModel(sensitivity, numData, numModel, resolutionPreconditioned, numRandomVectors);
	const std::vector<double> resolutionDataSolution =
		solveDenseLinearSystem(dataMatrix, numData, resolutionDataRhs, numRandomVectors);
	const std::vector<double> resolutionBackProjected =
		multiplyJTData(sensitivity, numData, numModel, resolutionDataSolution, numRandomVectors);
	return solveSparseRtr(
		rtrMatrix,
		numRandomVectors,
		subtractVectors(resolutionInitial, resolutionBackProjected));
}

std::vector<double> calculateCovarianceResult(
	DoubleSparseSquareSymmetricMatrix& rtrMatrix,
	const std::vector<double>& sensitivity,
	const std::vector<double>& dataMatrix,
	const std::vector<double>& randomVectors,
	const int numData,
	const int numModel,
	const int numRandomVectors)
{
	const std::vector<double> covarianceHelperC0 =
		solveSparseRtr(rtrMatrix, numRandomVectors, randomVectors);
	const std::vector<double> covarianceHelperDataRhs =
		multiplyJModel(sensitivity, numData, numModel, covarianceHelperC0, numRandomVectors);
	const std::vector<double> covarianceHelperDataSolution =
		solveDenseLinearSystem(dataMatrix, numData, covarianceHelperDataRhs, numRandomVectors);
	const std::vector<double> covarianceHelperBackProjected =
		multiplyJTData(sensitivity, numData, numModel, covarianceHelperDataSolution, numRandomVectors);
	const std::vector<double> covarianceHelper =
		solveSparseRtr(
			rtrMatrix,
			numRandomVectors,
			subtractVectors(randomVectors, covarianceHelperBackProjected));

	const std::vector<double> jTimesCovarianceHelper =
		multiplyJModel(sensitivity, numData, numModel, covarianceHelper, numRandomVectors);
	const std::vector<double> covarianceInitial =
		multiplyJTData(sensitivity, numData, numModel, jTimesCovarianceHelper, numRandomVectors);
	const std::vector<double> covariancePreconditioned =
		solveSparseRtr(rtrMatrix, numRandomVectors, covarianceInitial);
	const std::vector<double> covarianceDataRhs =
		multiplyJModel(sensitivity, numData, numModel, covariancePreconditioned, numRandomVectors);
	const std::vector<double> covarianceDataSolution =
		solveDenseLinearSystem(dataMatrix, numData, covarianceDataRhs, numRandomVectors);
	const std::vector<double> covarianceBackProjected =
		multiplyJTData(sensitivity, numData, numModel, covarianceDataSolution, numRandomVectors);
	return solveSparseRtr(
		rtrMatrix,
		numRandomVectors,
		subtractVectors(covarianceInitial, covarianceBackProjected));
}

void appendSummaryRows(
	const std::string& path,
	const AppraisalResolutionCovarianceProduction::RunConfig& config,
	const MatrixBundle& bundle,
	const AppraisalRougheningState::SparseMatrixStats& rtrStats,
	const std::vector<double>& randomVectors,
	const std::vector<double>& resultVector,
	const char* const family,
	const ResistivityBlock::AppraisalOutputFamily familyID,
	const double runtimeSeconds)
{
	const bool writeHeader = [] (const std::string& fileName) {
		std::ifstream input(fileName.c_str());
		return !input.good() || input.peek() == std::ifstream::traits_type::eof();
	}(path);

	std::ofstream output(path.c_str(), std::ios::app);
	if (!output) {
		throw std::runtime_error("Cannot open production appraisal summary CSV: " + path);
	}
	output << std::setprecision(17);
	if (writeHeader) {
		output
			<< "program,version,iteration,appraisal_mode,appraisal_family,"
			<< "checkpoint_random_vectors,total_random_vectors,num_model,num_data,file_count,total_input_bytes,"
			<< "roughness_operator,rtr_rows,rtr_columns,rtr_nonzeros,rtr_diagonal_nonzeros,rtr_nonfinite_values,"
			<< "numerator_min,numerator_max,numerator_mean,denominator_min,denominator_max,denominator_mean,"
			<< "diagonal_min,diagonal_max,diagonal_mean,diagonal_nonfinite_count,zero_denominator_count,"
			<< "block_diagonal_value_path,input_sensitivity_dir,output_dir,write_legacy_dsdk,runtime_seconds,status,notes"
			<< std::endl;
	}

	for (std::vector<int>::const_iterator itr = config.checkpoints.begin();
		 itr != config.checkpoints.end(); ++itr) {
		const DiagonalStats stats = calculateStats(
			resultVector,
			randomVectors,
			bundle.numModel,
			config.numRandomVectors,
			*itr);
		const std::string blockDiagonalValuePath = writeDiagonalBlockValues(
			config,
			family,
			familyID,
			*itr,
			resultVector,
			randomVectors,
			bundle.numModel);
		output
			<< config.programName << ","
			<< config.programVersion << ","
			<< config.iteration << ","
			<< config.appraisalMode << ","
			<< family << ","
			<< *itr << ","
			<< config.numRandomVectors << ","
			<< bundle.numModel << ","
			<< bundle.numData << ","
			<< bundle.fileCount << ","
			<< bundle.totalBytes << ","
			<< config.roughnessOperator << ","
			<< rtrStats.numRows << ","
			<< rtrStats.numColumns << ","
			<< rtrStats.numNonZeros << ","
			<< rtrStats.numDiagonalNonZeros << ","
			<< rtrStats.numNonFiniteValues << ","
			<< stats.numeratorMin << ","
			<< stats.numeratorMax << ","
			<< stats.numeratorMean << ","
			<< stats.denominatorMin << ","
			<< stats.denominatorMax << ","
			<< stats.denominatorMean << ","
			<< stats.diagonalMin << ","
			<< stats.diagonalMax << ","
			<< stats.diagonalMean << ","
			<< stats.diagonalNonfiniteCount << ","
			<< stats.zeroDenominatorCount << ","
			<< blockDiagonalValuePath << ","
			<< config.inputSensitivityDirectory << ","
			<< config.outputDirectory << ","
			<< (config.writeLegacyDsdkFiles ? "yes" : "no") << ","
			<< runtimeSeconds << ","
			<< "ok,"
			<< "production_sparse_rtr_with_diagonal_block_fields"
			<< std::endl;
	}
}

}

namespace AppraisalResolutionCovarianceProduction {

RunConfig::RunConfig():
	iteration(0),
	appraisalMode(-1),
	numRandomVectors(0),
	writeLegacyDsdkFiles(false),
	expectedNumModel(0),
	pardisoMode(0)
{
}

RunResult::RunResult():
	numModel(0),
	numData(0),
	fileCount(0),
	totalInputBytes(0),
	runtimeSeconds(0.0)
{
}

RunResult runProductionAppraisalSummary(
	const RunConfig& config,
	DoubleSparseSquareSymmetricMatrix& rtrMatrix,
	const std::vector<std::string>& sensitivityMatrixFiles)
{
	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	if (config.writeLegacyDsdkFiles) {
		throw std::runtime_error(
			"Legacy DSDK appraisal files are not supported by the production summary path.");
	}
	ensureDirectoryExists(config.outputDirectory);

	MatrixBundle bundle = readSensitivityMatrices(
		sensitivityMatrixFiles,
		config.expectedNumModel);
	if (rtrMatrix.getNumRows() != bundle.numModel || rtrMatrix.getNumColumns() != bundle.numModel) {
		std::ostringstream msg;
		msg << "Production appraisal R^T R dimension mismatch: R^T R is "
			<< rtrMatrix.getNumRows() << " x " << rtrMatrix.getNumColumns()
			<< ", but sensitivity numModel is " << bundle.numModel << ".";
		throw std::runtime_error(msg.str());
	}

	const AppraisalRougheningState::SparseMatrixStats rtrStats =
		AppraisalRougheningState::inspectSparseMatrix(rtrMatrix);
	if (rtrStats.numNonFiniteValues > 0) {
		throw std::runtime_error("Production appraisal R^T R contains non-finite values.");
	}

	const std::string oocHeaderName = "ooc_temp_appraisal_resolution_covariance";
	OutputFiles::m_logFile << "# Appraisal production summary: initialize sparse R^T R solver." << std::endl;
	rtrMatrix.initializeMatrixSolver(oocHeaderName, config.pardisoMode);
	rtrMatrix.analysisPhaseMatrixSolver();
	rtrMatrix.factorizationPhaseMatrixSolver();

	std::vector<double> modifiedSensitivity =
		solveSparseRtr(rtrMatrix, bundle.numData, bundle.sensitivity);
	const std::vector<double> dataMatrix = buildDataMatrix(
		bundle.sensitivity,
		modifiedSensitivity,
		bundle.numData,
		bundle.numModel);
	modifiedSensitivity.clear();

	const std::vector<double> randomVectors =
		buildRandomVectors(bundle.numModel, config.numRandomVectors);
	const std::string summaryPath = joinPath(
		config.outputDirectory,
		"appraisal_production_summary.csv");

	if (config.appraisalMode == APPRAISAL_MODE_RESOLUTION_AND_COVARIANCE_DIAGONALS ||
		config.appraisalMode == APPRAISAL_MODE_RESOLUTION_DIAGONAL) {
		const std::vector<double> resolutionResult =
			calculateResolutionResult(
				rtrMatrix,
				bundle.sensitivity,
				dataMatrix,
				randomVectors,
				bundle.numData,
				bundle.numModel,
				config.numRandomVectors);
		const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		appendSummaryRows(
			summaryPath,
			config,
			bundle,
			rtrStats,
			randomVectors,
			resolutionResult,
			"model_resolution_diagonal",
			ResistivityBlock::APPRAISAL_OUTPUT_MODEL_RESOLUTION_DIAGONAL,
			std::chrono::duration<double>(now - start).count());
	}
	if (config.appraisalMode == APPRAISAL_MODE_RESOLUTION_AND_COVARIANCE_DIAGONALS ||
		config.appraisalMode == APPRAISAL_MODE_COVARIANCE_DIAGONAL) {
		const std::vector<double> covarianceResult =
			calculateCovarianceResult(
				rtrMatrix,
				bundle.sensitivity,
				dataMatrix,
				randomVectors,
				bundle.numData,
				bundle.numModel,
				config.numRandomVectors);
		const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		appendSummaryRows(
			summaryPath,
			config,
			bundle,
			rtrStats,
			randomVectors,
			covarianceResult,
			"covariance_diagonal",
			ResistivityBlock::APPRAISAL_OUTPUT_COVARIANCE_DIAGONAL,
			std::chrono::duration<double>(now - start).count());
	}
	if (config.appraisalMode != APPRAISAL_MODE_RESOLUTION_AND_COVARIANCE_DIAGONALS &&
		config.appraisalMode != APPRAISAL_MODE_RESOLUTION_DIAGONAL &&
		config.appraisalMode != APPRAISAL_MODE_COVARIANCE_DIAGONAL) {
		throw std::runtime_error("Production appraisal summary supports APPRAISAL_MODE=0, 1, or 2 only.");
	}

	rtrMatrix.releaseMemory();

	const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	RunResult result;
	result.numModel = bundle.numModel;
	result.numData = bundle.numData;
	result.fileCount = bundle.fileCount;
	result.totalInputBytes = bundle.totalBytes;
	result.runtimeSeconds = std::chrono::duration<double>(end - start).count();
	result.summaryPath = summaryPath;
	return result;
}

} // namespace AppraisalResolutionCovarianceProduction
