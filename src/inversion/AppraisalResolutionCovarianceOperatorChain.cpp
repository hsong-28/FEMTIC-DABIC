//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#include "AppraisalResolutionCovarianceOperatorChain.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

void validateSize(const std::vector<double>& values, const long long expected, const char* const label) {
	if (expected < 0 || values.size() != static_cast<std::size_t>(expected)) {
		std::ostringstream msg;
		msg << "Invalid dense resolution/covariance operator-chain " << label << " size: expected "
			<< expected << ", got " << values.size() << ".";
		throw std::runtime_error(msg.str());
	}
	for (std::vector<double>::const_iterator itr = values.begin(); itr != values.end(); ++itr) {
		if (!std::isfinite(*itr)) {
			std::ostringstream msg;
			msg << "Non-finite value in dense resolution/covariance operator-chain " << label << ".";
			throw std::runtime_error(msg.str());
		}
	}
}

std::vector<double> transpose(
	const std::vector<double>& matrix,
	const int rows,
	const int columns)
{
	std::vector<double> result(static_cast<std::size_t>(rows * columns), 0.0);
	for (int iRow = 0; iRow < rows; ++iRow) {
		for (int iCol = 0; iCol < columns; ++iCol) {
			result[static_cast<std::size_t>(iCol * rows + iRow)] =
				matrix[static_cast<std::size_t>(iRow * columns + iCol)];
		}
	}
	return result;
}

std::vector<double> multiply(
	const std::vector<double>& left,
	const int leftRows,
	const int shared,
	const std::vector<double>& right,
	const int rightColumns)
{
	std::vector<double> result(static_cast<std::size_t>(leftRows * rightColumns), 0.0);
	for (int iRow = 0; iRow < leftRows; ++iRow) {
		for (int iCol = 0; iCol < rightColumns; ++iCol) {
			double sum = 0.0;
			for (int iShared = 0; iShared < shared; ++iShared) {
				sum += left[static_cast<std::size_t>(iRow * shared + iShared)] *
					right[static_cast<std::size_t>(iShared * rightColumns + iCol)];
			}
			result[static_cast<std::size_t>(iRow * rightColumns + iCol)] = sum;
		}
	}
	return result;
}

std::vector<double> identity(const int size) {
	std::vector<double> result(static_cast<std::size_t>(size * size), 0.0);
	for (int i = 0; i < size; ++i) {
		result[static_cast<std::size_t>(i * size + i)] = 1.0;
	}
	return result;
}

std::vector<double> subtract(
	const std::vector<double>& left,
	const std::vector<double>& right)
{
	if (left.size() != right.size()) {
		throw std::runtime_error("Mismatched dense resolution/covariance subtraction sizes.");
	}
	std::vector<double> result(left.size(), 0.0);
	for (std::size_t i = 0; i < left.size(); ++i) {
		result[i] = left[i] - right[i];
	}
	return result;
}

std::vector<double> solveLinearSystem(
	const std::vector<double>& matrix,
	const int size,
	const std::vector<double>& rightHandSide,
	const int numRightHandSide)
{
	std::vector<double> work(static_cast<std::size_t>(size * (size + numRightHandSide)), 0.0);
	for (int iRow = 0; iRow < size; ++iRow) {
		for (int iCol = 0; iCol < size; ++iCol) {
			work[static_cast<std::size_t>(iRow * (size + numRightHandSide) + iCol)] =
				matrix[static_cast<std::size_t>(iRow * size + iCol)];
		}
		for (int iRhs = 0; iRhs < numRightHandSide; ++iRhs) {
			work[static_cast<std::size_t>(iRow * (size + numRightHandSide) + size + iRhs)] =
				rightHandSide[static_cast<std::size_t>(iRow * numRightHandSide + iRhs)];
		}
	}

	for (int iPivot = 0; iPivot < size; ++iPivot) {
		int pivotRow = iPivot;
		double pivotAbs = std::fabs(work[static_cast<std::size_t>(iPivot * (size + numRightHandSide) + iPivot)]);
		for (int iRow = iPivot + 1; iRow < size; ++iRow) {
			const double valueAbs = std::fabs(work[static_cast<std::size_t>(iRow * (size + numRightHandSide) + iPivot)]);
			if (valueAbs > pivotAbs) {
				pivotAbs = valueAbs;
				pivotRow = iRow;
			}
		}
		if (pivotAbs < 1.0e-14) {
			throw std::runtime_error("Singular matrix in dense resolution/covariance operator-chain solve.");
		}
		if (pivotRow != iPivot) {
			for (int iCol = iPivot; iCol < size + numRightHandSide; ++iCol) {
				const std::size_t index0 = static_cast<std::size_t>(iPivot * (size + numRightHandSide) + iCol);
				const std::size_t index1 = static_cast<std::size_t>(pivotRow * (size + numRightHandSide) + iCol);
				const double tmp = work[index0];
				work[index0] = work[index1];
				work[index1] = tmp;
			}
		}

		const double pivot = work[static_cast<std::size_t>(iPivot * (size + numRightHandSide) + iPivot)];
		for (int iCol = iPivot; iCol < size + numRightHandSide; ++iCol) {
			work[static_cast<std::size_t>(iPivot * (size + numRightHandSide) + iCol)] /= pivot;
		}
		for (int iRow = 0; iRow < size; ++iRow) {
			if (iRow == iPivot) {
				continue;
			}
			const double factor = work[static_cast<std::size_t>(iRow * (size + numRightHandSide) + iPivot)];
			for (int iCol = iPivot; iCol < size + numRightHandSide; ++iCol) {
				work[static_cast<std::size_t>(iRow * (size + numRightHandSide) + iCol)] -=
					factor * work[static_cast<std::size_t>(iPivot * (size + numRightHandSide) + iCol)];
			}
		}
	}

	std::vector<double> result(static_cast<std::size_t>(size * numRightHandSide), 0.0);
	for (int iRow = 0; iRow < size; ++iRow) {
		for (int iRhs = 0; iRhs < numRightHandSide; ++iRhs) {
			result[static_cast<std::size_t>(iRow * numRightHandSide + iRhs)] =
				work[static_cast<std::size_t>(iRow * (size + numRightHandSide) + size + iRhs)];
		}
	}
	return result;
}

void validateInput(const AppraisalResolutionCovarianceOperatorChain::DenseInput& input) {
	if (input.numModel <= 0 || input.numData <= 0 || input.numRandomVectors <= 0) {
		throw std::runtime_error("Invalid dense resolution/covariance operator-chain dimensions.");
	}
	validateSize(input.rtrMatrix,
		static_cast<long long>(input.numModel) * static_cast<long long>(input.numModel),
		"R^T R matrix");
	validateSize(input.sensitivityMatrix,
		static_cast<long long>(input.numData) * static_cast<long long>(input.numModel),
		"sensitivity matrix");
	validateSize(input.randomVectors,
		static_cast<long long>(input.numModel) * static_cast<long long>(input.numRandomVectors),
		"random-vector matrix");
}

void accumulateDiagonal(
	const std::vector<double>& resultVector,
	const std::vector<double>& randomVectors,
	const int numModel,
	const int numRandomVectors,
	std::vector<double>& numerator,
	std::vector<double>& denominator,
	std::vector<double>& diagonal)
{
	numerator.assign(static_cast<std::size_t>(numModel), 0.0);
	denominator.assign(static_cast<std::size_t>(numModel), 0.0);
	diagonal.assign(static_cast<std::size_t>(numModel), 0.0);
	for (int iModel = 0; iModel < numModel; ++iModel) {
		for (int iVector = 0; iVector < numRandomVectors; ++iVector) {
			const std::size_t index = static_cast<std::size_t>(iModel * numRandomVectors + iVector);
			numerator[static_cast<std::size_t>(iModel)] +=
				resultVector[index] * randomVectors[index];
			denominator[static_cast<std::size_t>(iModel)] +=
				randomVectors[index] * randomVectors[index];
		}
		if (denominator[static_cast<std::size_t>(iModel)] < 1.0e-12) {
			diagonal[static_cast<std::size_t>(iModel)] = 0.0;
		} else {
			diagonal[static_cast<std::size_t>(iModel)] = std::sqrt(
				std::fabs(numerator[static_cast<std::size_t>(iModel)] /
						  denominator[static_cast<std::size_t>(iModel)]));
		}
	}
}

}

AppraisalResolutionCovarianceOperatorChain::DenseInput::DenseInput()
	: numModel(0),
	  numData(0),
	  numRandomVectors(0)
{
}

AppraisalResolutionCovarianceOperatorChain::ResolutionCovarianceResult::ResolutionCovarianceResult()
	: numModel(0),
	  numData(0),
	  numRandomVectors(0)
{
}

AppraisalResolutionCovarianceOperatorChain::ResolutionCovarianceResult
AppraisalResolutionCovarianceOperatorChain::calculateResolutionCovarianceDiagonal(const DenseInput& input)
{
	validateInput(input);

	ResolutionCovarianceResult result;
	result.numModel = input.numModel;
	result.numData = input.numData;
	result.numRandomVectors = input.numRandomVectors;

	const std::vector<double> inverseRtr = solveLinearSystem(
		input.rtrMatrix,
		input.numModel,
		identity(input.numModel),
		input.numModel);
	result.modifiedSensitivityMatrix = multiply(
		input.sensitivityMatrix,
		input.numData,
		input.numModel,
		inverseRtr,
		input.numModel);

	result.dataSpaceMatrix = multiply(
		result.modifiedSensitivityMatrix,
		input.numData,
		input.numModel,
		transpose(input.sensitivityMatrix, input.numData, input.numModel),
		input.numData);
	for (int iData = 0; iData < input.numData; ++iData) {
		result.dataSpaceMatrix[static_cast<std::size_t>(iData * input.numData + iData)] += 1.0;
	}

	const std::vector<double> covarianceHelperC0 = solveLinearSystem(
		input.rtrMatrix,
		input.numModel,
		input.randomVectors,
		input.numRandomVectors);
	const std::vector<double> covarianceHelperDataRhs = multiply(
		input.sensitivityMatrix,
		input.numData,
		input.numModel,
		covarianceHelperC0,
		input.numRandomVectors);
	const std::vector<double> covarianceHelperDataSolution = solveLinearSystem(
		result.dataSpaceMatrix,
		input.numData,
		covarianceHelperDataRhs,
		input.numRandomVectors);
	const std::vector<double> covarianceHelperBackProjected = multiply(
		transpose(input.sensitivityMatrix, input.numData, input.numModel),
		input.numModel,
		input.numData,
		covarianceHelperDataSolution,
		input.numRandomVectors);
	result.covarianceHelperVector = solveLinearSystem(
		input.rtrMatrix,
		input.numModel,
		subtract(input.randomVectors, covarianceHelperBackProjected),
		input.numRandomVectors);

	const std::vector<double> jTimesRandomVectors = multiply(
		input.sensitivityMatrix,
		input.numData,
		input.numModel,
		input.randomVectors,
		input.numRandomVectors);
	result.resolutionInitialVector = multiply(
		transpose(input.sensitivityMatrix, input.numData, input.numModel),
		input.numModel,
		input.numData,
		jTimesRandomVectors,
		input.numRandomVectors);

	const std::vector<double> jTimesCovarianceHelper = multiply(
		input.sensitivityMatrix,
		input.numData,
		input.numModel,
		result.covarianceHelperVector,
		input.numRandomVectors);
	result.covarianceInitialVector = multiply(
		transpose(input.sensitivityMatrix, input.numData, input.numModel),
		input.numModel,
		input.numData,
		jTimesCovarianceHelper,
		input.numRandomVectors);

	const std::vector<double> resolutionPreconditionedVector = solveLinearSystem(
		input.rtrMatrix,
		input.numModel,
		result.resolutionInitialVector,
		input.numRandomVectors);
	const std::vector<double> covariancePreconditionedVector = solveLinearSystem(
		input.rtrMatrix,
		input.numModel,
		result.covarianceInitialVector,
		input.numRandomVectors);

	const std::vector<double> resolutionDataRhs = multiply(
		input.sensitivityMatrix,
		input.numData,
		input.numModel,
		resolutionPreconditionedVector,
		input.numRandomVectors);
	const std::vector<double> covarianceDataRhs = multiply(
		input.sensitivityMatrix,
		input.numData,
		input.numModel,
		covariancePreconditionedVector,
		input.numRandomVectors);

	const std::vector<double> resolutionDataSolution = solveLinearSystem(
		result.dataSpaceMatrix,
		input.numData,
		resolutionDataRhs,
		input.numRandomVectors);
	const std::vector<double> covarianceDataSolution = solveLinearSystem(
		result.dataSpaceMatrix,
		input.numData,
		covarianceDataRhs,
		input.numRandomVectors);

	const std::vector<double> resolutionBackProjected = multiply(
		transpose(input.sensitivityMatrix, input.numData, input.numModel),
		input.numModel,
		input.numData,
		resolutionDataSolution,
		input.numRandomVectors);
	const std::vector<double> covarianceBackProjected = multiply(
		transpose(input.sensitivityMatrix, input.numData, input.numModel),
		input.numModel,
		input.numData,
		covarianceDataSolution,
		input.numRandomVectors);

	result.resolutionResultVector = solveLinearSystem(
		input.rtrMatrix,
		input.numModel,
		subtract(result.resolutionInitialVector, resolutionBackProjected),
		input.numRandomVectors);
	result.covarianceResultVector = solveLinearSystem(
		input.rtrMatrix,
		input.numModel,
		subtract(result.covarianceInitialVector, covarianceBackProjected),
		input.numRandomVectors);

	accumulateDiagonal(
		result.resolutionResultVector,
		input.randomVectors,
		input.numModel,
		input.numRandomVectors,
		result.dstkr,
		result.dsqkr,
		result.dsdkr);
	accumulateDiagonal(
		result.covarianceResultVector,
		input.randomVectors,
		input.numModel,
		input.numRandomVectors,
		result.dstkc,
		result.dsqkc,
		result.dsdkc);

	return result;
}
