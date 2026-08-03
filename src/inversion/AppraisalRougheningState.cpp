//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#include "AppraisalRougheningState.h"

#include <cmath>
#include <stdexcept>

#include "DoubleSparseMatrix.h"

namespace {

void updateValueRange(
	const double value,
	bool& hasValue,
	double& minValue,
	double& maxValue)
{
	if (!hasValue) {
		minValue = value;
		maxValue = value;
		hasValue = true;
		return;
	}
	if (value < minValue) {
		minValue = value;
	}
	if (value > maxValue) {
		maxValue = value;
	}
}

}

namespace AppraisalRougheningState {

SparseMatrixStats::SparseMatrixStats():
	numRows(0),
	numColumns(0),
	numNonZeros(0),
	numDiagonalNonZeros(0),
	numNonFiniteValues(0),
	minValue(0.0),
	maxValue(0.0),
	minAbsNonZeroValue(0.0),
	maxAbsValue(0.0),
	minDiagonalValue(0.0),
	maxDiagonalValue(0.0)
{
}

RougheningStateSummary::RougheningStateSummary():
	roughnessOperator("unknown"),
	expectedNumModel(0),
	dimensionMatchesExpectedModel(false),
	rtrDimensionMatchesConstrainingColumns(false)
{
}

SparseMatrixStats inspectSparseMatrix(const DoubleSparseMatrix& matrix)
{
	if (!matrix.hasConvertedToCRSFormat()) {
		throw std::runtime_error("Sparse matrix must be converted to CRS format before inspection.");
	}

	SparseMatrixStats stats;
	stats.numRows = matrix.getNumRows();
	stats.numColumns = matrix.getNumColumns();

	bool hasValue = false;
	bool hasAbsValue = false;
	bool hasDiagonalValue = false;
	double minAbsValue = 0.0;
	double maxAbsValue = 0.0;

	for (int iRow = 0; iRow < stats.numRows; ++iRow) {
		const int rowBegin = matrix.getRowIndexCRS(iRow);
		const int rowEnd = matrix.getRowIndexCRS(iRow + 1);
		for (int iNonZero = rowBegin; iNonZero < rowEnd; ++iNonZero) {
			const int iCol = matrix.getColumnsCRS(iNonZero);
			const double value = matrix.getValueCRS(iNonZero);
			++stats.numNonZeros;
			if (!std::isfinite(value)) {
				++stats.numNonFiniteValues;
				continue;
			}
			updateValueRange(value, hasValue, stats.minValue, stats.maxValue);
			const double absValue = std::fabs(value);
			if (absValue > 0.0) {
				updateValueRange(absValue, hasAbsValue, minAbsValue, maxAbsValue);
			}
			if (iRow == iCol) {
				++stats.numDiagonalNonZeros;
				updateValueRange(value, hasDiagonalValue, stats.minDiagonalValue, stats.maxDiagonalValue);
			}
		}
	}

	if (hasAbsValue) {
		stats.minAbsNonZeroValue = minAbsValue;
		stats.maxAbsValue = maxAbsValue;
	}
	if (!hasValue) {
		stats.minValue = 0.0;
		stats.maxValue = 0.0;
	}
	if (!hasDiagonalValue) {
		stats.minDiagonalValue = 0.0;
		stats.maxDiagonalValue = 0.0;
	}
	return stats;
}

RougheningStateSummary summarizeRougheningState(
	const DoubleSparseMatrix& constrainingMatrix,
	const DoubleSparseMatrix& rtrMatrix,
	const std::string& roughnessOperator,
	const int expectedNumModel)
{
	RougheningStateSummary summary;
	summary.roughnessOperator = roughnessOperator;
	summary.expectedNumModel = expectedNumModel;
	summary.constrainingMatrix = inspectSparseMatrix(constrainingMatrix);
	summary.rtrMatrix = inspectSparseMatrix(rtrMatrix);
	summary.dimensionMatchesExpectedModel =
		expectedNumModel > 0 &&
		summary.constrainingMatrix.numColumns == expectedNumModel &&
		summary.rtrMatrix.numRows == expectedNumModel &&
		summary.rtrMatrix.numColumns == expectedNumModel;
	summary.rtrDimensionMatchesConstrainingColumns =
		summary.constrainingMatrix.numColumns == summary.rtrMatrix.numRows &&
		summary.rtrMatrix.numRows == summary.rtrMatrix.numColumns;
	return summary;
}

} // namespace AppraisalRougheningState
