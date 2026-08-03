//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_APPRAISAL_ROUGHENING_STATE
#define DBLDEF_APPRAISAL_ROUGHENING_STATE

#include <string>

class DoubleSparseMatrix;

namespace AppraisalRougheningState {

struct SparseMatrixStats {
	SparseMatrixStats();

	int numRows;
	int numColumns;
	long long numNonZeros;
	long long numDiagonalNonZeros;
	long long numNonFiniteValues;
	double minValue;
	double maxValue;
	double minAbsNonZeroValue;
	double maxAbsValue;
	double minDiagonalValue;
	double maxDiagonalValue;
};

struct RougheningStateSummary {
	RougheningStateSummary();

	std::string roughnessOperator;
	int expectedNumModel;
	bool dimensionMatchesExpectedModel;
	bool rtrDimensionMatchesConstrainingColumns;
	SparseMatrixStats constrainingMatrix;
	SparseMatrixStats rtrMatrix;
};

SparseMatrixStats inspectSparseMatrix(const DoubleSparseMatrix& matrix);

RougheningStateSummary summarizeRougheningState(
	const DoubleSparseMatrix& constrainingMatrix,
	const DoubleSparseMatrix& rtrMatrix,
	const std::string& roughnessOperator,
	const int expectedNumModel);

} // namespace AppraisalRougheningState

#endif
