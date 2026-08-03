//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_APPRAISAL_SENSITIVITY_MATRIX
#define DBLDEF_APPRAISAL_SENSITIVITY_MATRIX

#include <string>
#include <vector>

class AppraisalSensitivityMatrix {
public:
	enum MatrixKind {
		MATRIX_KIND_UNKNOWN = 0,
		MATRIX_KIND_RAW = 1,
		MATRIX_KIND_MODIFIED = 2
	};

	struct MatrixNameInfo {
		MatrixNameInfo();

		int frequencyID;
		MatrixKind matrixKind;
		bool hasValidName;
	};

	struct MatrixData {
		MatrixData();

		int numData;
		int numModel;
		long long valueCount;
		long long expectedSizeBytes;
		long long actualSizeBytes;
		MatrixNameInfo nameInfo;
		std::vector<double> values;
	};

	static MatrixNameInfo parseSensitivityMatrixFileName(const std::string& fileName);
	static long long calculateExpectedFileSizeBytes(const int numData, const int numModel);

	// Pass a negative expected dimension to skip that dimension check.
	static MatrixData readAndValidate(
		const std::string& fileName,
		const int expectedNumData,
		const int expectedNumModel,
		const bool requireKnownName);
};

#endif
