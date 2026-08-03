//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_SENSITIVITY_DIAGNOSTICS
#define DBLDEF_SENSITIVITY_DIAGNOSTICS

#include <string>
#include <vector>

class SensitivityDiagnostics {
public:
	struct SensitivityEnergyDiagonalResult {
		SensitivityEnergyDiagonalResult();

		int numModel;
		long long fileCount;
		long long totalDataRows;
		std::vector<double> diagonal;
	};

	static SensitivityEnergyDiagonalResult calculateSensitivityEnergyDiagonalFromSensitivityMatrices(
		const std::vector<std::string>& sensitivityMatrixFiles,
		const int expectedNumModel);

	static void writeSensitivityEnergyDiagonalCsv(
		const std::string& fileName,
		const SensitivityEnergyDiagonalResult& result);
};

#endif
