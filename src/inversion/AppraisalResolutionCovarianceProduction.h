//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_APPRAISAL_RESOLUTION_COVARIANCE_PRODUCTION
#define DBLDEF_APPRAISAL_RESOLUTION_COVARIANCE_PRODUCTION

#include <string>
#include <vector>

class DoubleSparseSquareSymmetricMatrix;

namespace AppraisalResolutionCovarianceProduction {

struct RunConfig {
	RunConfig();

	std::string programName;
	std::string programVersion;
	int iteration;
	int appraisalMode;
	int numRandomVectors;
	std::vector<int> checkpoints;
	std::string inputSensitivityDirectory;
	std::string outputDirectory;
	std::string roughnessOperator;
	bool writeLegacyDsdkFiles;
	int expectedNumModel;
	int pardisoMode;
};

struct RunResult {
	RunResult();

	int numModel;
	int numData;
	int fileCount;
	long long totalInputBytes;
	double runtimeSeconds;
	std::string summaryPath;
};

RunResult runProductionAppraisalSummary(
	const RunConfig& config,
	DoubleSparseSquareSymmetricMatrix& rtrMatrix,
	const std::vector<std::string>& sensitivityMatrixFiles);

} // namespace AppraisalResolutionCovarianceProduction

#endif
