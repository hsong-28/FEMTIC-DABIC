//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2026
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_APPRAISAL_RESOLUTION_COVARIANCE_OPERATOR_CHAIN
#define DBLDEF_APPRAISAL_RESOLUTION_COVARIANCE_OPERATOR_CHAIN

#include <vector>

class AppraisalResolutionCovarianceOperatorChain {
public:
	struct DenseInput {
		DenseInput();

		int numModel;
		int numData;
		int numRandomVectors;
		std::vector<double> rtrMatrix;
		std::vector<double> sensitivityMatrix;
		std::vector<double> randomVectors;
	};

	struct ResolutionCovarianceResult {
		ResolutionCovarianceResult();

		int numModel;
		int numData;
		int numRandomVectors;
		std::vector<double> modifiedSensitivityMatrix;
		std::vector<double> dataSpaceMatrix;
		std::vector<double> covarianceHelperVector;
		std::vector<double> resolutionInitialVector;
		std::vector<double> covarianceInitialVector;
		std::vector<double> resolutionResultVector;
		std::vector<double> covarianceResultVector;
		std::vector<double> dstkr;
		std::vector<double> dsqkr;
		std::vector<double> dsdkr;
		std::vector<double> dstkc;
		std::vector<double> dsqkc;
		std::vector<double> dsdkc;
	};

	static ResolutionCovarianceResult calculateResolutionCovarianceDiagonal(const DenseInput& input);
};

#endif
