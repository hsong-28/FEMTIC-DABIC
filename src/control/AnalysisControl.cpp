//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// Modified from Copyright (c) 2021 Yoshiya Usui

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//-------------------------------------------------------------------------------------------------------
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include "AppraisalResolutionCovarianceProduction.h"
#include "AppraisalRougheningState.h"
#include "AnalysisControl.h"
#include "ControlKeywords.h"
#include "ResistivityBlock.h"
#include "RougheningMatrix.h"
#include "MeshData.h"
#include "MeshDataBrickElement.h"
#include "MeshDataNonConformingHexaElement.h"
#include "mpi.h"
#include "Forward3D.h"
#include "CommonParameters.h"
#include "ObservedData.h"
#include "OutputFiles.h"
#include "FemticDabicRunSummary.h"
#include "InversionGaussNewtonModelSpace.h"
#include "InversionGaussNewtonDataSpace.h"
#include "Forward3DBrickElement0thOrder.h"
#include "Forward3DTetraElement0thOrder.h"
#include "mkl.h"
#include <assert.h>
#include "InversionGaussNewtonDataSpace_ABIC.h" //ABIC inversion; by Han Song (2024)
#include "InversionGaussNewtonDataSpaceLCurve.h"
#include "InversionGaussNewtonDataSpace_OCCAM.h"
#include "ConstrainingModel.h"					//Cross-Gradient constrainted inversion; by Dieno Diba (2023) & Han Song (2026)

#ifdef _USE_OMP
#include <omp.h>
#endif

namespace {

const char* kLogStartForwardComputationPrefix = "# Start Forward Computation.  Iteration : ";
const char* kLogReuseSelectedTrialForwardResponseCachePrefix =
	"# Reuse selected-trial forward-response cache.  Iteration : ";
const char* kLogCalculateSensitivityWithCachedForwardResponses =
	"# Calculate sensitivity matrix independently while keeping cached forward responses.";
const char* kAppraisalRougheningStateSummaryFile =
	"appraisal_roughening_state_summary.csv";

const char* yesNo(const bool value)
{
	return value ? "yes" : "no";
}

std::string joinPathForAppraisal(const std::string& directory, const std::string& fileName)
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

std::vector<std::string> buildAppraisalSensitivityFileList(const AnalysisControl* const ptrAnalysisControl)
{
	const ObservedData* const ptrObservedData = ObservedData::getInstance();
	const int numFrequencies = ptrObservedData->getTotalNumberOfDifferenetFrequencies();
	std::vector<std::string> files;
	files.reserve(numFrequencies);
	for (int iFreq = 0; iFreq < numFrequencies; ++iFreq) {
		std::ostringstream fileName;
		fileName << "sensMatFreq" << iFreq;
		files.push_back(joinPathForAppraisal(
			ptrAnalysisControl->getAppraisalInputSensitivityDirectory(),
			fileName.str()));
	}
	return files;
}

void writeSparseMatrixStats(
	std::ostream& output,
	const AppraisalRougheningState::SparseMatrixStats& stats)
{
	output << stats.numRows << ","
		   << stats.numColumns << ","
		   << stats.numNonZeros << ","
		   << stats.numDiagonalNonZeros << ","
		   << stats.numNonFiniteValues << ","
		   << stats.minValue << ","
		   << stats.maxValue << ","
		   << stats.minAbsNonZeroValue << ","
		   << stats.maxAbsValue << ","
		   << stats.minDiagonalValue << ","
		   << stats.maxDiagonalValue;
}

void appendAppraisalRougheningStateSummary(
	const int iteration,
	const AnalysisControl* const ptrAnalysisControl,
	const Inversion* const ptrInversion)
{
	if (ptrAnalysisControl->getMyPE() != 0 || !ptrAnalysisControl->isAppraisalEnabled()) {
		return;
	}

	RougheningMatrix constrainingMatrix;
	DoubleSparseSquareSymmetricMatrix rtrMatrix;
	ptrInversion->buildProductionAppraisalRougheningState(constrainingMatrix, rtrMatrix);

	const std::string roughnessOperator =
		ptrAnalysisControl->useDifferenceFilter() ? "difference" : "laplacian";
	const AppraisalRougheningState::RougheningStateSummary summary =
		AppraisalRougheningState::summarizeRougheningState(
			constrainingMatrix,
			rtrMatrix,
			roughnessOperator,
			ptrInversion->getNumberOfModel());

	const bool writeHeader = []() {
		std::ifstream input(kAppraisalRougheningStateSummaryFile);
		return !input.good() || input.peek() == std::ifstream::traits_type::eof();
	}();

	std::ofstream output(kAppraisalRougheningStateSummaryFile, std::ios::app);
	if (output.fail()) {
		OutputFiles::m_logFile << "File open error !! : " << kAppraisalRougheningStateSummaryFile << std::endl;
		exit(1);
	}
	output << std::setprecision(17);
	if (writeHeader) {
		output
			<< "program,version,iteration,appraisal_mode,roughness_operator,"
			<< "expected_num_model,dimension_matches_expected_model,rtr_dimension_matches_constraining_columns,"
			<< "use_difference_filter,lp_degree,lp_epsilon,lp_lower,lp_upper,minimum_norm,distortion_type,run_cross_gradient,"
			<< "r_rows,r_columns,r_nonzeros,r_diagonal_nonzeros,r_nonfinite_values,r_min_value,r_max_value,"
			<< "r_min_abs_nonzero_value,r_max_abs_value,r_min_diagonal_value,r_max_diagonal_value,"
			<< "rtr_rows,rtr_columns,rtr_nonzeros,rtr_diagonal_nonzeros,rtr_nonfinite_values,rtr_min_value,rtr_max_value,"
			<< "rtr_min_abs_nonzero_value,rtr_max_abs_value,rtr_min_diagonal_value,rtr_max_diagonal_value"
			<< std::endl;
	}
	output
		<< CommonParameters::programName << ","
		<< CommonParameters::versionID << ","
		<< iteration << ","
		<< ptrAnalysisControl->getAppraisalMode() << ","
		<< summary.roughnessOperator << ","
		<< summary.expectedNumModel << ","
		<< yesNo(summary.dimensionMatchesExpectedModel) << ","
		<< yesNo(summary.rtrDimensionMatchesConstrainingColumns) << ","
		<< yesNo(ptrAnalysisControl->useDifferenceFilter()) << ","
		<< ptrAnalysisControl->getDegreeOfLpOptimization() << ","
		<< ptrAnalysisControl->getSmallValueOfMinimumGradientSupport() << ","
		<< ptrAnalysisControl->getLowerLimitOfDifflog10RhoForLpOptimization() << ","
		<< ptrAnalysisControl->getUpperLimitOfDifflog10RhoForLpOptimization() << ","
		<< yesNo(ptrAnalysisControl->MinNormInv()) << ","
		<< ptrAnalysisControl->getTypeOfDistortion() << ","
		<< yesNo(ptrAnalysisControl->runCG()) << ",";
	writeSparseMatrixStats(output, summary.constrainingMatrix);
	output << ",";
	writeSparseMatrixStats(output, summary.rtrMatrix);
	output << std::endl;

	OutputFiles::m_logFile
		<< "# Appraisal roughening-state summary is saved in "
		<< kAppraisalRougheningStateSummaryFile
		<< ". Iteration : " << iteration
		<< ", roughness_operator : " << summary.roughnessOperator
		<< ", expected_num_model : " << summary.expectedNumModel
		<< ", dimension_matches_expected_model : " << yesNo(summary.dimensionMatchesExpectedModel)
		<< ", rtr_diagonal_nonzeros : " << summary.rtrMatrix.numDiagonalNonZeros
		<< ", rtr_nonfinite_values : " << summary.rtrMatrix.numNonFiniteValues
		<< std::endl;
}

void runProductionAppraisalSummaryIfRequested(
	const int iteration,
	const bool sensitivityAvailable,
	const AnalysisControl* const ptrAnalysisControl,
	const Inversion* const ptrInversion)
{
	if (!ptrAnalysisControl->isAppraisalEnabled() ||
		!sensitivityAvailable) {
		return;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (ptrAnalysisControl->getMyPE() == 0) {
		RougheningMatrix constrainingMatrix;
		DoubleSparseSquareSymmetricMatrix rtrMatrix;
		ptrInversion->buildProductionAppraisalRougheningState(constrainingMatrix, rtrMatrix);

		AppraisalResolutionCovarianceProduction::RunConfig config;
		config.programName = CommonParameters::programName;
		config.programVersion = CommonParameters::versionID;
		config.iteration = iteration;
		config.appraisalMode = ptrAnalysisControl->getAppraisalMode();
		config.numRandomVectors = ptrAnalysisControl->getNumRandomVectorsForAppraisal();
		config.checkpoints = ptrAnalysisControl->getAppraisalCheckpoints();
		config.inputSensitivityDirectory = ptrAnalysisControl->getAppraisalInputSensitivityDirectory();
		config.outputDirectory = ptrAnalysisControl->getAppraisalOutputDirectory();
		config.roughnessOperator =
			ptrAnalysisControl->useDifferenceFilter() ? "difference" : "laplacian";
		config.writeLegacyDsdkFiles = ptrAnalysisControl->writeLegacyAppraisalDsdkFiles();
		config.expectedNumModel = ptrInversion->getNumberOfModel();
		config.pardisoMode = ptrAnalysisControl->getModeOfPARDISO();

		const std::vector<std::string> sensitivityFiles =
			buildAppraisalSensitivityFileList(ptrAnalysisControl);
		const AppraisalResolutionCovarianceProduction::RunResult result =
			AppraisalResolutionCovarianceProduction::runProductionAppraisalSummary(
				config,
				rtrMatrix,
				sensitivityFiles);
		OutputFiles::m_logFile
			<< "# Appraisal production summary is saved in " << result.summaryPath
			<< ". Iteration : " << iteration
			<< ", appraisal_mode : " << config.appraisalMode
			<< ", num_model : " << result.numModel
			<< ", num_data : " << result.numData
			<< ", file_count : " << result.fileCount
			<< ", runtime_seconds : " << result.runtimeSeconds
			<< std::endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

const char* inversionMethodLabel(const int inversionMethod){

	switch( inversionMethod ){
	case Inversion::GAUSS_NEWTON_MODEL_SPECE:
		return "Gauss-Newton model-space inversion";
	case Inversion::GAUSS_NEWTON_DATA_SPECE:
		return "Gauss-Newton data-space inversion";
	case Inversion::ABIC_DATA_SPECE:
		return "ABIC data-space Gauss-Newton inversion";
	case Inversion::OCCAM_DATA_SPECE:
		return "OCCAM data-space inversion";
	case Inversion::LINEAR_LCURVE_DATA_SPECE:
		return "linear cubic-spline L-curve data-space inversion";
	case Inversion::NONLINEAR_LCURVE_DATA_SPECE:
		return "nonlinear cubic-spline L-curve data-space inversion";
	case Inversion::DATA_FIT_COOLING_DATA_SPECE:
		return "data-fit-bracketed cooling data-space inversion";
	default:
		return "unknown inversion method";
	}
}

const char* tradeOffParameterLabel(const int tradeOffParameterMode){

	switch( tradeOffParameterMode ){
	case AnalysisControl::TO_Fixed:
		return "fixed trade-off parameter";
	case AnalysisControl::TO_ABIC_LS:
		return "ABIC line search";
	case AnalysisControl::TO_OCCAM_LS:
		return "OCCAM line search";
	case AnalysisControl::TO_LINEAR_LCURVE:
		return "linear cubic-spline L-curve selection";
	case AnalysisControl::TO_NONLINEAR_LCURVE:
		return "nonlinear cubic-spline L-curve selection";
	case AnalysisControl::TO_DATA_FIT_COOLING:
		return "data-fit-bracketed cooling";
	default:
		return "unknown trade-off parameter mode";
	}
}

bool isKnownTradeOffParameterMode(const int tradeOffParameterMode){

	switch( tradeOffParameterMode ){
	case AnalysisControl::TO_Fixed:
	case AnalysisControl::TO_ABIC_LS:
	case AnalysisControl::TO_OCCAM_LS:
	case AnalysisControl::TO_LINEAR_LCURVE:
	case AnalysisControl::TO_NONLINEAR_LCURVE:
	case AnalysisControl::TO_DATA_FIT_COOLING:
		return true;
	default:
		return false;
	}
}

bool isImplementedTradeOffParameterMode(const int tradeOffParameterMode){

	return tradeOffParameterMode == AnalysisControl::TO_Fixed ||
		tradeOffParameterMode == AnalysisControl::TO_ABIC_LS ||
		tradeOffParameterMode == AnalysisControl::TO_OCCAM_LS ||
		tradeOffParameterMode == AnalysisControl::TO_LINEAR_LCURVE ||
		tradeOffParameterMode == AnalysisControl::TO_NONLINEAR_LCURVE ||
		tradeOffParameterMode == AnalysisControl::TO_DATA_FIT_COOLING;
}

const char* appraisalModeLabel(const int appraisalMode){

	switch( appraisalMode ){
	case AnalysisControl::APPRAISAL_DISABLED:
		return "disabled";
	case AnalysisControl::APPRAISAL_RESOLUTION_AND_COVARIANCE_DIAGONALS:
		return "model-resolution + covariance diagonals";
	case AnalysisControl::APPRAISAL_RESOLUTION_DIAGONAL:
		return "model-resolution diagonal";
	case AnalysisControl::APPRAISAL_COVARIANCE_DIAGONAL:
		return "covariance diagonal";
	default:
		return "unknown appraisal mode";
	}
}

bool isKnownAppraisalMode(const int appraisalMode){

	return appraisalMode >= AnalysisControl::APPRAISAL_RESOLUTION_AND_COVARIANCE_DIAGONALS &&
		appraisalMode <= AnalysisControl::APPRAISAL_COVARIANCE_DIAGONAL;
}

bool isSupportedAppraisalMode(const int appraisalMode){

	return appraisalMode == AnalysisControl::APPRAISAL_RESOLUTION_AND_COVARIANCE_DIAGONALS ||
		appraisalMode == AnalysisControl::APPRAISAL_RESOLUTION_DIAGONAL ||
		appraisalMode == AnalysisControl::APPRAISAL_COVARIANCE_DIAGONAL;
}

std::vector<int> buildDefaultAppraisalCheckpoints(const int numRandomVectors){

	std::vector<int> checkpoints;
	if (numRandomVectors <= 0) {
		return checkpoints;
	}

	const int firstLargeCheckpoint = 128;
	if (numRandomVectors < firstLargeCheckpoint) {
		for (int checkpoint = 1; checkpoint <= numRandomVectors; checkpoint *= 2) {
			checkpoints.push_back(checkpoint);
			if (checkpoint > numRandomVectors / 2) {
				break;
			}
		}
	} else {
		for (int checkpoint = firstLargeCheckpoint; checkpoint <= numRandomVectors; checkpoint *= 2) {
			checkpoints.push_back(checkpoint);
			if (checkpoint > numRandomVectors / 2) {
				break;
			}
		}
	}

	if (checkpoints.empty() || checkpoints.back() != numRandomVectors) {
		checkpoints.push_back(numRandomVectors);
	}
	return checkpoints;
}

void rejectDeprecatedAppraisalKeyword(const char* const keyword){

	OutputFiles::m_logFile
		<< "Error : " << keyword
		<< " is no longer part of the maintained appraisal control contract. "
		<< "Use APPRAISAL_MODE and APPRAISAL_RANDOM_VECTORS only; "
		<< "checkpoints and appraisal input/output directories are assigned by the program defaults."
		<< std::endl;
	exit(1);
}

std::string abicLineSearchLabel(
	const int tradeOffParameterMode,
	const AnalysisControl::ABICSearchMode searchMode){

	if( tradeOffParameterMode != AnalysisControl::TO_ABIC_LS ){
		return "not used";
	}
	if( searchMode == AnalysisControl::ABIC_SEARCH_EXACT ){
		return "exact";
	}
	return "inexact";
}

std::string regularizationFilterLabel(
	const bool useDifferenceFilter,
	const bool useMinimumNormStabilizer){

	std::string label = useDifferenceFilter ? "difference filter" : "Laplacian filter";
	if( useMinimumNormStabilizer ){
		label += " with minimum-norm stabilizer";
	}
	return label;
}

std::string abicConsoleRegularizationLabel(
	const bool useDifferenceFilter,
	const bool useMinimumNormStabilizer){

	std::string label = useDifferenceFilter ? "Difference Filter" : "Laplacian Filter";
	if( useMinimumNormStabilizer ){
		label += " with Minimum Norm (MN) Stabilizer";
	}
	return label;
}

const char* abicConsoleSearchModeLabel(const AnalysisControl::ABICSearchMode searchMode){

	if( searchMode == AnalysisControl::ABIC_SEARCH_INEXACT ){
		return " # Searching for the trade-off parameter using bracketing (inexact ABIC minimization)";
	}
	return " # Searching for the trade-off parameter that minimizes ABIC through Brent line search";
}

bool shouldReportExactAbicBrentProgress(const AnalysisControl::ABICSearchMode searchMode){

	return searchMode == AnalysisControl::ABIC_SEARCH_EXACT;
}

std::string inversionUpdateLabel(
	const bool levenbergMarquardt,
	const double dampingOfLM){

	if( !levenbergMarquardt ){
		return "Gauss-Newton";
	}

	std::ostringstream label;
	label << "Levenberg-Marquardt damping, damping = " << dampingOfLM;
	return label.str();
}

}

// Return the instance of the class
AnalysisControl *AnalysisControl::getInstance()
{
	static AnalysisControl instance; // The only instance
	return &instance;
}

// Constructor
AnalysisControl::AnalysisControl() :

									 //-----------------------
									 //--- D-DABIC V1.7.0 ---
									 //-----------------------
									 m_abicSearchMode(ABIC_SEARCH_EXACT),
									 m_dataFitCoolingInitialAlpha(std::sqrt(1000.0)),
									 m_dataFitCoolingInitialRmsDecreaseThreshold(0.20),
									 m_dataFitCoolingTriggerThreshold(0.05),
									 m_dataFitCoolingFactor(0.9),
									 m_dataFitCoolingMinimumAlpha(0.1),
									 m_dataFitCoolingHasSelectedAlpha(false),
									 m_dataFitCoolingPersistentAlpha(0.0),
									 m_dataFitCoolingPreviousAcceptedRms(-1.0),
									 m_dataFitCoolingTrialRms(-1.0),
									 m_dataFitCoolingCount(0),
									 m_dataFitCoolingCurrentUpdateIteration(-1),
									 m_dataFitCoolingCurrentAlphaSource("not_selected"),
									 m_stopAfterDataFitCooling(false),
									 //-----------------------
									 //--- D-DABIC V1.6.0 ---
									 //-----------------------
									 m_smallvauleOfMinimumSupport(0.1),
									 m_smallvauleOfMinimumGradientSupport(0.1),
									 //-----------------------
									 //--- D-DABIC V1.5.4 ---
									 //-----------------------
									 m_Levenberg_Marquardt(false),
									 m_dampingof_LM(0.0),
									 //-----------------------
									 //--- D-DABIC V1.5.3 ---
									 //-----------------------
									 m_typeOfReferenceModel(0),
									 m_degreeOfLpMinimumNorm(2),
									 m_lowerLimitOfDifflog10RhoForLpMinimumNorm(0.01),
									 m_upperLimitOfDifflog10RhoForLpMinimumNorm(2.0),
									 //-----------------------
									 //--- D-DABIC V1.5 ---
									 //-----------------------
									 m_tradeOffParameterForCrossGradient(0.0),
									 m_smallvalueForCrossGradient(0.1),
									 m_CrossGradientInv(false),
									 m_typeOfCG(0),
									 //-----------------------
									 //--- D-DABIC V1.3 ---
									 //-----------------------
									 m_tradeOffParameterForMinNorm(0.0),
									 m_MinNormInv(false),
									 //-----------------------
									 //--- D-DABIC V1.2 ---
									 //-----------------------
									 m_tolreq(0.01),
									 m_NumOF_TO(1),
									 m_lCurveUseLogLog(true),
									 m_lCurveUseRootNorm(true),
									 m_hasLCurveSelectionDiagnostics(false),
									 m_lCurveSelectionIteration(-1),
									 m_lCurveModeName(""),
									 m_lCurveRoughnessOperator(""),
									 m_lCurveFailureIndicators(""),
									 m_lCurveSelectedAlpha(-1.0),
									 m_lCurveFinalAlpha(-1.0),
									 m_lCurveSelectedPredictedDataMisfit(-1.0),
									 m_lCurveSelectedPredictedRms(-1.0),
									 m_lCurveSelectedModelRoughness(-1.0),
									 m_lCurveMaxCurvature(-1.0),
									 m_tradeOffParameters(NULL),
									 m_residualVectorThisPE(NULL),
									 m_tradeOffParameterABICA(0.0),
									 m_tradeOffParameterABICB(0.0),
									 m_tradeOffParameterABICC(0.0),
									 m_tradeOffParameterABIClb(0.0),
									 m_tradeOffParameterABICub(0.0),
									 m_ABICA({0.0, 0.0}),
									 m_ABICB({0.0, 0.0}),
									 m_ABICC({0.0, 0.0}),
									 m_abic({0.0, 0.0}),
									 m_abicpre({0.0, 0.0}),
									 m_ABIClb(0.0),
									 m_ABICub(0.0),
									 m_stepsizelb(0.0),
									 m_stepsizeub(0.0),
									 m_ABICconverage(false),
									 m_ABICinversion(false),
									 m_inexactMinimizationOfOCCAM(false),
									 m_updatedmean(0.0),
									 m_OCCAMinversion(false),
									 m_OCCAMsmoothing(false),
									 m_leavingOCCAM(false),
									 m_tradeOffParameterOCCA(0.0),
									 m_tradeOffParameterOCCB(0.0),
									 m_tradeOffParameterOCCC(0.0),
									 m_tradeOffParameterOCClb(0.0),
									 m_tradeOffParameterOCCub(0.0),
									 m_rmsOCCA(0.0),
									 m_rmsOCCB(0.0),
									 m_rmsOCCC(0.0),
									 m_rmsOCClb(0.0),
									 m_rmsOCCub(0.0),
									 m_stepcutOCC(0.0),
									 m_objPre(0.0),
									 m_objPreiter(0.0),
									 m_leavingABIC(false),

									 //--------------------
									 //--- femtic V4.2 ---
									 //--------------------
									 m_myPE(-1),
									 m_totalPE(-1),
									 m_numThreads(1),
									 m_startTime(NULL),
									 m_boundaryConditionBottom(AnalysisControl::BOUNDARY_BOTTOM_PERFECT_CONDUCTOR),
									 m_orderOfFiniteElement(0),
									 m_modeOfPARDISO(PARDISOSolver::INCORE_MODE),
									 m_numberingMethod(AnalysisControl::NOT_ASSIGNED),
									 m_isOutput2DResult(false),
									 m_tradeOffParameterForResistivityValue(1.0),
									 m_tradeOffParameterForResistivityValuePre(1.0),
									 m_tradeOffParameterForDistortionMatrixComplexity(0.0),
									 m_tradeOffParameterForDistortionGain(0.0),
									 m_tradeOffParameterForDistortionRotation(0.0),
									 m_iterationNumInit(0),
									 m_iterationNumMax(0),
									 m_thresholdValueForDecreasing(0.001),
									 m_decreaseRatioForConvegence(1.0),
									 m_stepLengthDampingFactorCur(0.5),
									 m_stepLengthDampingFactorPre(0.5),
									 m_stepLengthDampingFactorMin(0.1),
									 m_stepLengthDampingFactorMax(1.0),
									 m_numOfIterIncreaseStepLength(3),
									 m_factorDecreasingStepLength(0.50),
									 m_factorIncreasingStepLength(1.25),
									 m_numCutbackMax(5),
									 m_holdMemoryForwardSolver(false),
									 m_ptrForward3DBrickElement0thOrder(NULL),
									 m_ptrForward3DTetraElement0thOrder(NULL),
									 m_ptrInversion(NULL),
									 m_ptrInversiondataspace(NULL),
									 m_objectFunctionalPre(0.0),
									 m_dataMisfitPre(0.0),
									 m_modelRoughnessPre(0.0),
									 m_normOfDistortionMatrixDifferencesPre(0.0),
									 m_normOfGainsPre(0.0),
									 m_normOfRotationsPre(0.0),
									 m_numConsecutiveIterFunctionalDecreasing(0),
									 m_continueWithoutCutback(false),
									 m_stopAfterNonlinearLCurveDiagnostics(false),
									 m_maxMemoryPARDISO(3000),
									 m_typeOfMesh(MeshData::HEXA),
									 m_typeOfRoughningMatrix(AnalysisControl::USE_ELEMENTS_SHARE_FACES),
									 m_typeOfElectricField(AnalysisControl::USE_HORIZONTAL_ELECTRIC_FIELD),
									 m_isTypeOfElectricFieldSetIndivisually(false),
									 m_typeOfOwnerElement(AnalysisControl::USE_LOWER_ELEMENT),
									 m_isTypeOfOwnerElementSetIndivisually(false),
									 m_divisionNumberOfMultipleRHSInForward(1),
									 m_divisionNumberOfMultipleRHSInInversion(1),
									 m_binaryOutput(true),
									 m_positiveDefiniteNormalEqMatrix(false),
									 m_typeOfDistortion(AnalysisControl::DISTORTION_TYPE_UNDEFINED),
									 m_inversionMethod(Inversion::GAUSS_NEWTON_MODEL_SPECE),
									 m_isObsLocMovedToCenter(false),
									 m_apparentResistivityAndPhaseTreatmentOption(NO_SPECIAL_TREATMENT_APP_AND_PHASE),
									 m_isRougheningMatrixOutputted(false),
									 m_directoryOfOutOfCoreFilesForSensitivityMatrix("."),
									 m_appraisalMode(AnalysisControl::APPRAISAL_DISABLED),
									 m_numRandomVectorsForAppraisal(8192),
									 m_appraisalInputSensitivityDirectory("."),
									 m_appraisalOutputDirectory("appraisal"),
									 m_writeLegacyAppraisalDsdkFiles(false),

#ifdef _ANISOTOROPY
									 m_typeOfDataSpaceAlgorithm(NEW_DATA_SPACE_ALGORITHM),
									 m_typeOfAnisotropy(NO_ANISOTROPY)
#else
									 m_typeOfDataSpaceAlgorithm(NEW_DATA_SPACE_ALGORITHM),
									 m_useDifferenceFilter(false),
									 m_degreeOfLpOptimization(2),
									 m_lowerLimitOfDifflog10RhoForLpOptimization(0.01),
									 m_upperLimitOfDifflog10RhoForLpOptimization(2.0),
									 m_maxIterationIRWLSForLpOptimization(3),
									 m_thresholdIRWLSForLpOptimization(1.0)
#endif
{
	for (int iDir = 0; iDir < 3; ++iDir)
	{
		m_alphaWeight[iDir] = 1.0;
	}
	m_appraisalCheckpoints.push_back(128);
	m_appraisalCheckpoints.push_back(256);
	m_appraisalCheckpoints.push_back(512);
	m_appraisalCheckpoints.push_back(1024);
	m_appraisalCheckpoints.push_back(2048);
	m_appraisalCheckpoints.push_back(4096);
	m_appraisalCheckpoints.push_back(8192);

#ifdef _USE_OMP
	m_numThreads = omp_get_num_threads();
#endif

	//==============================
	// Germany Time （UTC+1/+2）
	//==============================
	time(&m_startTime); // Measure the start time
	std::tm *berlinTime = gmtime(&m_startTime);
	int germanHour = berlinTime->tm_hour + 1; // CET (UTC+1)

	//
	if (berlinTime->tm_mon >= 2 && berlinTime->tm_mon <= 8)
	{
		germanHour += 1;
	}

	//
	m_germanYear = berlinTime->tm_year + 1900;
	m_germanMonth = berlinTime->tm_mon + 1;
	m_germanDay = berlinTime->tm_mday;
	m_germanHour = germanHour;
	m_germanMin = berlinTime->tm_min;
	m_germanSec = berlinTime->tm_sec;

	// Get process ID and total process number
	MPI_Comm_rank(MPI_COMM_WORLD, &m_myPE);
	MPI_Comm_size(MPI_COMM_WORLD, &m_totalPE);

	// Assign the backward/forward element flags used for EM-field calculation.
	m_useBackwardOrForwardElement.directionX = AnalysisControl::BACKWARD_ELEMENT;
	m_useBackwardOrForwardElement.directionY = AnalysisControl::BACKWARD_ELEMENT;
}

// Destructor
AnalysisControl::~AnalysisControl()
{

	if (!m_outputParametersForVis.empty())
	{
		m_outputParametersForVis.clear();
	}

	if (m_ptrForward3DBrickElement0thOrder != NULL)
	{
		delete m_ptrForward3DBrickElement0thOrder;
		m_ptrForward3DBrickElement0thOrder = NULL;
	}

	if (m_ptrForward3DTetraElement0thOrder != NULL)
	{
		delete m_ptrForward3DTetraElement0thOrder;
		m_ptrForward3DTetraElement0thOrder = NULL;
	}

	if (m_ptrInversion != NULL)
	{
		m_ptrInversion->deleteOutOfCoreFileAll();
		delete m_ptrInversion;
		m_ptrInversion = NULL;
	}

	if (m_ptrInversiondataspace != NULL)
	{
		delete m_ptrInversiondataspace;
		m_ptrInversiondataspace = NULL;
	}

	if (m_tradeOffParameters != NULL)
	{
		delete[] m_tradeOffParameters;
		m_tradeOffParameters = NULL;
	}

	if (m_residualVectorThisPE != NULL)
	{
		delete[] m_residualVectorThisPE;
		m_residualVectorThisPE = NULL;
	}
}

void AnalysisControl::run()
{

	const FemticDabicRunSummary::StartupSummary startupSummary = {
		outputElapsedTime(),
		m_germanYear,
		m_germanMonth,
		m_germanDay,
		m_germanHour,
		m_germanMin,
		m_germanSec
	};
	FemticDabicRunSummary::outputStartupLog(OutputFiles::m_logFile, startupSummary);

	// Get process ID
	const int myProcessID = getMyPE();

	if (myProcessID == 0)
	{
		FemticDabicRunSummary::outputStartupConsole(std::cout, startupSummary);
	}

	//---------------------------------------------------
	//--- Read analysis control data from control.dat ---
	//---------------------------------------------------

	OutputFiles::m_logFile << "# Read analysis control data from control.dat." << outputElapsedTime() << std::endl;
	inputControlData();
	int iterInit = m_iterationNumInit;
	const FemticDabicRunSummary::RunConfigurationSummary runConfigurationSummary = {
		m_iterationNumInit,
		m_iterationNumMax,
		inversionMethodLabel(m_inversionMethod),
		tradeOffParameterLabel(m_typeOfTradeOffParam),
		abicLineSearchLabel(
			m_typeOfTradeOffParam,
			m_abicSearchMode),
		regularizationFilterLabel(m_useDifferenceFilter, m_MinNormInv),
		inversionUpdateLabel(m_Levenberg_Marquardt, m_dampingof_LM)
	};
	FemticDabicRunSummary::outputRunConfigurationLog(OutputFiles::m_logFile, runConfigurationSummary);
	if (myProcessID == 0)
	{
		FemticDabicRunSummary::outputRunConfigurationConsole(std::cout, runConfigurationSummary);
	}

	//-------------------------------------------------------
	//--- Create object of Forward analysis and inversion ---
	//-------------------------------------------------------
	if (m_typeOfMesh == MeshData::HEXA)
	{
		m_ptrForward3DBrickElement0thOrder = new Forward3DBrickElement0thOrder();
	}
	else if (m_typeOfMesh == MeshData::TETRA)
	{
		m_ptrForward3DTetraElement0thOrder = new Forward3DTetraElement0thOrder();
	}
	else if (m_typeOfMesh == MeshData::NONCONFORMING_HEXA)
	{
		m_ptrForward3DNonConformingHexaElement0thOrder = new Forward3DNonConformingHexaElement0thOrder();
	}
	else
	{
		OutputFiles::m_logFile << "Error : Type of mesh is wrong !! : " << m_typeOfMesh << "." << std::endl;
		exit(1);
	}

	switch (getInversionMethod())
	{
	case Inversion::GAUSS_NEWTON_MODEL_SPECE:
		m_ptrInversion = new InversionGaussNewtonModelSpace();
		break;
	case Inversion::GAUSS_NEWTON_DATA_SPECE:
		m_ptrInversion = new InversionGaussNewtonDataSpace();
		break;
	case Inversion::ABIC_DATA_SPECE:
		// ABIC inversion;
		m_ptrInversion = new InversionGaussNewtonDataSpace_ABIC();
		break;
	case Inversion::LINEAR_LCURVE_DATA_SPECE:
		m_ptrInversion = new InversionGaussNewtonDataSpaceLCurve();
		m_ptrInversiondataspace = new InversionGaussNewtonDataSpace();
		break;
	case Inversion::OCCAM_DATA_SPECE:
		m_ptrInversion = new InversionGaussNewtonDataSpace_OCCAM();
		break;
	case Inversion::NONLINEAR_LCURVE_DATA_SPECE:
		m_ptrInversion = new InversionGaussNewtonDataSpaceLCurve();
		m_ptrInversiondataspace = new InversionGaussNewtonDataSpace();
		break;
	case Inversion::DATA_FIT_COOLING_DATA_SPECE:
		m_ptrInversion = new InversionGaussNewtonDataSpace();
		break;
	default:
		OutputFiles::m_logFile << "Error : Type of inversion method is wrong  !! : " << getInversionMethod() << std::endl;
		exit(1);
		break;
	}

	//------------------------------------
	//--- Read mesh data from mesh.dat ---
	//------------------------------------
	OutputFiles::m_logFile << "# Read mesh data from mesh.dat ." << outputElapsedTime() << std::endl;
	getPointerOfForward3D()->callInputMeshData();

	//---------------------------------------------
	//--- Read resistivity-block model data ---
	//---------------------------------------------
	OutputFiles::m_logFile << "# Read resistivity-block model data." << outputElapsedTime() << std::endl;
	ResistivityBlock *pResistivityBlock = ResistivityBlock::getInstance();
	pResistivityBlock->inputResisitivityBlock();

	if (m_MinNormInv)
	{
		//---------------------------------------------
		//--- Read data of reference model ---
		//---------------------------------------------
		OutputFiles::m_logFile << "# Read reference model ." << outputElapsedTime() << std::endl;
		pResistivityBlock->inputReferenceModel();
	}

	if (m_CrossGradientInv)
	{
		//---------------------------------------------
		//--- Read data of constraining model ---
		//---------------------------------------------
		OutputFiles::m_logFile << "# Read data of constraining model ." << outputElapsedTime() << std::endl;
		ConstrainingModel *pConstrainingModel = ConstrainingModel::getInstance();
		pConstrainingModel->inputConstrainingModel();
	}

	//-------------------------------------------
	//--- Read observed data from observe.dat ---
	//-------------------------------------------
	OutputFiles::m_logFile << "# Read observed data ." << outputElapsedTime() << std::endl;
	ObservedData *pObservedData = ObservedData::getInstance();
	pObservedData->inputObservedData();
	pObservedData->calcFrequenciesCalculatedByThisPE();

	const int nfreq = pObservedData->getTotalNumberOfDifferenetFrequencies();
	if (nfreq <= 0)
	{
		OutputFiles::m_logFile << "Error : Total number of frequencies is less than zero !!" << std::endl;
		exit(1);
	}

	//-----------------------------------
	//--- Read distortion matrix data ---
	//-----------------------------------
	if (estimateDistortionMatrix())
	{
		OutputFiles::m_logFile << "# Read distortion matrix data ." << outputElapsedTime() << std::endl;
		// pObservedData->inputStaticShiftData();
		pObservedData->inputDistortionMatrixData();
	}

	//------------------------------------------------------------------------------------------
	//--- Allocate memory for the calculated values and errors of all stations               ---
	//--- after setting up frequencies calculated by this PE, at which observed value exists ---
	//------------------------------------------------------------------------------------------
	pObservedData->allocateMemoryForCalculatedValuesOfAllStations();

	//-------------------------------------------
	//--- Find element including each station ---
	//-------------------------------------------
	OutputFiles::m_logFile << "# Find element including each station ." << outputElapsedTime() << std::endl;
	pObservedData->findElementIncludingEachStation();

	//------------------------------------------------------------------------
	//--- Output information of locations of observed stations to vtk file ---
	//------------------------------------------------------------------------
	OutputFiles *const ptrOutputFiles = OutputFiles::getInstance();
	if (myProcessID == 0)
	{ // If this PE number is zero
		ptrOutputFiles->openVTKFileForObservedStation();
		pObservedData->outputLocationsOfObservedStationsToVtk();
	}

	//----------------------------------------------------------------
	//--- Initialize response functions and errors of all stations ---
	//----------------------------------------------------------------
	pObservedData->initializeResponseFunctionsAndErrorsOfAllStations();

	//-----------------------------------------------------
	//--- Output number of model parameters to log file ---
	//-----------------------------------------------------
	m_ptrInversion->outputNumberOfModel();

	//-----------------------------------
	//--- Calculate Roughening Matrix ---
	//-----------------------------------
	OutputFiles::m_logFile << "# Calculate Roughening Matrix ." << outputElapsedTime() << std::endl;
	pResistivityBlock->calcRougheningMatrix();

	// if (m_CrossGradientInv)
	// {
	// 	//-----------------------------------
	// 	//--- Calculate Cross-gradient Matrix ---
	// 	//-----------------------------------
	// 	OutputFiles::m_logFile << "# Calculate Cross-gradient Matrix ." << outputElapsedTime() << std::endl;
	// 	pResistivityBlock->calcCrossGradientMatrix();
	// }

	//-------------------------------------------
	//--- Output geometory file and case file ---
	//-------------------------------------------
	if (!m_outputParametersForVis.empty() && writeBinaryFormat() && myProcessID == 0)
	{ // Write to BINARY file
		ptrOutputFiles->outputCaseFile();
		getPointerOfMeshData()->outputMeshDataToBinary();
		pResistivityBlock->outputResistivityDataToBinary();
	}

	//---------------------
	//--- Open cnv file ---
	//---------------------
	if (myProcessID == 0)
	{
		ptrOutputFiles->openCnvFile(iterInit);
	}
	AnalysisControl::ConvergenceBehaviors convergenceFlag = AnalysisControl::DURING_RETRIALS;

	for (int iter = m_iterationNumInit; iter <= m_iterationNumMax; ++iter)
	{

		m_iterationNumCurrent = iter;
		m_residualupdated = 0;

		if (m_leavingABIC)
		{
			OutputFiles::m_logFile << "# Leaving ABIC." << std::endl;
			if (myProcessID == 0)
			{
				std::cout << " # Leaving ABIC." << std::endl;
			}
			break;
		}

		if (m_CrossGradientInv)
		{
			//-----------------------------------
			//--- Calculate Cross-gradient Matrix ---
			//-----------------------------------
			OutputFiles::m_logFile << "# Calculate Cross-gradient Matrix ." << outputElapsedTime() << std::endl;
			pResistivityBlock->calcCrossGradientMatrix();
		}
		appendAppraisalRougheningStateSummary(iter, this, m_ptrInversion);

		// Open csv file in which the results of 2D forward computations is written
		if (m_isOutput2DResult)
		{
			ptrOutputFiles->openCsvFileFor2DFwd(iter);
		}

		if (!m_outputParametersForVis.empty() && !writeBinaryFormat())
		{
			ptrOutputFiles->openVTKFile(iter);
			pResistivityBlock->outputResistivityDataToVTK();
		}

		if (m_iterationNumMax > iter && doesOutputToVTK(AnalysisControl::OUTPUT_SENSITIVITY))
		{ // if output sensitivity
			m_ptrInversion->allocateMemoryForSensitivityScalarValues();
		}

		int iCutBack = 0;
		ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
		for (; iCutBack <= m_numCutbackMax; ++iCutBack)
		{
			seticut(iCutBack);
			OutputFiles::m_logFile << "###############################################################################" << std::endl;
			OutputFiles::m_logFile << kLogStartForwardComputationPrefix << iter << ",  Retrial : " << iCutBack << std::endl;
			OutputFiles::m_logFile << "###############################################################################" << std::endl;

			//---------------------------
			//--- Forward computation ---
			//---------------------------
			const bool reuseSelectedTrialForwardResponseCache =
				iCutBack == 0 && canUseSelectedTrialForwardResponseCache(iter);
			const bool calculateSensitivity = doesCalculateSensitivity(iter);
			const double acceptedForwardStartSec = getElapsedTimeInSeconds();
			if (reuseSelectedTrialForwardResponseCache)
			{
				OutputFiles::m_logFile << kLogReuseSelectedTrialForwardResponseCachePrefix << iter << ",  Retrial : " << iCutBack << std::endl;
				if (calculateSensitivity)
				{
					OutputFiles::m_logFile << kLogCalculateSensitivityWithCachedForwardResponses << std::endl;
					calcForwardComputation(iter, true);
				}
			}
			else
			{
				calcForwardComputation(iter, false);
			}
			OutputFiles::m_logFile << "# Timing accepted forward stage. Iteration : " << iter
								   << ",  Retrial : " << iCutBack
								   << ", response_cache_reused : " << (reuseSelectedTrialForwardResponseCache ? "yes" : "no")
								   << ", sensitivity_requested : " << (calculateSensitivity ? "yes" : "no")
								   << ", elapsed : " << getElapsedTimeInSeconds() - acceptedForwardStartSec
								   << " sec." << std::endl;
			m_updatedmean = ptrResistivityBlock->calctResistivityUpdatedratio();
			//--------------------------------------------
			//--- Output information about convergence ---
			//--------------------------------------------
			convergenceFlag = adjustStepLengthDampingFactor(iter, iCutBack);
			if (convergenceFlag == AnalysisControl::GO_TO_NEXT_ITERATION || convergenceFlag == AnalysisControl::INVERSIN_CONVERGED)
			{
				runProductionAppraisalSummaryIfRequested(
					iter,
					calculateSensitivity,
					this,
					m_ptrInversion);
				break; // Go out of the loop
			}
			if (isDataFitCoolingMode() && m_stopAfterDataFitCooling)
			{
				break;
			}

			//-----------------------------------------------------------
			//--- Change resistivity values and distortion parameters ---
			//-----------------------------------------------------------
			if (m_typeOfTradeOffParam == AnalysisControl::TO_ABIC_LS)
			{
				pResistivityBlock->updateResistivityValues_aut();
			}
			else
			{
				pResistivityBlock->updateResistivityValues();
			}
			pObservedData->updateDistortionParams();
		}
		if (m_CrossGradientInv)
		{
			pResistivityBlock->updateCrossGradientValues();
			if (myProcessID == 0 && iter > m_iterationNumInit)
			{ // If this PE number is zero and iteration number is not the first one
				pResistivityBlock->outputCrossGradientBlock(iter);
			}
		}

		if (m_stopAfterDataFitCooling)
		{
			OutputFiles::m_logFile
				<< "# Stop inversion loop because the selected full-step cooling response was not reproducible."
				<< std::endl;
			break;
		}

		if (iCutBack > m_numCutbackMax)
		{
			OutputFiles::m_logFile << "# Reach maximum retrial number." << std::endl;
			break;
		}

		// Output induction arrows to vtk file
		pObservedData->outputInductionArrowToVtk(iter);

		// Open csv file in which the results of 3D forward computations is written
		ptrOutputFiles->openCsvFileFor3DFwd(iter);
		// Output results
		pObservedData->outputCalculatedValuesOfAllStations();

		// Output resistivity model
		if (writeBinaryFormat())
		{ // Write to BINARY file
			if (myProcessID == 0)
			{
				pResistivityBlock->outputResistivityValuesToBinary(iter);
			}
		}
		else
		{ // Write to ASCII file
			pResistivityBlock->outputResistivityValuesToVTK();
		}

		if (myProcessID == 0 && iter > m_iterationNumInit)
		{ // If this PE number is zero and iteration number is not the first one
			pResistivityBlock->outputResisitivityBlock(iter);
			pResistivityBlock->output3DResistivity(iter);
			if (estimateDistortionMatrix())
			{
				pObservedData->outputDistortionParams(iter);
			}
		}
		if (iter > m_iterationNumInit && m_MinNormInv && m_typeOfReferenceModel == AnalysisControl::AfterAdjustment)
		{
			ptrResistivityBlock->copyResistivityValuesNotFixedCurToReferenceModel();
		}

		// Output sensitivity
		if (doesCalculateSensitivity(iter) && doesOutputToVTK(AnalysisControl::OUTPUT_SENSITIVITY))
		{ // if output sensitivity
			if (writeBinaryFormat())
			{ // Write to BINARY file
				m_ptrInversion->outputSensitivityScalarValuesToBinary(iter);
			}
			else
			{ // Write to ASCII file
				m_ptrInversion->outputSensitivityScalarValuesToVtk(iter);
			}
			m_ptrInversion->releaseMemoryOfSensitivityScalarValues();
		}

		if (m_ABICconverage)
		{
			if (myProcessID == 0)
			{
				OutputFiles::m_logFile << "# Tolerance met. Leaving ABIC." << std::endl;
				std::cout << "# Tolerance met. Leaving ABIC." << std::endl;
			}
		}
		if (m_leavingOCCAM)
		{
			if (myProcessID == 0)
			{
				OutputFiles::m_logFile << "# Leaving OCCAM." << std::endl;
				std::cout << "# Leaving OCCAM." << std::endl;
			}
		}

		//-----------------
		//--- Inversion ---
		//-----------------
		if (convergenceFlag == AnalysisControl::INVERSIN_CONVERGED)
		{
			OutputFiles::m_logFile << "# Converged." << std::endl;
			break;
		}

		if (iter >= m_iterationNumMax)
		{
			OutputFiles::m_logFile << "# Reach maximum iteration number." << std::endl;
			break;
		}

		OutputFiles::m_logFile << "###############################################################################" << std::endl;
		OutputFiles::m_logFile << "# Start Inversion.  Iteration : " << iter << std::endl;
		OutputFiles::m_logFile << "###############################################################################" << std::endl;

		ObservedData *const ptrObservedData = ObservedData::getInstance();
		ptrResistivityBlock->copyResistivityValuesNotFixedCurToPre(); //
		ptrObservedData->copyDistortionParamsCurToPre();

		if (useDifferenceFilter())
		{
			const int maxIter = getMaxIterationIRWLSForLpOptimization();
			double modelRoughnessPre = ptrResistivityBlock->calcModelRoughnessForDifferenceFilter();
			for (int iter = 0; iter < maxIter; ++iter)
			{
				OutputFiles::m_logFile << "# Iteration number of reweighted iterative algorithm for Lp optimization : " << iter + 1 << std::endl;
				if (m_typeOfTradeOffParam == AnalysisControl::TO_Fixed)
				{
					if (myProcessID == 0)
					{
						if (m_MinNormInv && m_tradeOffParameterForMinNorm > CommonParameters::EPS)
						{
							std::cout << " # Difference Filter with Minimum Norm (MN) Stabilizer." << std::endl;
						}
					}
					// m_tradeOffParameterForResistivityValue = m_tradeOffParameterForResistivityValue;
					m_ptrInversion->inversionCalculation();
				}
				else if (m_typeOfTradeOffParam == AnalysisControl::TO_ABIC_LS)
				{
					if (myProcessID == 0)
					{
						std::cout << " # Entering ABIC ("
								  << abicConsoleRegularizationLabel(
									  useDifferenceFilter(),
									  m_MinNormInv && m_tradeOffParameterForMinNorm > CommonParameters::EPS)
								  << ")." << std::endl;
						std::cout << abicConsoleSearchModeLabel(m_abicSearchMode) << std::endl;
					}
					int numDataThisPE = ptrObservedData->getNumObservedDataThisPETotal();
					OutputFiles::m_logFile << "# Number of data of this PE : " << numDataThisPE << std::endl;
					if (m_residualVectorThisPE != NULL)
					{
						delete[] m_residualVectorThisPE;
						m_residualVectorThisPE = NULL;
					}
					m_residualVectorThisPE = new double[numDataThisPE];
					ptrObservedData->calculateResidualVectorOfDataThisPE(m_residualVectorThisPE); // d-F(m_pre)
					m_abic = m_abicpre;
					m_stepLengthDampingFactorPre = m_stepLengthDampingFactorCur;
					iCutBack = 0;
					for (; iCutBack < m_numCutbackMax; iCutBack++)
					{
						m_stepLengthDampingFactorCur = (1.0 / pow(2.0, iCutBack)) * m_stepLengthDampingFactorPre;
						if (m_stepLengthDampingFactorCur < m_stepLengthDampingFactorMin)
						{
							OutputFiles::m_logFile << "# Model update is too small." << std::endl;
							if (myProcessID == 0)
							{
								std::cout << "# Model update is too small." << std::endl;
							}
							m_leavingABIC = true;
							break;
						}
						if (shouldReuseInexactABICAlphaOnCutback(iCutBack))
						{
							if (myProcessID == 0)
							{
								std::cout << " # Reusing current inexact ABIC trade-off parameter with reduced step size." << std::endl;
								std::cout << "# Inexact ABIC cutback trial. alpha : "
										  << m_tradeOffParameterForResistivityValue
										  << ", step_length : " << m_stepLengthDampingFactorCur
										  << ", cutback_count : " << iCutBack << "." << std::endl;
							}
							OutputFiles::m_logFile << "# Reusing current inexact ABIC trade-off parameter with reduced step size." << std::endl;
							OutputFiles::m_logFile << "# Inexact ABIC cutback trial. alpha : "
											   << m_tradeOffParameterForResistivityValue
											   << ", step_length : " << m_stepLengthDampingFactorCur
											   << ", cutback_count : " << iCutBack << "." << std::endl;
							runReducedStepTrialWithCurrentABICAlpha();
							if (m_abic[1] < m_tolreq)
							{
								m_ABICconverage = true;
								break;
							}
							if (m_iterationNumCurrent > m_iterationNumInit)
							{
								if (m_abic[0] < m_abicpre[0] && m_abic[1] < m_abicpre[1])
								{
									break;
								}
								if (myProcessID == 0)
								{
									if (m_abicpre[1] <= m_abic[1])
									{
										std::cout << " # m_dataMisfitPre: " << m_abicpre[1] << "  <  " << "m_dataMisfitCur: " << m_abic[1] << std::endl;
									}
									if (m_abicpre[0] <= m_abic[0])
									{
										std::cout << " # m_abicPre: " << m_abicpre[0] << "  <  " << "m_abicCur: " << m_abic[0] << std::endl;
									}
									if (m_abicpre[1] <= m_abic[1] || m_abicpre[0] <= m_abic[0])
									{
										std::cout << " # Cutting the stepsize and re-testing current inexact alpha " << std::endl;
										std::cout << " # ...... " << std::endl;
									}
								}
								m_numConsecutiveIterFunctionalDecreasing = 0;
							}
							else
							{
								if (m_abic[1] < m_rmsPre)
								{
									break;
								}
								if (myProcessID == 0)
								{
									std::cout << " # m_dataMisfitPre: " << m_rmsPre << "  <  " << "m_dataMisfitCur: " << m_abic[1] << std::endl;
									std::cout << " # Cutting the stepsize and re-testing current inexact alpha " << std::endl;
									std::cout << " # ...... " << std::endl;
								}
								m_numConsecutiveIterFunctionalDecreasing = 0;
							}
							continue;
						}
						m_tradeOffParameterForResistivityValue = m_tradeOffParameterForResistivityValuePre;
						m_tradeOffParameterABICA = log10(m_tradeOffParameterForResistivityValue);
						m_tradeOffParameterABICB = m_tradeOffParameterABICA - getInitialABICLog10BracketSpan();
						if (myProcessID == 0)
						{
							if (shouldReportExactAbicBrentProgress(m_abicSearchMode))
							{
								std::cout << " # ...Bracketing Minimum..." << std::endl;
							}
						}
						minbrkABIC(); // m_ABICB[0] = min(abic); pwk1 is the corresponding model vector
						if (m_ABICB[1] < m_tolreq)
						{
							m_ABICconverage = true;
							m_abic = m_ABICB;
							m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterABICB);
							break;
						}
						else
						{
							if (myProcessID == 0)
							{
								if (shouldReportExactAbicBrentProgress(m_abicSearchMode))
									std::cout << " # ...Finding minimum by Brent's minimizing method..." << std::endl;
							}
							m_abic = fminbrentABIC(iCutBack); // pwk1 is the corresponding model vector

							if (myProcessID == 0)
							{
								if (shouldReportExactAbicBrentProgress(m_abicSearchMode))
								{
									std::cout << " # Minimum ABIC from fminbrent is at trade-off parameter = " << m_tradeOffParameterForResistivityValue << std::endl;
								}
								else 
								{
									std::cout << " # Inexact Minimum ABIC is at trade-off parameter = " << m_tradeOffParameterForResistivityValue << std::endl;
								}
							}
#ifdef FEMTIC_DABIC_TEST_FORCE_FIRST_INEXACT_ABIC_CUTBACK
							if (usesInexactABICSearch() && iCutBack == 0)
							{
								if (myProcessID == 0)
								{
									std::cout << "# TEST HOOK: forcing first inexact ABIC trial rejection." << std::endl;
								}
								OutputFiles::m_logFile << "# TEST HOOK: forcing first inexact ABIC trial rejection." << std::endl;
								continue;
							}
#endif
							if (m_abic[1] < m_tolreq)
							{
								m_ABICconverage = true;
								break;
							}

							if (m_iterationNumCurrent > m_iterationNumInit)
							{
								if (m_abic[0] < m_abicpre[0] && m_abic[1] < m_abicpre[1])
								{
									break;
								}
								else
								{
									if (myProcessID == 0)
									{
										if (m_abicpre[1] <= m_abic[1])
										{
											std::cout << " # m_dataMisfitPre: " << m_abicpre[1] << "  <  " << "m_dataMisfitCur: " << m_abic[1] << std::endl;
										}
										if (m_abicpre[0] <= m_abic[0])
										{
											std::cout << " # m_abicPre: " << m_abicpre[0] << "  <  " << "m_abicCur: " << m_abic[0] << std::endl;
										}
										if (m_abicpre[1] <= m_abic[1] || m_abicpre[0] <= m_abic[0])
										{
											std::cout << " # Cutting the stepsize and re-searching " << std::endl;
											std::cout << " # ...... " << std::endl;
										}
									}
									m_numConsecutiveIterFunctionalDecreasing = 0; // reset value
								}
							}
							else
							{
								if (m_abic[1] < m_rmsPre)
								{
									break;
								}
								else
								{
									if (myProcessID == 0)
									{
										std::cout << " # m_dataMisfitPre: " << m_rmsPre << "  <  " << "m_dataMisfitCur: " << m_abic[1] << std::endl;
										std::cout << " # Cutting the stepsize and re-searching " << std::endl;
										std::cout << " # ...... " << std::endl;
									}
									m_numConsecutiveIterFunctionalDecreasing = 0; // reset value
								}
							}
						}
					}
					if (iCutBack == m_numCutbackMax)
					{
						OutputFiles::m_logFile << "# Reach maximum retrial number." << std::endl;
						if (myProcessID == 0)
						{
							std::cout << " # Reach maximum retrial number." << std::endl;
						}
						m_leavingABIC = true;
					}

					if (m_ABICconverage)
					{
						// TOLERANCE IS BELOW THAT REQUIRED; FIND INTERCEPT.
						if (myProcessID == 0)
						{
							std::cout << " # Finding Intercept: bracketing the root (RMS - m_tolreq = 0)..." << std::endl;
						}
						m_stepsizelb = m_stepLengthDampingFactorCur;
						m_ABIClb = m_abic;
						m_stepsizeub = m_stepLengthDampingFactorCur;
						m_ABICub = m_abic;
						int count = 0;
						m_stepLengthDampingFactorPre = m_stepLengthDampingFactorCur;
						while (m_ABICub[1] < m_tolreq)
						{
							m_ABIClb = m_ABICub;
							m_stepsizelb = m_stepsizeub;
							if (count > 0)
							{
								ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
								ptrObservedData->copyDistortionParamsCurToPWK1();
							}
							count += +1;
							m_stepLengthDampingFactorCur = (1.0 / pow(2.0, count)) * m_stepLengthDampingFactorPre;
							m_stepsizeub = m_stepLengthDampingFactorCur;
							m_ptrInversion->inversionCalculation();
							m_ABICub = m_ptrInversion->getabic();
						} // after this loop, m_ABICub >= m_tolreq; m_tradeOffParameterABICub > m_tradeOffParameterABIClb.
						if (myProcessID == 0)
						{
							std::cout << " # Finding Intercept: approaching the root (RMS - m_tolreq = 0)..." << std::endl;
						}
						ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
						ptrObservedData->copyDistortionParamsCurToPWK2();
						m_stepLengthDampingFactorCur = frootABIC();
						if (myProcessID == 0)
						{
							std::cout << " # Tolerance is met (approximately) at trade-off parameter  = " << m_tradeOffParameterForResistivityValue << "  with step_size = " << m_stepLengthDampingFactorCur << std::endl;
						}
						ptrResistivityBlock->copyPWK2NotFixedToPWK1();
						ptrObservedData->copyDistortionParamsPWK2ToPWK1();
					}
					ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
					ptrObservedData->copyDistortionParamsPWK1ToCur();
					if (ptrObservedData->hasSelectedTrialForwardResponseCache())
					{
						ptrObservedData->restoreSelectedTrialForwardResponseCache();
					}
				}
				else if (m_typeOfTradeOffParam == AnalysisControl::TO_OCCAM_LS)
				{
					runOCCAMLineSearch("Difference Filter");
				}
				else if (m_typeOfTradeOffParam == AnalysisControl::TO_LINEAR_LCURVE)
				{
					if (m_ptrInversiondataspace == NULL)
					{
						OutputFiles::m_logFile << "Error : L-curve final data-space inversion object is NULL." << std::endl;
						exit(1);
					}
					if (myProcessID == 0)
					{
						std::cout << " # Entering linear cubic-spline L-curve selection (Difference Filter)." << std::endl;
					}
					m_ptrInversion->inversionCalculation();
					if (myProcessID == 0)
					{
						const double selectedTradeOff = m_ptrInversion->getAlphawithmaxc();
						std::cout << " # Linear L-curve selected trade-off parameter : " << selectedTradeOff << std::endl;
						m_tradeOffParameterForResistivityValue = selectedTradeOff;
						setLCurveFinalTradeOffParameterForDiagnostics(m_tradeOffParameterForResistivityValue);
						std::cout << " # Final linear L-curve trade-off parameter : " << m_tradeOffParameterForResistivityValue << std::endl;
					}
					MPI_Bcast(&m_tradeOffParameterForResistivityValue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
					m_ptrInversiondataspace->inversionCalculation();
				}
				else if (m_typeOfTradeOffParam == AnalysisControl::TO_NONLINEAR_LCURVE)
				{
					runNonlinearLCurveDiagnostics(m_iterationNumCurrent, "Difference Filter");
					if (!m_stopAfterNonlinearLCurveDiagnostics)
					{
						m_ptrInversiondataspace->inversionCalculation();
					}
				}
				else if (m_typeOfTradeOffParam == AnalysisControl::TO_DATA_FIT_COOLING)
				{
					if (!m_dataFitCoolingHasSelectedAlpha)
					{
						m_stopAfterDataFitCooling = !runInitialDataFitCoolingBracket();
					}
					else
					{
						m_stopAfterDataFitCooling = !runPersistentDataFitCoolingAlpha();
					}
				}
				const double modelRoughness = ptrResistivityBlock->calcModelRoughnessForDifferenceFilter();
				OutputFiles::m_logFile << "# Model-roughness is changed from " << modelRoughnessPre << " to " << modelRoughness << std::endl;
				if (modelRoughnessPre > CommonParameters::EPS &&
					fabs(modelRoughness - modelRoughnessPre) / modelRoughnessPre < getThresholdIRWLSForLpOptimization() * 0.01)
				{
					OutputFiles::m_logFile << "# Reweighted iterative algorithm for Lp optimization is converged" << std::endl;
					break;
				}
				modelRoughnessPre = modelRoughness;
			}
			// Delete old out-of-core files
			m_ptrInversion->deleteOutOfCoreFileAll();
		}
		else
		{
			if (m_typeOfTradeOffParam == AnalysisControl::TO_Fixed)
			{
				// m_tradeOffParameterForResistivityValue = m_tradeOffParameterForResistivityValue;
				m_ptrInversion->inversionCalculation();
			}
			else if (m_typeOfTradeOffParam == AnalysisControl::TO_ABIC_LS)
			{
				if (myProcessID == 0)
				{
					std::cout << " # Entering ABIC ("
							  << abicConsoleRegularizationLabel(
								  useDifferenceFilter(),
								  m_MinNormInv && m_tradeOffParameterForMinNorm > CommonParameters::EPS)
							  << ")." << std::endl;
					std::cout << abicConsoleSearchModeLabel(m_abicSearchMode) << std::endl;
				}
				int numDataThisPE = ptrObservedData->getNumObservedDataThisPETotal();
				OutputFiles::m_logFile << "# Number of data of this PE : " << numDataThisPE << std::endl;
				if (m_residualVectorThisPE != NULL)
				{
					delete[] m_residualVectorThisPE;
					m_residualVectorThisPE = NULL;
				}
				m_residualVectorThisPE = new double[numDataThisPE];
				ptrObservedData->calculateResidualVectorOfDataThisPE(m_residualVectorThisPE); // d-F(m) updates
				m_abic = m_abicpre;
				m_stepLengthDampingFactorPre = m_stepLengthDampingFactorCur;
				iCutBack = 0;
				for (; iCutBack < m_numCutbackMax; iCutBack++)
				{
					m_stepLengthDampingFactorCur = (1.0 / pow(2.0, iCutBack)) * m_stepLengthDampingFactorPre;
					if (m_stepLengthDampingFactorCur < m_stepLengthDampingFactorMin)
					{
						OutputFiles::m_logFile << "# Model update is too small." << std::endl;
						std::cout << "# Model update is too small." << std::endl;
						m_leavingABIC = true;
						break;
					}
					if (shouldReuseInexactABICAlphaOnCutback(iCutBack))
					{
						if (myProcessID == 0)
						{
							std::cout << " # Reusing current inexact ABIC trade-off parameter with reduced step size." << std::endl;
							std::cout << "# Inexact ABIC cutback trial. alpha : "
									  << m_tradeOffParameterForResistivityValue
									  << ", step_length : " << m_stepLengthDampingFactorCur
									  << ", cutback_count : " << iCutBack << "." << std::endl;
						}
						OutputFiles::m_logFile << "# Reusing current inexact ABIC trade-off parameter with reduced step size." << std::endl;
						OutputFiles::m_logFile << "# Inexact ABIC cutback trial. alpha : "
										   << m_tradeOffParameterForResistivityValue
										   << ", step_length : " << m_stepLengthDampingFactorCur
										   << ", cutback_count : " << iCutBack << "." << std::endl;
						runReducedStepTrialWithCurrentABICAlpha();
						if (m_abic[1] < m_tolreq)
						{
							m_ABICconverage = true;
							break;
						}
						if (m_iterationNumCurrent > m_iterationNumInit)
						{
							if (m_abic[0] < m_abicpre[0] && m_abic[1] < m_abicpre[1])
							{
								break;
							}
							if (myProcessID == 0)
							{
								if (m_abicpre[1] <= m_abic[1])
								{
									std::cout << " # m_dataMisfitPre: " << m_abicpre[1] << "  <  " << "m_dataMisfitCur: " << m_abic[1] << std::endl;
								}
								if (m_abicpre[0] <= m_abic[0])
								{
									std::cout << " # m_abicPre: " << m_abicpre[0] << "  <  " << "m_abicCur: " << m_abic[0] << std::endl;
								}
								if (m_abicpre[1] <= m_abic[1] || m_abicpre[0] <= m_abic[0])
								{
									std::cout << " # Cutting the stepsize and re-testing current inexact alpha " << std::endl;
									std::cout << " # ...... " << std::endl;
								}
							}
							m_numConsecutiveIterFunctionalDecreasing = 0;
						}
						else
						{
							if (m_abic[1] < m_rmsPre)
							{
								break;
							}
							if (myProcessID == 0)
							{
								std::cout << " # m_dataMisfitPre: " << m_rmsPre << "  <  " << "m_dataMisfitCur: " << m_abic[1] << std::endl;
								std::cout << " # Cutting the stepsize and re-testing current inexact alpha " << std::endl;
								std::cout << " # ...... " << std::endl;
							}
							m_numConsecutiveIterFunctionalDecreasing = 0;
						}
						continue;
					}
					m_tradeOffParameterForResistivityValue = m_tradeOffParameterForResistivityValuePre;
					m_tradeOffParameterABICA = log10(m_tradeOffParameterForResistivityValue);
					m_tradeOffParameterABICB = m_tradeOffParameterABICA - getInitialABICLog10BracketSpan();
					if (myProcessID == 0)
					{
						if (shouldReportExactAbicBrentProgress(m_abicSearchMode))
						{
							std::cout << " # ...Bracketing Minimum..." << std::endl;
						}
					}
					minbrkABIC(); // m_ABICB[0] = min(abic); pwk1 is the corresponding model vector
					if (m_ABICB[1] < m_tolreq)
					{
						m_ABICconverage = true;
						m_abic = m_ABICB;
						m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterABICB);
						break;
					}
					else
					{
						if (myProcessID == 0)
						{
							if (shouldReportExactAbicBrentProgress(m_abicSearchMode))
							{
								std::cout << " # ...Finding minimum by Brent's minimizing method..." << std::endl;
							}
						}
						m_abic = fminbrentABIC(iCutBack); // pwk1 is the corresponding model vector

						if (myProcessID == 0)
						{
							if (shouldReportExactAbicBrentProgress(m_abicSearchMode))
							{
								std::cout << " # Minimum ABIC from fminbrent is at trade-off parameter = " << m_tradeOffParameterForResistivityValue << std::endl;
							}
							else
							{
								std::cout << " # Inexact Minimum ABIC is at trade-off parameter = " << m_tradeOffParameterForResistivityValue << std::endl;
							}
						}
#ifdef FEMTIC_DABIC_TEST_FORCE_FIRST_INEXACT_ABIC_CUTBACK
						if (usesInexactABICSearch() && iCutBack == 0)
						{
							if (myProcessID == 0)
							{
								std::cout << "# TEST HOOK: forcing first inexact ABIC trial rejection." << std::endl;
							}
							OutputFiles::m_logFile << "# TEST HOOK: forcing first inexact ABIC trial rejection." << std::endl;
							continue;
						}
#endif
						if (m_abic[1] < m_tolreq)
						{
							m_ABICconverage = true;
							break;
						}

						if (m_iterationNumCurrent > m_iterationNumInit)
						{
							if (m_abic[0] < m_abicpre[0] && m_abic[1] < m_abicpre[1])
							{
								break;
							}
							else
							{
								if (myProcessID == 0)
								{
									if (m_abicpre[1] <= m_abic[1])
									{
										std::cout << " # m_dataMisfitPre: " << m_abicpre[1] << "  <  " << "m_dataMisfitCur: " << m_abic[1] << std::endl;
									}
									if (m_abicpre[0] <= m_abic[0])
									{
										std::cout << " # m_abicPre: " << m_abicpre[0] << "  <  " << "m_abicCur: " << m_abic[0] << std::endl;
									}
									if (m_abicpre[1] <= m_abic[1] || m_abicpre[0] <= m_abic[0])
									{
										std::cout << " # Cutting the stepsize and re-searching " << std::endl;
										std::cout << " # ...... " << std::endl;
									}
								}
								m_numConsecutiveIterFunctionalDecreasing = 0; // reset value
							}
						}
						else
						{
							if (m_abic[1] < m_rmsPre)
							{
								break;
							}
							else
							{
								if (myProcessID == 0)
								{
									std::cout << " # m_dataMisfitPre: " << m_rmsPre << "  <  " << "m_dataMisfitCur: " << m_abic[1] << std::endl;
									std::cout << " # Cutting the stepsize and re-searching " << std::endl;
									std::cout << " # ...... " << std::endl;
								}
								m_numConsecutiveIterFunctionalDecreasing = 0; // reset value
							}
						}
					}
				}

				if (iCutBack == m_numCutbackMax)
				{
					OutputFiles::m_logFile << "# Reach maximum retrial number." << std::endl;
					if (myProcessID == 0)
					{
						std::cout << "# Reach maximum retrial number." << std::endl;
					}
					m_leavingABIC = true;
				}

				if (m_ABICconverage)
				{
					// TOLERANCE IS BELOW THAT REQUIRED; FIND INTERCEPT.
					if (myProcessID == 0)
					{
						std::cout << " # Finding Intercept: bracketing the root (RMS - m_tolreq = 0)..." << std::endl;
					}
					m_stepsizelb = m_stepLengthDampingFactorCur;
					m_ABIClb = m_abic;
					m_stepsizeub = m_stepLengthDampingFactorCur;
					m_ABICub = m_abic;
					int count = 0;
					m_stepLengthDampingFactorPre = m_stepLengthDampingFactorCur;
					while (m_ABICub[1] < m_tolreq)
					{
						m_ABIClb = m_ABICub;
						m_stepsizelb = m_stepsizeub;
						if (count > 0)
						{
							ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
							ptrObservedData->copyDistortionParamsCurToPWK1();
						}
						m_stepLengthDampingFactorCur = (1.0 / pow(2.0, count)) * m_stepLengthDampingFactorPre;
						m_stepsizeub = m_stepLengthDampingFactorCur;
						m_ptrInversion->inversionCalculation();
						m_ABICub = m_ptrInversion->getabic();
						count += +1;
					} // after this loop, m_rmsOCCub >= m_tolreq; m_tradeOffParameterOCCub > m_tradeOffParameterOCClb.
					if (myProcessID == 0)
					{
						std::cout << " # Finding Intercept: approaching the root (RMS - m_tolreq = 0)..." << std::endl;
					}
					ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
					ptrObservedData->copyDistortionParamsCurToPWK2();
					m_stepLengthDampingFactorCur = pow(10.0, frootABIC());
					if (myProcessID == 0)
					{
						std::cout << " # Tolerance is met (approximately) at trade-off parameter  = " << m_tradeOffParameterForResistivityValue << std::endl;
					}
					ptrResistivityBlock->copyPWK2NotFixedToPWK1();
					ptrObservedData->copyDistortionParamsPWK2ToPWK1();
				}
				ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
				ptrObservedData->copyDistortionParamsPWK1ToCur();
				if (ptrObservedData->hasSelectedTrialForwardResponseCache())
				{
					ptrObservedData->restoreSelectedTrialForwardResponseCache();
				}
			}
			else if (m_typeOfTradeOffParam == AnalysisControl::TO_OCCAM_LS)
			{
				runOCCAMLineSearch("Laplacian Filter");
			}
			else if (m_typeOfTradeOffParam == AnalysisControl::TO_LINEAR_LCURVE)
			{
				if (m_ptrInversiondataspace == NULL)
				{
					OutputFiles::m_logFile << "Error : L-curve final data-space inversion object is NULL." << std::endl;
					exit(1);
				}
				if (myProcessID == 0)
				{
					std::cout << " # Entering linear cubic-spline L-curve selection (Laplacian Filter)." << std::endl;
				}
				m_ptrInversion->inversionCalculation();
				if (myProcessID == 0)
				{
					const double selectedTradeOff = m_ptrInversion->getAlphawithmaxc();
					std::cout << " # Linear L-curve selected trade-off parameter : " << selectedTradeOff << std::endl;
					m_tradeOffParameterForResistivityValue = selectedTradeOff;
					setLCurveFinalTradeOffParameterForDiagnostics(m_tradeOffParameterForResistivityValue);
					std::cout << " # Final linear L-curve trade-off parameter : " << m_tradeOffParameterForResistivityValue << std::endl;
				}
				MPI_Bcast(&m_tradeOffParameterForResistivityValue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				m_ptrInversiondataspace->inversionCalculation();
			}
			else if (m_typeOfTradeOffParam == AnalysisControl::TO_NONLINEAR_LCURVE)
			{
				runNonlinearLCurveDiagnostics(m_iterationNumCurrent, "Laplacian Filter");
				if (!m_stopAfterNonlinearLCurveDiagnostics)
				{
					m_ptrInversiondataspace->inversionCalculation();
				}
			}
			else if (m_typeOfTradeOffParam == AnalysisControl::TO_DATA_FIT_COOLING)
			{
				if (!m_dataFitCoolingHasSelectedAlpha)
				{
					m_stopAfterDataFitCooling = !runInitialDataFitCoolingBracket();
				}
				else
				{
					m_stopAfterDataFitCooling = !runPersistentDataFitCoolingAlpha();
				}
			}
		}
		if (m_stopAfterNonlinearLCurveDiagnostics || m_stopAfterDataFitCooling)
		{
			OutputFiles::m_logFile << (m_stopAfterDataFitCooling
				? "# Stop inversion loop because data-fit cooling found no acceptable full-step alpha."
				: "# Stop inversion loop after nonlinear L-curve diagnostics.") << std::endl;
			break;
		}
		m_tradeOffParameterForResistivityValuePre = m_tradeOffParameterForResistivityValue;
	}

	if (myProcessID == 0)
	{
		time_t endTime;
		time(&endTime);
		const FemticDabicRunSummary::FinishSummary finishSummary = {
			m_startTime,
			endTime,
			m_totalPE,
			m_numThreads
		};
		FemticDabicRunSummary::outputFinishConsole(std::cout, finishSummary);
	}

	if (m_ptrForward3DBrickElement0thOrder != NULL)
	{
		delete m_ptrForward3DBrickElement0thOrder;
		m_ptrForward3DBrickElement0thOrder = NULL;
	}

	if (m_ptrForward3DTetraElement0thOrder != NULL)
	{
		delete m_ptrForward3DTetraElement0thOrder;
		m_ptrForward3DTetraElement0thOrder = NULL;
	}

	if (m_ptrInversion != NULL)
	{
		m_ptrInversion->deleteOutOfCoreFileAll();
		delete m_ptrInversion;
		m_ptrInversion = NULL;
	}

	OutputFiles::m_logFile << "# End " << CommonParameters::programName << " " << outputElapsedTime() << std::endl;
}

// Read analysis control data from "control.dat"
void AnalysisControl::inputControlData()
{

	std::ifstream inFile("control.dat", std::ios::in);
	if (inFile.fail())
	{
		// std::cerr << "File open error : control.dat !!" << std::endl;
		OutputFiles::m_logFile << "File open error : control.dat !!" << std::endl;
		exit(1);
	}

	// Flag specifing whether each parameter has already read from control.dat
	bool hasAlreadyRead[numParamWrittenInControlFile];
	ControlKeywords::resetReadFlags(hasAlreadyRead, numParamWrittenInControlFile);

	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();

	while (!inFile.eof())
	{
		std::string line;
		inFile >> line;

#ifdef _DEBUG_WRITE
		std::cout << "line : " << line << std::endl;
#endif

		double dbuf(0.0);
		int ibuf(0);
		if (line.substr(0, 25).compare("BOUNDARY_CONDITION_BOTTOM") == 0)
		{ // Read the type of boundary condition at the bottom of the model
			const int paramID = AnalysisControl::BOUNDARY_CONDITION_BOTTOM;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "BOUNDARY_CONDITION_BOTTOM");
			inFile >> ibuf;
			if (ibuf != AnalysisControl::BOUNDARY_BOTTOM_ONE_DIMENSIONAL &&
				ibuf != AnalysisControl::BOUNDARY_BOTTOM_PERFECT_CONDUCTOR)
			{
				OutputFiles::m_logFile << "Error : Wrong type of boundary condition at the bottom of the model !! " << ibuf << "." << std::endl;
				exit(1);
			}
			m_boundaryConditionBottom = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 9).compare("MESH_TYPE") == 0)
		{ // Type of mesh
			const int paramID = AnalysisControl::MESH_TYPE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "MESH_TYPE");
			inFile >> ibuf;
			if (ibuf != MeshData::HEXA && ibuf != MeshData::TETRA && ibuf != MeshData::NONCONFORMING_HEXA)
			{
				OutputFiles::m_logFile << "Error : The number following MESH_TYPE must be " << MeshData::HEXA << ", " << MeshData::TETRA << " or " << MeshData::NONCONFORMING_HEXA << " !!" << std::endl;
				exit(1);
			}
			m_typeOfMesh = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 11).compare("NUM_THREADS") == 0)
		{ // Read total number of threads
			const int paramID = AnalysisControl::NUM_THREADS;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "NUM_THREADS");
			inFile >> ibuf;
			if (ibuf < 0)
			{
				OutputFiles::m_logFile << "Error : Number of threads must be greater than or equals to 1 !! " << std::endl;
				exit(1);
			}
			m_numThreads = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 10).compare("FWD_SOLVER") == 0)
		{
			const int paramID = AnalysisControl::FWD_SOLVER;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "FWD_SOLVER");
			inFile >> ibuf;
			if (ibuf != PARDISOSolver::INCORE_MODE && ibuf != PARDISOSolver::SELECT_MODE_AUTOMATICALLY && ibuf != PARDISOSolver::OUT_OF_CORE_MODE)
			{
				OutputFiles::m_logFile << "Error : Parameter specifing the mode of forward solver must be 0, 1 or 2 !! " << ibuf << "." << std::endl;
				exit(1);
			}
			m_modeOfPARDISO = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 9).compare("MEM_LIMIT") == 0)
		{
			const int paramID = AnalysisControl::MEM_LIMIT;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "MEM_LIMIT");
			inFile >> dbuf;
			m_maxMemoryPARDISO = static_cast<int>(dbuf);
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 16).compare("NUMBERING_METHOD") == 0)
		{
			const int paramID = AnalysisControl::NUMBERING_METHOD;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "NUMBERING_METHOD");
			inFile >> ibuf;
			if (ibuf != AnalysisControl::NOT_ASSIGNED && ibuf != AnalysisControl::XYZ && ibuf != AnalysisControl::YZX && ibuf != AnalysisControl::ZXY)
			{
				OutputFiles::m_logFile << "Error : Number of parameter specifing the way numbering must be -1, 0, 1 or 2 !!" << std::endl;
				exit(1);
			}
			m_numberingMethod = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 12).compare("OUTPUT_PARAM") == 0)
		{
			const int paramID = AnalysisControl::OUTPUT_PARAM;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "OUTPUT_PARAM");
			int num(0);
			inFile >> num;
			if (num < 0)
			{
				OutputFiles::m_logFile << "Error : Number of parameter to be outputed to VTK is less than 0 !!" << std::endl;
				exit(1);
			}
			else if (num > 0)
			{
				for (int i = 0; i < num; ++i)
				{
					inFile >> ibuf;
					m_outputParametersForVis.insert(ibuf);
				}
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 16).compare("OUTPUT_OPTION") == 0)
		{
			const int paramID = AnalysisControl::OUTPUT_OPTION;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "OUTPUT_OPTION");
			int ibufX(0);
			int ibufY(0);
			inFile >> ibufX >> ibufY;
			if (ibufX == 0)
			{
				m_useBackwardOrForwardElement.directionX = AnalysisControl::BACKWARD_ELEMENT;
			}
			else
			{
				m_useBackwardOrForwardElement.directionX = AnalysisControl::FORWARD_ELEMENT;
			}
			if (ibufY == 0)
			{
				m_useBackwardOrForwardElement.directionY = AnalysisControl::BACKWARD_ELEMENT;
			}
			else
			{
				m_useBackwardOrForwardElement.directionY = AnalysisControl::FORWARD_ELEMENT;
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 17).compare("OUTPUT_2D_RESULTS") == 0)
		{
			const int paramID = AnalysisControl::OUTPUT_2D_RESULTS;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "OUTPUT_2D_RESULTS");
			m_isOutput2DResult = true;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 20).compare("WEIGHT_OF_DISTORTION") == 0)
		{

			if (!hasAlreadyRead[AnalysisControl::DISTORTION])
			{
				OutputFiles::m_logFile << "Error : You must write DISTORTION data above WEIGHT_OF_DISTORTION" << std::endl;
				exit(1);
			}

			const int paramID = AnalysisControl::WEIGHT_OF_DISTORTION;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "WEIGHT_OF_DISTORTION");

			switch (m_typeOfDistortion)
			{
			case AnalysisControl::NO_DISTORTION:
				break;
			case AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE:
				inFile >> m_tradeOffParameterForDistortionMatrixComplexity;
				break;
			case AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS:
				inFile >> m_tradeOffParameterForDistortionGain >> m_tradeOffParameterForDistortionRotation;
				break;
			case AnalysisControl::ESTIMATE_GAINS_ONLY:
				inFile >> m_tradeOffParameterForDistortionGain;
				break;
			default:
				OutputFiles::m_logFile << "Error : Wrong type of distortion : " << ibuf << std::endl;
				exit(1);
				break;
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 9).compare("ITERATION") == 0)
		{
			const int paramID = AnalysisControl::ITERATION;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "ITERATION");
			inFile >> m_iterationNumInit >> m_iterationNumMax;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 18).compare("DECREASE_THRESHOLD") == 0)
		{
			const int paramID = AnalysisControl::DECREASE_THRESHOLD;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "DECREASE_THRESHOLD");
			inFile >> m_thresholdValueForDecreasing;
			if (m_thresholdValueForDecreasing < 0)
			{
				OutputFiles::m_logFile << "Error : Threshold value for determining if objective functional decrease must be positive." << std::endl;
				exit(1);
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 8).compare("CONVERGE") == 0)
		{
			const int paramID = AnalysisControl::CONVERGE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "CONVERGE");
			inFile >> m_decreaseRatioForConvegence;
			if (m_decreaseRatioForConvegence < 0)
			{
				OutputFiles::m_logFile << "Error : Criterion for convergence must be positive." << std::endl;
				exit(1);
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 7).compare("RETRIAL") == 0)
		{
			const int paramID = AnalysisControl::RETRIAL;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "RETRIAL");
			inFile >> m_numCutbackMax;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 11).compare("STEP_LENGTH") == 0)
		{
			const int paramID = AnalysisControl::STEP_LENGTH;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "STEP_LENGTH");
			inFile >> m_stepLengthDampingFactorCur >> m_stepLengthDampingFactorMin >> m_stepLengthDampingFactorMax >> m_numOfIterIncreaseStepLength >> m_factorDecreasingStepLength >> m_factorIncreasingStepLength;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 10).compare("DISTORTION") == 0)
		{
			const int paramID = AnalysisControl::DISTORTION;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "DISTORTION");
			inFile >> ibuf;

			if (ibuf != AnalysisControl::NO_DISTORTION &&
				ibuf != AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE &&
				ibuf != AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS &&
				ibuf != AnalysisControl::ESTIMATE_GAINS_ONLY)
			{
				OutputFiles::m_logFile << "Error : Wrong type ID is specified below DISTORTION : " << ibuf << std::endl;
				exit(1);
				break;
			}
			m_typeOfDistortion = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 27).compare("TYPE_OF_TRADE_OFF_PARAMETER") == 0)
		{
			const int paramID = AnalysisControl::TYPE_OF_TRADE_OFF_PARAMETER;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "TYPE_OF_TRADE_OFF_PARAMETER");
			inFile >> ibuf;

			if (!isKnownTradeOffParameterMode(ibuf))
			{
				OutputFiles::m_logFile << "Error : Wrong type ID is specified below TYPE_OF_TRADE_OFF_PARAMETER : " << ibuf << std::endl;
				exit(1);
				break;
			}
			if (!isImplementedTradeOffParameterMode(ibuf))
			{
				OutputFiles::m_logFile
					<< "Error : TYPE_OF_TRADE_OFF_PARAMETER " << ibuf
					<< " (" << tradeOffParameterLabel(ibuf)
					<< ") is recognized but not enabled in this maintained FEMTIC-DABIC branch yet."
					<< " Supported values are 0 (fixed trade-off parameter), 1 (ABIC line search),"
					<< " 2 (OCCAM line search), 3 (linear cubic-spline L-curve selection),"
					<< " 4 (nonlinear cubic-spline L-curve selection),"
					<< " and 5 (data-fit-bracketed cooling)."
					<< std::endl;
				exit(1);
				break;
			}
			m_typeOfTradeOffParam = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 15).compare("TRADE_OFF_PARAM") == 0)
		{

			if (!hasAlreadyRead[AnalysisControl::TYPE_OF_TRADE_OFF_PARAMETER])
			{
				OutputFiles::m_logFile << "Error : You must write TYPE_OF_TRADE_OFF_PARAMETER data above TRADE_OFF_PARAM" << std::endl;
				exit(1);
			}
			const int paramID = AnalysisControl::TRADE_OFF_PARAM;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "TRADE_OFF_PARAM");
			switch (m_typeOfTradeOffParam)
			{
			case AnalysisControl::TO_Fixed:
				inFile >> m_tradeOffParameterForResistivityValue;
				break;
			case AnalysisControl::TO_ABIC_LS:
				inFile >> m_tradeOffParameterForResistivityValue >> m_tolreq;
				m_tradeOffParameterForResistivityValuePre = m_tradeOffParameterForResistivityValue;
				break;
			case AnalysisControl::TO_OCCAM_LS:
				inFile >> m_tradeOffParameterForResistivityValue >> m_tolreq;
				m_tradeOffParameterForResistivityValuePre = m_tradeOffParameterForResistivityValue;
				break;
			case AnalysisControl::TO_DATA_FIT_COOLING:
				inFile >> m_dataFitCoolingInitialAlpha >> m_tolreq;
				if (m_dataFitCoolingInitialAlpha <= 0.0)
				{
					OutputFiles::m_logFile << "Error : Data-fit cooling initial alpha must be positive." << std::endl;
					exit(1);
				}
				if (m_tolreq <= 0.0)
				{
					OutputFiles::m_logFile << "Error : Data-fit cooling target RMS must be positive." << std::endl;
					exit(1);
				}
				m_tradeOffParameterForResistivityValue = m_dataFitCoolingInitialAlpha;
				m_tradeOffParameterForResistivityValuePre = m_dataFitCoolingInitialAlpha;
				break;
			case AnalysisControl::TO_LINEAR_LCURVE:
			{
				int useLogLog = 0;
				int useRootNorm = 0;
				inFile >> m_NumOF_TO >> useLogLog >> useRootNorm;
				if (m_NumOF_TO <= 0)
				{
					OutputFiles::m_logFile << "Error : Number of L-curve trade-off parameters must be positive : " << m_NumOF_TO << std::endl;
					exit(1);
				}
				if ((useLogLog != 0 && useLogLog != 1) || (useRootNorm != 0 && useRootNorm != 1))
				{
					OutputFiles::m_logFile << "Error : L-curve loglog and norm flags must be 0 or 1." << std::endl;
					exit(1);
				}
				m_lCurveUseLogLog = (useLogLog == 1);
				m_lCurveUseRootNorm = (useRootNorm == 1);
				if (m_tradeOffParameters != NULL)
				{
					delete[] m_tradeOffParameters;
					m_tradeOffParameters = NULL;
				}
				m_tradeOffParameters = new double[m_NumOF_TO];
				for (int i = 0; i < m_NumOF_TO; ++i)
				{
					inFile >> m_tradeOffParameters[i];
					if (m_tradeOffParameters[i] <= 0.0)
					{
						OutputFiles::m_logFile << "Error : L-curve trade-off parameter must be positive : " << m_tradeOffParameters[i] << std::endl;
						exit(1);
					}
				}
				m_tradeOffParameterForResistivityValue = m_tradeOffParameters[0];
				m_tradeOffParameterForResistivityValuePre = m_tradeOffParameterForResistivityValue;
				break;
			}
			case AnalysisControl::TO_NONLINEAR_LCURVE:
			{
				const double alphaLowerBound = 0.1;
				const double alphaUpperBound = 100.0;
				double startAlpha = 0.0;
				inFile >> startAlpha;
				if (startAlpha <= 0.0)
				{
					OutputFiles::m_logFile << "Error : Nonlinear L-curve start trade-off parameter must be positive : " << startAlpha << std::endl;
					exit(1);
				}
				if (startAlpha < alphaLowerBound || startAlpha > alphaUpperBound)
				{
					OutputFiles::m_logFile
						<< "Error : Nonlinear L-curve start trade-off parameter must be within ["
						<< alphaLowerBound << ", " << alphaUpperBound << "] : "
						<< startAlpha << std::endl;
					exit(1);
				}
				std::vector<double> nonlinearAlphas;
				for (double alpha = startAlpha;
					alpha >= alphaLowerBound * (1.0 - 1.0e-10);
					alpha /= std::sqrt(10.0))
				{
					nonlinearAlphas.push_back(std::max(alpha, alphaLowerBound));
					if (alpha <= alphaLowerBound * (1.0 + 1.0e-10))
					{
						break;
					}
				}
				if (nonlinearAlphas.empty() ||
					nonlinearAlphas.back() > alphaLowerBound * (1.0 + 1.0e-8))
				{
					nonlinearAlphas.push_back(alphaLowerBound);
				}
				m_NumOF_TO = static_cast<int>(nonlinearAlphas.size());
				m_lCurveUseLogLog = true;
				m_lCurveUseRootNorm = true;
				if (m_tradeOffParameters != NULL)
				{
					delete[] m_tradeOffParameters;
					m_tradeOffParameters = NULL;
				}
				m_tradeOffParameters = new double[m_NumOF_TO];
				for (int i = 0; i < m_NumOF_TO; ++i)
				{
					m_tradeOffParameters[i] = nonlinearAlphas[static_cast<std::vector<double>::size_type>(i)];
				}
				m_tradeOffParameterForResistivityValue = m_tradeOffParameters[0];
				m_tradeOffParameterForResistivityValuePre = m_tradeOffParameterForResistivityValue;
				break;
			}
			default:
				OutputFiles::m_logFile << "Error : Wrong type of parameter selection scheme : " << ibuf << std::endl;
				exit(1);
				break;
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 10).compare("TYPE_OF_CG") == 0)
		{
			const int paramID = AnalysisControl::TYPE_OF_CG;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "TYPEOF_TO");
			inFile >> ibuf;

			if (ibuf != AnalysisControl::FD_CG &&
				ibuf != AnalysisControl::CD_CG &&
				ibuf != AnalysisControl::MS_CG)
			{
				OutputFiles::m_logFile << "Error : Wrong type ID is specified below TYPE_OF_CG : " << ibuf << std::endl;
				exit(1);
				break;
			}
			m_typeOfCG = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 12).compare("TRADE_OFF_CG") == 0)
		{
			if (!hasAlreadyRead[AnalysisControl::TYPE_OF_CG])
			{
				OutputFiles::m_logFile << "Error : You must write TYPE_OF_CG data above TRADE_OFF_CG" << std::endl;
				exit(1);
			}
			const int paramID = AnalysisControl::TRADE_OFF_CG;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "TRADE_OFF_CG");
			switch (m_typeOfCG)
			{
			case AnalysisControl::FD_CG:
				inFile >> m_tradeOffParameterForCrossGradient;
				break;
			case AnalysisControl::CD_CG:
				inFile >> m_tradeOffParameterForCrossGradient;
				break;
			case AnalysisControl::MS_CG:
				inFile >> m_tradeOffParameterForCrossGradient;
				inFile >> m_smallvalueForCrossGradient;
				break;
			default:
				OutputFiles::m_logFile << "Error : Wrong type of Cross-Gradient operator : " << ibuf << std::endl;
				exit(1);
				break;
			}
			m_CrossGradientInv = true;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 17).compare("TYPE_OF_REFERENCE") == 0)
		{
			const int paramID = AnalysisControl::TYPE_OF_REFERENCE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "REFERENCE_MOD");
			inFile >> m_typeOfReferenceModel;

			if (m_typeOfReferenceModel < 0)
			{
				OutputFiles::m_logFile << "Error : 	m_typeOfReferenceModel must be an integer >=0 " << std::endl;
				exit(1);
				break;
			}
			else
			{
				hasAlreadyRead[paramID] = true;
			}
		}
		else if (line.substr(0, 19).compare("WEIGHT_OF_REFERENCE") == 0)
		{
			if (!hasAlreadyRead[AnalysisControl::TYPE_OF_REFERENCE])
			{
				OutputFiles::m_logFile << "Error : You must write TYPE_OF_CG data above TRADE_OFF_CG" << std::endl;
				exit(1);
			}
			const int paramID = AnalysisControl::WEIGHT_OF_REFERENCE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "REFERENCE_MOD");
			inFile >> m_tradeOffParameterForMinNorm;

			if (m_tradeOffParameterForMinNorm < 0.0)
			{
				OutputFiles::m_logFile << "Error : 	m_tradeOffParameterForMinNorm must >= 0.0 : " << std::endl;
				exit(1);
				break;
			}
			else
			{
				m_MinNormInv = true;
				hasAlreadyRead[paramID] = true;
			}
		}
		else if (line.substr(0, 19).compare("NORM_OF_MINIMUMNORM") == 0)
		{
			if (!hasAlreadyRead[AnalysisControl::TYPE_OF_REFERENCE])
			{
				OutputFiles::m_logFile << "Error : You must write TYPE_OF_REFERENCE data above NORM_OF_MINIMUMNORM" << std::endl;
				exit(1);
			}
			const int paramID = AnalysisControl::NORM_OF_MINIMUMNORM;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "NORM_OF_MINIMUMNORM");
			inFile >> ibuf;
			m_degreeOfLpMinimumNorm = ibuf;
			if (m_degreeOfLpMinimumNorm == 0)
			{
				inFile >> dbuf;
				m_smallvauleOfMinimumSupport = dbuf;
			}
			else if (m_degreeOfLpMinimumNorm == 1 || m_degreeOfLpMinimumNorm == 2)
			{
				inFile >> dbuf;
				m_lowerLimitOfDifflog10RhoForLpMinimumNorm = dbuf;
				inFile >> dbuf;
				m_upperLimitOfDifflog10RhoForLpMinimumNorm = dbuf;
			}
			m_MinNormInv = true;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 12).compare("ROUGH_MATRIX") == 0)
		{
			const int paramID = AnalysisControl::ROUGH_MATRIX;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "ROUGH_MATRIX");
			inFile >> ibuf;
			if (ibuf >= AnalysisControl::EndOfTypeOfRoughningMatrix)
			{
				OutputFiles::m_logFile << "Error : Inputted parameter specifing the way of creating roughning matrix is wrong !! : " << ibuf << std::endl;
				exit(1);
			}
			else
			{
				m_typeOfRoughningMatrix = ibuf;
			}

			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 10).compare("ELEC_FIELD") == 0)
		{
			const int paramID = AnalysisControl::ELEC_FIELD;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "ELEC_FIELD");
			inFile >> ibuf;
			if (ibuf < 0)
			{
				m_isTypeOfElectricFieldSetIndivisually = true;
			}
			else if (ibuf != AnalysisControl::USE_HORIZONTAL_ELECTRIC_FIELD &&
					 ibuf != AnalysisControl::USE_TANGENTIAL_ELECTRIC_FIELD)
			{
				OutputFiles::m_logFile << "Error : Unknown type of the electric field is specified in ELEC_FIELD : " << ibuf << std::endl;
				exit(1);
			}
			m_typeOfElectricField = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 15).compare("DIV_NUM_RHS_FWD") == 0)
		{
			const int paramID = AnalysisControl::DIV_NUM_RHS_FWD;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "DIV_NUM_RHS_FWD");
			inFile >> m_divisionNumberOfMultipleRHSInForward;
			if (m_divisionNumberOfMultipleRHSInForward < 1)
			{
				OutputFiles::m_logFile << "Error : Division number of right-hand sides must be greater than zero !! Specified number is " << m_divisionNumberOfMultipleRHSInForward << "." << std::endl;
				exit(1);
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 15).compare("DIV_NUM_RHS_INV") == 0)
		{
			const int paramID = AnalysisControl::DIV_NUM_RHS_INV;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "DIV_NUM_RHS_INV");
			inFile >> m_divisionNumberOfMultipleRHSInInversion;
			if (m_divisionNumberOfMultipleRHSInInversion < 1)
			{
				OutputFiles::m_logFile << "Error : Division number of right-hand sides must be greater than zero !! Specified number is " << m_divisionNumberOfMultipleRHSInInversion << "." << std::endl;
				exit(1);
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 18).compare("RESISTIVITY_BOUNDS") == 0)
		{
			const int paramID = AnalysisControl::RESISTIVITY_BOUNDS;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "RESISTIVITY_BOUNDS");
			inFile >> ibuf;
			ptrResistivityBlock->setTypeBoundConstraints(ibuf);
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 10).compare("OFILE_TYPE") == 0)
		{
			const int paramID = AnalysisControl::OFILE_TYPE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "OFILE_TYPE");
			inFile >> ibuf;
			if (ibuf == 0)
			{ // ASCII format
				m_binaryOutput = false;
			}
			else
			{ // Binary format
				m_binaryOutput = true;
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 12).compare("HOLD_FWD_MEM") == 0)
		{
			const int paramID = AnalysisControl::HOLD_FWD_MEM;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "HOLD_FWD_MEM");
			m_holdMemoryForwardSolver = true;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 12).compare("ALPHA_WEIGHT") == 0)
		{
			const int paramID = AnalysisControl::ALPHA_WEIGHT;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "ALPHA_WEIGHT");
			for (int iDir = 0; iDir < 3; ++iDir)
			{
				double dbuf(0.0);
				inFile >> dbuf;
				if (dbuf < 0.0)
				{
					OutputFiles::m_logFile << "Error : Weighting factor of alpha must be positive !! : " << dbuf << std::endl;
					exit(1);
				}
				m_alphaWeight[iDir] = dbuf;
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 25).compare("INV_MAT_POSITIVE_DEFINITE") == 0)
		{
			const int paramID = AnalysisControl::INV_MAT_POSITIVE_DEFINITE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "INV_MAT_POSITIVE_DEFINITE");
			m_positiveDefiniteNormalEqMatrix = true;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 18).compare("BOTTOM_RESISTIVITY") == 0)
		{
			const int paramID = AnalysisControl::BOTTOM_RESISTIVITY;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "BOTTOM_RESISTIVITY");
			ptrResistivityBlock->setFlagIncludeBottomResistivity(true);
			double dbuf(0.0);
			inFile >> dbuf;
			if (dbuf < 0.0)
			{
				OutputFiles::m_logFile << "Error : Bottom resistivity is set to be negative !! : " << dbuf << std::endl;
				exit(1);
			}
			ptrResistivityBlock->setBottomResistivity(dbuf);
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 23).compare("BOTTOM_ROUGHNING_FACTOR") == 0)
		{
			const int paramID = AnalysisControl::BOTTOM_ROUGHNING_FACTOR;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "BOTTOM_ROUGHNING_FACTOR");
			inFile >> dbuf;
			if (dbuf < 0.0)
			{
				OutputFiles::m_logFile << "Error : Roughning factor at bottom is set to be negative !! : " << dbuf << std::endl;
				exit(1);
			}
			ptrResistivityBlock->setRoughningFactorAtBottom(dbuf);
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 10).compare("INV_METHOD") == 0)
		{
			const int paramID = AnalysisControl::INV_METHOD;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "INV_METHOD");
			inFile >> m_inversionMethod;
			if (m_inversionMethod != Inversion::GAUSS_NEWTON_DATA_SPECE &&
				m_inversionMethod != Inversion::GAUSS_NEWTON_MODEL_SPECE &&
				m_inversionMethod != Inversion::ABIC_DATA_SPECE &&
				m_inversionMethod != Inversion::OCCAM_DATA_SPECE &&
				m_inversionMethod != Inversion::LINEAR_LCURVE_DATA_SPECE &&
				m_inversionMethod != Inversion::NONLINEAR_LCURVE_DATA_SPECE &&
				m_inversionMethod != Inversion::DATA_FIT_COOLING_DATA_SPECE)
			{
				// Code block
				OutputFiles::m_logFile << "Error : Type of inversion method is wrong  !! : " << m_inversionMethod << std::endl;
				exit(1);
			}
			if (m_inversionMethod == Inversion::ABIC_DATA_SPECE)
			{
				m_ABICinversion = true;
			}
			if (m_inversionMethod == Inversion::OCCAM_DATA_SPECE)
			{
				m_OCCAMinversion = true;
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 23).compare("RUN_INEXACT_LINE_SEARCH") == 0)
		{
			if (!hasAlreadyRead[AnalysisControl::INV_METHOD])
			{
				OutputFiles::m_logFile << "Error : You must write INV_METHOD data above RUN_INEXACT_LINE_SEARCH" << std::endl;
				exit(1);
			}
			if (!m_ABICinversion)
			{
				OutputFiles::m_logFile << "Error : You must select ABIC inversion while define RUN_INEXACT_LINE_SEARCH" << std::endl;
				exit(1);
			}
			if (hasAlreadyRead[AnalysisControl::ABIC_SEARCH_MODE])
			{
				OutputFiles::m_logFile
					<< "Error : ABIC_SEARCH_MODE conflicts with legacy RUN_INEXACT_LINE_SEARCH."
					<< std::endl;
				exit(1);
			}
			const int paramID = AnalysisControl::RUN_INEXACT_LINE_SEARCH;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "RUN_INEXACT_LINE_SEARCH");
			int enable = 0;
			int legacyMode = 0;
			inFile >> enable >> legacyMode;
			if (enable != 1 || legacyMode != 2)
			{
				OutputFiles::m_logFile << "Error : Legacy ABIC mode " << legacyMode
					<< " is no longer supported. Use ABIC_SEARCH_MODE EXACT or INEXACT."
					<< std::endl;
				exit(1);
			}
			m_abicSearchMode = ABIC_SEARCH_INEXACT;
			OutputFiles::m_logFile
				<< "# Deprecated: Legacy RUN_INEXACT_LINE_SEARCH 1 2 maps to ABIC_SEARCH_MODE INEXACT."
				<< std::endl;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 16).compare("ABIC_SEARCH_MODE") == 0)
		{
			if (!hasAlreadyRead[AnalysisControl::INV_METHOD])
			{
				OutputFiles::m_logFile
					<< "Error : You must write INV_METHOD above ABIC_SEARCH_MODE."
					<< std::endl;
				exit(1);
			}
			if (!m_ABICinversion)
			{
				OutputFiles::m_logFile
					<< "Error : ABIC_SEARCH_MODE is available only for ABIC inversion."
					<< std::endl;
				exit(1);
			}
			if (hasAlreadyRead[AnalysisControl::RUN_INEXACT_LINE_SEARCH])
			{
				OutputFiles::m_logFile
					<< "Error : ABIC_SEARCH_MODE conflicts with legacy RUN_INEXACT_LINE_SEARCH."
					<< std::endl;
				exit(1);
			}
			const int paramID = AnalysisControl::ABIC_SEARCH_MODE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "ABIC_SEARCH_MODE");
			std::string mode;
			inFile >> mode;
			if (mode == "EXACT")
			{
				m_abicSearchMode = ABIC_SEARCH_EXACT;
				OutputFiles::m_logFile << "# ABIC search mode: exact" << std::endl;
			}
			else if (mode == "INEXACT")
			{
				m_abicSearchMode = ABIC_SEARCH_INEXACT;
				OutputFiles::m_logFile << "# ABIC search mode: inexact" << std::endl;
			}
			else
			{
				OutputFiles::m_logFile
					<< "Error : ABIC_SEARCH_MODE must be EXACT or INEXACT."
					<< std::endl;
				exit(1);
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 29).compare("RUN_INEXACT_OCCAM_LINE_SEARCH") == 0)
		{
			if (!hasAlreadyRead[AnalysisControl::INV_METHOD])
			{
				OutputFiles::m_logFile << "Error : You must write INV_METHOD data above RUN_INEXACT_OCCAM_LINE_SEARCH" << std::endl;
				exit(1);
			}
			if (!m_OCCAMinversion)
			{
				OutputFiles::m_logFile << "Error : You must select OCCAM inversion while defining RUN_INEXACT_OCCAM_LINE_SEARCH" << std::endl;
				exit(1);
			}
			const int paramID = AnalysisControl::RUN_INEXACT_OCCAM_LINE_SEARCH;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "RUN_INEXACT_OCCAM_LINE_SEARCH");
			inFile >> ibuf;
			if (ibuf == 1)
			{
				m_inexactMinimizationOfOCCAM = true;
				OutputFiles::m_logFile << "# Run OCCAM with inexact Phase-I RMS minimization" << std::endl;
			}
			else if (ibuf != 0)
			{
				OutputFiles::m_logFile << "Error : RUN_INEXACT_OCCAM_LINE_SEARCH must be 0 or 1 : " << ibuf << std::endl;
				exit(1);
			}
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 13).compare("ALPHA_COOLING") == 0)
		{
			if (!hasAlreadyRead[AnalysisControl::INV_METHOD])
			{
				OutputFiles::m_logFile << "Error : You must write INV_METHOD data above ALPHA_COOLING" << std::endl;
				exit(1);
			}
			if (m_inversionMethod != Inversion::DATA_FIT_COOLING_DATA_SPECE)
			{
				OutputFiles::m_logFile << "Error : Data-fit-bracketed cooling requires INV_METHOD 6." << std::endl;
				exit(1);
			}
			const int paramID = AnalysisControl::ALPHA_COOLING;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "ALPHA_COOLING");
			if (!(inFile >> m_dataFitCoolingInitialRmsDecreaseThreshold
				>> m_dataFitCoolingTriggerThreshold
				>> m_dataFitCoolingFactor
				>> m_dataFitCoolingMinimumAlpha))
			{
				OutputFiles::m_logFile
					<< "Error : ALPHA_COOLING requires four values: initial RMS decrease threshold, "
					<< "cooling trigger threshold, cooling factor, and minimum alpha." << std::endl;
				exit(1);
			}
			if (m_dataFitCoolingInitialRmsDecreaseThreshold <= 0.0 ||
				m_dataFitCoolingInitialRmsDecreaseThreshold >= 1.0)
			{
				OutputFiles::m_logFile << "Error : ALPHA_COOLING initial RMS decrease threshold must be between 0 and 1." << std::endl;
				exit(1);
			}
			if (m_dataFitCoolingTriggerThreshold <= 0.0 ||
				m_dataFitCoolingTriggerThreshold >= 1.0)
			{
				OutputFiles::m_logFile << "Error : ALPHA_COOLING cooling trigger threshold must be between 0 and 1." << std::endl;
				exit(1);
			}
			if (m_dataFitCoolingFactor <= 0.0 || m_dataFitCoolingFactor >= 1.0)
			{
				OutputFiles::m_logFile << "Error : ALPHA_COOLING factor must be between 0 and 1." << std::endl;
				exit(1);
			}
			if (m_dataFitCoolingMinimumAlpha <= 0.0)
			{
				OutputFiles::m_logFile << "Error : ALPHA_COOLING minimum alpha must be positive." << std::endl;
				exit(1);
			}
			hasAlreadyRead[paramID] = true;
		}

		else if (line.substr(0, 14).compare("APPRAISAL_MODE") == 0)
		{
			const int paramID = AnalysisControl::APPRAISAL_MODE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "APPRAISAL_MODE");
			inFile >> ibuf;
			if (!isKnownAppraisalMode(ibuf))
			{
				OutputFiles::m_logFile
					<< "Error : APPRAISAL_MODE must be 0, 1, or 2 : " << ibuf
					<< ". Use 0 for model-resolution + covariance diagonals, "
					<< "1 for model-resolution diagonal only, and "
					<< "2 for covariance diagonal only. Omit APPRAISAL_MODE to disable appraisal."
					<< std::endl;
				exit(1);
			}
			if (!isSupportedAppraisalMode(ibuf))
			{
				OutputFiles::m_logFile
					<< "Error : APPRAISAL_MODE " << ibuf << " ("
					<< appraisalModeLabel(ibuf)
					<< ") is currently unsupported. The maintained appraisal migration "
					<< "scope is limited to model-resolution diagonal (2) and "
					<< "covariance diagonal (3)." << std::endl;
				exit(1);
			}
			m_appraisalMode = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 24).compare("APPRAISAL_RANDOM_VECTORS") == 0)
		{
			const int paramID = AnalysisControl::APPRAISAL_RANDOM_VECTORS;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "APPRAISAL_RANDOM_VECTORS");
			inFile >> ibuf;
			if (ibuf <= 0)
			{
				OutputFiles::m_logFile << "Error : APPRAISAL_RANDOM_VECTORS must be positive : " << ibuf << std::endl;
				exit(1);
			}
			m_numRandomVectorsForAppraisal = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 21).compare("APPRAISAL_CHECKPOINTS") == 0)
		{
			rejectDeprecatedAppraisalKeyword("APPRAISAL_CHECKPOINTS");
		}
		else if (line.substr(0, 31).compare("APPRAISAL_INPUT_SENSITIVITY_DIR") == 0)
		{
			rejectDeprecatedAppraisalKeyword("APPRAISAL_INPUT_SENSITIVITY_DIR");
		}
		else if (line.substr(0, 20).compare("APPRAISAL_OUTPUT_DIR") == 0)
		{
			rejectDeprecatedAppraisalKeyword("APPRAISAL_OUTPUT_DIR");
		}
		else if (line.substr(0, 27).compare("APPRAISAL_WRITE_LEGACY_DSDK") == 0)
		{
			rejectDeprecatedAppraisalKeyword("APPRAISAL_WRITE_LEGACY_DSDK");
		}
		else if (line.substr(0, 16).compare("BOUNDS_DIST_THLD") == 0)
		{
			const int paramID = AnalysisControl::BOUNDS_DIST_THLD;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "BOUNDS_DIST_THLD");
			inFile >> dbuf;
			if (dbuf <= 0.0)
			{
				OutputFiles::m_logFile << "Error : Minimum distance to resistivity bounds must be positive !!" << std::endl;
				exit(1);
			}
			ptrResistivityBlock->setMinDistanceToBounds(dbuf);
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 3).compare("IDW") == 0)
		{
			const int paramID = AnalysisControl::IDW;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "IDW");
			inFile >> dbuf;
			if (dbuf < 0.0)
			{
				OutputFiles::m_logFile << "Error : Factor of inverse distance weighting must not be negative !!" << std::endl;
				exit(1);
			}
			ptrResistivityBlock->setInverseDistanceWeightingFactor(dbuf);
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 11).compare("SMALL_VALUE") == 0)
		{
			const int paramID = AnalysisControl::SMALL_VALUE;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "LEVENBERG_MARQUARDT");
			inFile >> dbuf;
			if (dbuf < 0.0)
			{
				OutputFiles::m_logFile << "Error : Small value added to the diagonals of roughning matrix must not be negative !!" << std::endl;
				exit(1);
			}
			ptrResistivityBlock->setFlagAddSmallValueToDiagonals(true);
			ptrResistivityBlock->setSmallValueAddedToDiagonals(dbuf);
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 19).compare("LEVENBERG_MARQUARDT") == 0)
		{
			const int paramID = AnalysisControl::Levenberg_Marquardt;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "SMALL_VALUE");
			inFile >> dbuf;
			if (dbuf < 0.0)
			{
				OutputFiles::m_logFile << "Error : Damping of Levenberg_Marquardt added to the diagonals of Hessian matrix must not be negative !!" << std::endl;
				exit(1);
			}
			m_Levenberg_Marquardt = true;
			m_dampingof_LM = dbuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 12).compare("MOVE_OBS_LOC") == 0)
		{
			const int paramID = AnalysisControl::MOVE_OBS_LOC;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "MOVE_OBS_LOC");
			m_isObsLocMovedToCenter = true;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 13).compare("OWNER_ELEMENT") == 0)
		{
			const int paramID = AnalysisControl::OWNER_ELEMENT;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "OWNER_ELEMENT");
			inFile >> ibuf;
			if (ibuf < 0)
			{
				m_isTypeOfOwnerElementSetIndivisually = true;
			}
			else if (ibuf != AnalysisControl::USE_LOWER_ELEMENT && ibuf != AnalysisControl::USE_UPPER_ELEMENT)
			{
				OutputFiles::m_logFile << "Error : Unknown type of owner element is specified in OWNER_ELEMENT : " << ibuf << std::endl;
				exit(1);
			}
			m_typeOfOwnerElement = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 14).compare("APP_PHS_OPTION") == 0)
		{
			const int paramID = AnalysisControl::APP_PHS_OPTION;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "APP_PHS_OPTION");
			inFile >> ibuf;
			if (ibuf != NO_SPECIAL_TREATMENT_APP_AND_PHASE && ibuf != USE_Z_IF_SIGN_OF_RE_Z_DIFFER)
			{
				OutputFiles::m_logFile << "Error : Unknown option is specified in APP_PHS_OPTION : " << ibuf << std::endl;
				exit(1);
			}
			m_apparentResistivityAndPhaseTreatmentOption = ibuf;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 19).compare("OUTPUT_ROUGH_MATRIX") == 0)
		{
			const int paramID = AnalysisControl::OUTPUT_ROUGH_MATRIX;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "OUTPUT_ROUGH_MATRIX");
			m_isRougheningMatrixOutputted = true;
			hasAlreadyRead[paramID] = true;
		}
		else if (line.substr(0, 17).compare("DATA_SPACE_METHOD") == 0)
		{
			const int paramID = AnalysisControl::DATA_SPACE_METHOD;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "DATA_SPACE_METHOD");
			inFile >> ibuf;
			m_typeOfDataSpaceAlgorithm = ibuf;
			hasAlreadyRead[paramID] = true;
#ifdef _ANISOTOROPY
		}
		else if (line.substr(0, 10).compare("ANISOTROPY") == 0)
		{
			const int paramID = AnalysisControl::ANISOTROPY;
			ControlKeywords::ensureNotAlreadyRead(hasAlreadyRead, paramID, "ANISOTROPY");
			inFile >> ibuf;
			m_typeOfAnisotropy = ibuf;
			hasAlreadyRead[paramID] = true;
#endif
		}
		else if (line.substr(0, 11).compare("DIFF_FILTER") == 0)
		{
			m_useDifferenceFilter = true;
			inFile >> ibuf;
			m_degreeOfLpOptimization = ibuf;
			if (m_degreeOfLpOptimization == 0)
			{
				inFile >> dbuf;
				m_smallvauleOfMinimumGradientSupport = dbuf;
			}
			else if (m_degreeOfLpOptimization == 1 || m_degreeOfLpOptimization == 2)
			{
				inFile >> dbuf;
				m_lowerLimitOfDifflog10RhoForLpOptimization = dbuf;
				inFile >> dbuf;
				m_upperLimitOfDifflog10RhoForLpOptimization = dbuf;
			}
			inFile >> ibuf;
			m_maxIterationIRWLSForLpOptimization = ibuf;
			inFile >> dbuf;
			m_thresholdIRWLSForLpOptimization = dbuf;
		}else if (line.substr(0,19).compare("SENSE_MAT_DIRECTORY") == 0) {
			inFile >> m_directoryOfOutOfCoreFilesForSensitivityMatrix;
		}
		else if (ControlKeywords::isEndKeywordLine(line))
		{
			break;
		}
		else
		{
			OutputFiles::m_logFile << "Error : Improper data !! " << line << std::endl;
			exit(1);
		}
	}
	inFile.close();

	if (!hasAlreadyRead[AnalysisControl::DISTORTION])
	{
		OutputFiles::m_logFile << "Error : You must write DISTORTION data in control.dat" << std::endl;
		exit(1);
	}

	if (m_iterationNumMax > m_iterationNumInit && !hasAlreadyRead[AnalysisControl::TRADE_OFF_PARAM])
	{
		OutputFiles::m_logFile << "Error : Trade-off parameter must be specified for inversion !!" << std::endl;
		exit(1);
	}
	if (m_inversionMethod == Inversion::DATA_FIT_COOLING_DATA_SPECE &&
		m_typeOfTradeOffParam != AnalysisControl::TO_DATA_FIT_COOLING)
	{
		OutputFiles::m_logFile
			<< "Error : INV_METHOD 6 requires TYPE_OF_TRADE_OFF_PARAMETER 5."
			<< std::endl;
		exit(1);
	}
	if (m_typeOfTradeOffParam == AnalysisControl::TO_DATA_FIT_COOLING &&
		m_inversionMethod != Inversion::DATA_FIT_COOLING_DATA_SPECE)
	{
		OutputFiles::m_logFile
			<< "Error : Data-fit-bracketed cooling requires INV_METHOD 6."
			<< std::endl;
		exit(1);
	}
	if (m_inversionMethod == Inversion::DATA_FIT_COOLING_DATA_SPECE)
	{
		if (!hasAlreadyRead[AnalysisControl::ALPHA_COOLING])
		{
			OutputFiles::m_logFile << "Error : ALPHA_COOLING is required for INV_METHOD 6." << std::endl;
			exit(1);
		}
		if (m_dataFitCoolingMinimumAlpha > m_dataFitCoolingInitialAlpha)
		{
			OutputFiles::m_logFile << "Error : ALPHA_COOLING minimum alpha must not exceed the initial alpha." << std::endl;
			exit(1);
		}
		if (m_useDifferenceFilter && m_degreeOfLpOptimization != 2)
		{
			OutputFiles::m_logFile
				<< "Error : Data-fit-bracketed cooling supports Difference-filter L2 and Laplacian L2 only."
				<< std::endl;
			exit(1);
		}
	}

	if (m_inversionMethod == Inversion::OCCAM_DATA_SPECE &&
		m_typeOfTradeOffParam == AnalysisControl::TO_LINEAR_LCURVE)
	{
		OutputFiles::m_logFile
			<< "Error : The old linear cubic-spline L-curve control combination "
			<< "(INV_METHOD 3, TYPE_OF_TRADE_OFF_PARAMETER 3) has been migrated."
			<< " Use INV_METHOD " << Inversion::LINEAR_LCURVE_DATA_SPECE
			<< " and TYPE_OF_TRADE_OFF_PARAMETER " << AnalysisControl::TO_LINEAR_LCURVE
			<< "." << std::endl;
		exit(1);
	}
	if (m_inversionMethod == Inversion::LINEAR_LCURVE_DATA_SPECE &&
		m_typeOfTradeOffParam == AnalysisControl::TO_OCCAM_LS)
	{
		OutputFiles::m_logFile
			<< "Error : The old OCCAM control combination "
			<< "(INV_METHOD 4, TYPE_OF_TRADE_OFF_PARAMETER 2) has been migrated."
			<< " Use INV_METHOD " << Inversion::OCCAM_DATA_SPECE
			<< " and TYPE_OF_TRADE_OFF_PARAMETER " << AnalysisControl::TO_OCCAM_LS
			<< "." << std::endl;
		exit(1);
	}
	if (m_typeOfTradeOffParam == AnalysisControl::TO_OCCAM_LS &&
		m_inversionMethod != Inversion::OCCAM_DATA_SPECE)
	{
		OutputFiles::m_logFile
			<< "Error : OCCAM line search requires INV_METHOD "
			<< Inversion::OCCAM_DATA_SPECE << "." << std::endl;
		exit(1);
	}
	if (m_typeOfTradeOffParam == AnalysisControl::TO_LINEAR_LCURVE &&
		m_inversionMethod != Inversion::LINEAR_LCURVE_DATA_SPECE)
	{
		OutputFiles::m_logFile << "Error : Linear cubic-spline L-curve selection requires INV_METHOD "
			<< Inversion::LINEAR_LCURVE_DATA_SPECE << "." << std::endl;
		exit(1);
	}
	if (m_typeOfTradeOffParam == AnalysisControl::TO_NONLINEAR_LCURVE &&
		m_inversionMethod != Inversion::NONLINEAR_LCURVE_DATA_SPECE)
	{
		OutputFiles::m_logFile << "Error : Nonlinear cubic-spline L-curve selection requires INV_METHOD "
			<< Inversion::NONLINEAR_LCURVE_DATA_SPECE << "." << std::endl;
		exit(1);
	}
	if (m_inversionMethod == Inversion::LINEAR_LCURVE_DATA_SPECE &&
		m_typeOfTradeOffParam != AnalysisControl::TO_LINEAR_LCURVE)
	{
		OutputFiles::m_logFile << "Error : INV_METHOD " << Inversion::LINEAR_LCURVE_DATA_SPECE
			<< " requires TYPE_OF_TRADE_OFF_PARAMETER "
			<< AnalysisControl::TO_LINEAR_LCURVE << "." << std::endl;
		exit(1);
	}
	if (m_inversionMethod == Inversion::NONLINEAR_LCURVE_DATA_SPECE &&
		m_typeOfTradeOffParam != AnalysisControl::TO_NONLINEAR_LCURVE)
	{
		OutputFiles::m_logFile << "Error : INV_METHOD " << Inversion::NONLINEAR_LCURVE_DATA_SPECE
			<< " requires TYPE_OF_TRADE_OFF_PARAMETER "
			<< AnalysisControl::TO_NONLINEAR_LCURVE << "." << std::endl;
		exit(1);
	}
	if (m_inversionMethod == Inversion::OCCAM_DATA_SPECE)
	{
		if (m_typeOfTradeOffParam != AnalysisControl::TO_OCCAM_LS)
		{
			OutputFiles::m_logFile << "Error : INV_METHOD " << Inversion::OCCAM_DATA_SPECE
				<< " requires TYPE_OF_TRADE_OFF_PARAMETER "
				<< AnalysisControl::TO_OCCAM_LS << "." << std::endl;
			exit(1);
		}
	}
	if (isAppraisalEnabled())
	{
		if (m_numRandomVectorsForAppraisal <= 0)
		{
			OutputFiles::m_logFile << "Error : APPRAISAL_RANDOM_VECTORS must be positive." << std::endl;
			exit(1);
		}
		m_appraisalCheckpoints = buildDefaultAppraisalCheckpoints(m_numRandomVectorsForAppraisal);
		if (m_appraisalCheckpoints.empty())
		{
			OutputFiles::m_logFile << "Error : appraisal checkpoints could not be generated." << std::endl;
			exit(1);
		}
		for (std::vector<int>::const_iterator itr = m_appraisalCheckpoints.begin(); itr != m_appraisalCheckpoints.end(); ++itr)
		{
			if (*itr > m_numRandomVectorsForAppraisal)
			{
				OutputFiles::m_logFile << "Error : generated appraisal checkpoint is greater than APPRAISAL_RANDOM_VECTORS : "
					<< *itr << " > " << m_numRandomVectorsForAppraisal << "." << std::endl;
				exit(1);
			}
		}
	}

	OutputFiles::m_logFile << "#==============================================================================" << std::endl;
	OutputFiles::m_logFile << "# Summary of control data" << std::endl;
	OutputFiles::m_logFile << "#==============================================================================" << std::endl;

	if (m_boundaryConditionBottom == AnalysisControl::BOUNDARY_BOTTOM_ONE_DIMENSIONAL)
	{
		OutputFiles::m_logFile << "# 1D boundary condition is specified at the bottom boundary." << std::endl;
	}
	else if (m_boundaryConditionBottom == AnalysisControl::BOUNDARY_BOTTOM_PERFECT_CONDUCTOR)
	{
		OutputFiles::m_logFile << "# Condition of perfect electric conductor is specified at the bottom boundary." << std::endl;
	}
	else
	{
		OutputFiles::m_logFile << "Error : Wrong type of boundary condition at the bottom of the model !! m_boundaryConditionBottom = " << m_boundaryConditionBottom << "." << std::endl;
		exit(1);
	}

	if (m_typeOfMesh == MeshData::HEXA)
	{
		OutputFiles::m_logFile << "# Type of mesh : Hexahedron." << std::endl;
	}
	else if (m_typeOfMesh == MeshData::TETRA)
	{
		OutputFiles::m_logFile << "# Type of mesh : Tetrahedron." << std::endl;
	}
	else if (m_typeOfMesh == MeshData::NONCONFORMING_HEXA)
	{
		OutputFiles::m_logFile << "# Type of mesh : Deformed hexahedron." << std::endl;
	}
	else
	{
		OutputFiles::m_logFile << "Error : Wrong type of mesh !! m_typeOfMesh = " << m_typeOfMesh << "." << std::endl;
		exit(1);
	}
	OutputFiles::m_logFile << "# Number of threads is specified to be " << m_numThreads << " ." << std::endl;
	// Specifies the number of threads to use
#ifdef _USE_OMP
	omp_set_num_threads(m_numThreads);
#endif
	mkl_set_num_threads(m_numThreads);

	switch (m_modeOfPARDISO)
	{
	case PARDISOSolver::INCORE_MODE:
		OutputFiles::m_logFile << "# In-core mode is used." << std::endl;
		break;
	case PARDISOSolver::SELECT_MODE_AUTOMATICALLY:
		OutputFiles::m_logFile << "# Either in-core or out-of-core is used depending on the memory required." << std::endl;
		break;
	case PARDISOSolver::OUT_OF_CORE_MODE:
		OutputFiles::m_logFile << "# Out-of-core mode is used." << std::endl;
		break;
	default:
		OutputFiles::m_logFile << "Error : Wrong value m_modeOfPARDISO !! m_modeOfPARDISO = " << m_modeOfPARDISO << std::endl;
		exit(1);
		break;
	}
	OutputFiles::m_logFile << "# Division number of right-hand sides at solve phase in forward calculation : " << m_divisionNumberOfMultipleRHSInForward << std::endl;
	OutputFiles::m_logFile << "# Division number of right-hand sides at solve phase in inversion : " << m_divisionNumberOfMultipleRHSInInversion << std::endl;

	if (m_modeOfPARDISO == PARDISOSolver::SELECT_MODE_AUTOMATICALLY || m_modeOfPARDISO == PARDISOSolver::OUT_OF_CORE_MODE)
	{

#ifdef _LINUX
		std::ostringstream strMem;
		strMem << m_maxMemoryPARDISO;
		if (setenv("MKL_PARDISO_OOC_MAX_CORE_SIZE", strMem.str().c_str(), 1) != 0)
		{
			OutputFiles::m_logFile << "Error : Environment variable MKL_PARDISO_OOC_MAX_CORE_SIZE was not set correctly ! " << std::endl;
			exit(1);
		}
#else
		std::ostringstream strEnv;
		strEnv << "MKL_PARDISO_OOC_MAX_CORE_SIZE=" << m_maxMemoryPARDISO;
#ifdef _DEBUG_WRITE
		std::cout << "strEnv " << strEnv.str() << std::endl; // For debug
#endif
		if (putenv(const_cast<char *>(strEnv.str().c_str())) != 0)
		{
			OutputFiles::m_logFile << "Error : Environment variable MKL_PARDISO_OOC_MAX_CORE_SIZE was not set correctly ! " << std::endl;
			exit(1);
		}
#endif

		OutputFiles::m_logFile << "# Maximum value of the memory used by out-of-core mode of forward solver : " << m_maxMemoryPARDISO << " [MB]" << std::endl;
	}

	if (m_positiveDefiniteNormalEqMatrix)
	{
		OutputFiles::m_logFile << "# Coefficient matrix of normal equation is assumed to be positive definite." << std::endl;
	}
	else
	{
		OutputFiles::m_logFile << "# Coefficient matrix of normal equation is assumed to be indefinite." << std::endl;
	}

	switch (m_numberingMethod)
	{
	case AnalysisControl::NOT_ASSIGNED:
		OutputFiles::m_logFile << "# Renumbering is not performed." << std::endl;
		break;
	case AnalysisControl::XYZ:
		OutputFiles::m_logFile << "# Numbering edges or nodes in the way X => Y => Z ." << std::endl;
		break;
	case AnalysisControl::YZX:
		OutputFiles::m_logFile << "# Numbering edges or nodes in the way Y => Z => X ." << std::endl;
		break;
	case AnalysisControl::ZXY:
		OutputFiles::m_logFile << "# Numbering edges or nodes in the way Z => X => Y ." << std::endl;
		break;
	default:
		OutputFiles::m_logFile << "Error : Wrong value m_numberingMethod !! m_numberingMethod = " << m_modeOfPARDISO << std::endl;
		exit(1);
		break;
	}

	if (m_typeOfMesh == MeshData::HEXA)
	{
		if (m_typeOfElectricField != AnalysisControl::USE_HORIZONTAL_ELECTRIC_FIELD)
		{
			OutputFiles::m_logFile << "Warning : Horizontal electric field is used for hexahedral mesh." << std::endl;
			m_typeOfElectricField = AnalysisControl::USE_HORIZONTAL_ELECTRIC_FIELD;
		}
	}
	if (m_isTypeOfElectricFieldSetIndivisually)
	{
		OutputFiles::m_logFile << "# Electric field type of each site is specified in observe.dat." << std::endl;
	}
	else
	{
		switch (m_typeOfElectricField)
		{
		case AnalysisControl::USE_TANGENTIAL_ELECTRIC_FIELD:
			OutputFiles::m_logFile << "# Tangential electric field is used for calculating response functions." << std::endl;
			break;
		case AnalysisControl::USE_HORIZONTAL_ELECTRIC_FIELD:
			OutputFiles::m_logFile << "# Horizontal electric field is used for calculating response functions." << std::endl;
			break;
		default:
			OutputFiles::m_logFile << "Error : Unknown type of the electric field : " << m_typeOfElectricField << std::endl;
			exit(1);
			break;
		}
	}

	if (m_isTypeOfOwnerElementSetIndivisually)
	{
		OutputFiles::m_logFile << "# Owner element type of each site is specified in observe.dat." << std::endl;
	}
	else
	{
		switch (m_typeOfOwnerElement)
		{
		case AnalysisControl::USE_LOWER_ELEMENT:
			OutputFiles::m_logFile << "# EM field is interpolated from the values of the edges of the lower element." << std::endl;
			break;
		case AnalysisControl::USE_UPPER_ELEMENT:
			OutputFiles::m_logFile << "# EM field is interpolated from the values of the edges of the upper element." << std::endl;
			break;
		default:
			OutputFiles::m_logFile << "Error : Unknown type of owner element : " << m_typeOfOwnerElement << std::endl;
			exit(1);
			break;
		}
	}

	switch (m_apparentResistivityAndPhaseTreatmentOption)
	{
	case AnalysisControl::NO_SPECIAL_TREATMENT_APP_AND_PHASE:
		break;
	case AnalysisControl::USE_Z_IF_SIGN_OF_RE_Z_DIFFER:
		OutputFiles::m_logFile << "# Impedance tensor is used instead of apparent resistivity and phase if signs of Re(Z) are different between observed and calculated responses." << std::endl;
		break;
	default:
		OutputFiles::m_logFile << "Error : Unknown type of owner element : " << m_typeOfOwnerElement << std::endl;
		exit(1);
		break;
	}

	if (m_typeOfMesh == MeshData::HEXA)
	{
		if (m_useBackwardOrForwardElement.directionX == AnalysisControl::BACKWARD_ELEMENT)
		{
			OutputFiles::m_logFile << "# Element of -X direction is used for points locating on boudarieds of elements." << std::endl;
		}
		else
		{
			OutputFiles::m_logFile << "# Element of +X direction is used for points locating on boudarieds of elements." << std::endl;
		}

		if (m_useBackwardOrForwardElement.directionY == AnalysisControl::BACKWARD_ELEMENT)
		{
			OutputFiles::m_logFile << "# Element of -Y direction is used for points locating on boudarieds of elements." << std::endl;
		}
		else
		{
			OutputFiles::m_logFile << "# Element of +Y direction is used for points locating on boudarieds of elements." << std::endl;
		}
	}

	if (m_isObsLocMovedToCenter)
	{
		OutputFiles::m_logFile << "# Observation point is moved to the horizontal center of the element including it." << std::endl;
	}

	if (m_holdMemoryForwardSolver)
	{
		OutputFiles::m_logFile << "# Hold memory of coefficient matrix and sparse solver after forward calculation." << std::endl;
	}
	else
	{
		OutputFiles::m_logFile << "# Release memory of coefficient matrix and sparse solver after forward calculation." << std::endl;
	}

	const int bountConstraingMethod = (ResistivityBlock::getInstance())->getTypeBoundConstraints();
	if (bountConstraingMethod == ResistivityBlock::SIMPLE_BOUND_CONSTRAINING)
	{
		OutputFiles::m_logFile << "# Type of bound constraints method : Simple bound constraining" << std::endl;
	}
	else if (bountConstraingMethod == ResistivityBlock::TRANSFORMING_METHOD)
	{
		OutputFiles::m_logFile << "# Type of bound constraints method : Transforming method" << std::endl;
	}
	else
	{
		OutputFiles::m_logFile << "Error : Wrong type of bound constraining method !! : " << bountConstraingMethod << " ." << std::endl;
		exit(1);
	}

	OutputFiles::m_logFile << "# Minimum distance to resistivity bounds in common logarithm scale : " << ptrResistivityBlock->getMinDistanceToBounds() << " ." << std::endl;

	if (ptrResistivityBlock->includeBottomResistivity())
	{
		OutputFiles::m_logFile << "# Bottom resistivity : " << ptrResistivityBlock->getBottomResistivity() << " [Ohm-m]" << std::endl;
		OutputFiles::m_logFile << "# Roughning factor at the bottom : " << ptrResistivityBlock->getRoughningFactorAtBottom() << std::endl;
	}
	else if (ptrResistivityBlock->getFlagAddSmallValueToDiagonals())
	{
		if (getTypeOfDataSpaceAlgorithm() == AnalysisControl::NEW_DATA_SPACE_ALGORITHM_USING_INV_RTR_MATRIX)
		{
			OutputFiles::m_logFile << "# Small value added to the diagonals of [R]T*[R] matrix : " << ptrResistivityBlock->getSmallValueAddedToDiagonals() << std::endl;
		}
		else
		{
			OutputFiles::m_logFile << "# Small value added to the diagonals of roughning matrix : " << ptrResistivityBlock->getSmallValueAddedToDiagonals() << std::endl;
		}
	}
	else if (getInversionMethod() == Inversion::GAUSS_NEWTON_DATA_SPECE)
	{
		OutputFiles::m_logFile << "Error : You must give small number added to diagonals of roughning matrix" << std::endl;
		OutputFiles::m_logFile << "        when data space inverson method is selected !!" << std::endl;
#ifdef _DEBUG_WRITE
#else
		exit(1);
#endif
	}

	if (m_useDifferenceFilter)
	{
		if (getTypeOfDataSpaceAlgorithm() != AnalysisControl::NEW_DATA_SPACE_ALGORITHM_USING_INV_RTR_MATRIX)
		{
			OutputFiles::m_logFile << "Error : You must select " << AnalysisControl::NEW_DATA_SPACE_ALGORITHM_USING_INV_RTR_MATRIX << " as DATA_SPACE_METHOD when you use Lp optimization" << std::endl;
			exit(1);
		}
		OutputFiles::m_logFile << "# Degree of Lp optimization : " << m_degreeOfLpOptimization << std::endl;
		if (m_degreeOfLpOptimization == 0)
		{
			OutputFiles::m_logFile << "# Small value for minimum-gradient-support L0 optimization : " << m_smallvauleOfMinimumGradientSupport << std::endl;
		}
		else
		{
			OutputFiles::m_logFile << "# Range of difference of log10(rho) for Lp optimization : " << m_lowerLimitOfDifflog10RhoForLpOptimization << " - " << m_upperLimitOfDifflog10RhoForLpOptimization << std::endl;
		}
		OutputFiles::m_logFile << "# Maximum iteration number of IRWLS for Lp optimization : " << m_maxIterationIRWLSForLpOptimization << std::endl;
		OutputFiles::m_logFile << "# Convergence criteria of IRWLS for Lp optimization [%] : " << m_thresholdIRWLSForLpOptimization << std::endl;
	}

#ifdef _ANISOTOROPY
	switch (getTypeOfAnisotropy())
	{
	case AnalysisControl::NO_ANISOTROPY:
		// No anisotropy => Nothing to do
		break;
	case AnalysisControl::AXIAL_ANISOTROPY:
		OutputFiles::m_logFile << "# Axial anisotropy is considered." << std::endl;
		if (m_typeOfMesh != MeshData::TETRA)
		{
			OutputFiles::m_logFile << "Error : Axial anisotropys is supported only for tetrahedral mesh !!" << std::endl;
			exit(1);
		}
		break;
	default:
		OutputFiles::m_logFile << "Error : Wrong type of anisotropy : " << getTypeOfAnisotropy() << std::endl;
		exit(1);
		break;
	}
#endif

	// Open VTK file
	if (!m_outputParametersForVis.empty())
	{
		if (writeBinaryFormat())
		{
			OutputFiles::m_logFile << "# Following variables are written to BINARY file." << std::endl;
		}
		else
		{
			OutputFiles::m_logFile << "# Following variables are written to to ASCII file." << std::endl;
		}
		if (doesOutputToVTK(AnalysisControl::OUTPUT_RESISTIVITY_VALUES_TO_VTK))
		{
			OutputFiles::m_logFile << "#  - Resistivity" << std::endl;
		}
		if (doesOutputToVTK(AnalysisControl::OUTPUT_ELECTRIC_FIELD_VECTORS_TO_VTK))
		{
			OutputFiles::m_logFile << "#  - Electric field" << std::endl;
		}
		if (doesOutputToVTK(AnalysisControl::OUTPUT_MAGNETIC_FIELD_VECTORS_TO_VTK))
		{
			OutputFiles::m_logFile << "#  - Magnetic field" << std::endl;
		}
		if (doesOutputToVTK(AnalysisControl::OUTPUT_CURRENT_DENSITY))
		{
			OutputFiles::m_logFile << "#  - Current density" << std::endl;
		}
		if (doesOutputToVTK(AnalysisControl::OUTPUT_SENSITIVITY))
		{
			OutputFiles::m_logFile << "#  - Sensitivity" << std::endl;
		}
		if (doesOutputToVTK(AnalysisControl::OUTPUT_SENSITIVITY_DENSITY))
		{
			OutputFiles::m_logFile << "#  - Sensitivity density" << std::endl;
		}
	}

	// Open csv file in which the results of 2D forward computations is written
	if (m_isOutput2DResult)
	{
		OutputFiles::m_logFile << "# Output results of 2D forward computations to csv file." << std::endl;
		// OutputFiles* const ptrOutputFiles = OutputFiles::getInstance();
		// ptrOutputFiles->openCsvFileFor2DFwd();
	}

	OutputFiles::m_logFile << "# Method of inversion : ";
	switch (getInversionMethod())
	{
	case Inversion::GAUSS_NEWTON_MODEL_SPECE:
		OutputFiles::m_logFile << "Gauss-newton method (Model space)" << std::endl;
		break;
	case Inversion::GAUSS_NEWTON_DATA_SPECE:
		switch (getTypeOfDataSpaceAlgorithm())
		{
		case AnalysisControl::NEW_DATA_SPACE_ALGORITHM:
			OutputFiles::m_logFile << "Gauss-newton method (Data space)" << std::endl;
			break;
		case AnalysisControl::NEW_DATA_SPACE_ALGORITHM_USING_INV_RTR_MATRIX:
			OutputFiles::m_logFile << "Gauss-newton method (Data space) using inverse of [R]T*[R] matrix" << std::endl;
			break;
		default:
			OutputFiles::m_logFile << "Error : Type of data space inversion algorithm is wrong  !! : " << getTypeOfDataSpaceAlgorithm() << std::endl;
			exit(1);
			break;
		}
		break;
	case Inversion::ABIC_DATA_SPECE:
		OutputFiles::m_logFile << "ABIC inversion in data space" << std::endl;
		break;
	case Inversion::OCCAM_DATA_SPECE:
		OutputFiles::m_logFile << "OCCAM inversion in data space" << std::endl;
		break;
	case Inversion::LINEAR_LCURVE_DATA_SPECE:
		OutputFiles::m_logFile << "Linear cubic-spline L-curve inversion in data space" << std::endl;
		break;
	case Inversion::NONLINEAR_LCURVE_DATA_SPECE:
		OutputFiles::m_logFile << "Nonlinear cubic-spline L-curve inversion in data space" << std::endl;
		break;
	case Inversion::DATA_FIT_COOLING_DATA_SPECE:
		OutputFiles::m_logFile << "Data-fit-bracketed cooling inversion in data space" << std::endl;
		break;
	default:
		OutputFiles::m_logFile << "Error : Type of inversion method is wrong  !! : " << getInversionMethod() << std::endl;
		exit(1);
		break;
	}

	if (estimateDistortionMatrix())
	{
		if (m_typeOfDistortion == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
		{
			OutputFiles::m_logFile << "# Components of distortion matrices are estimated directly as model parameters." << std::endl;
		}
		else if (m_typeOfDistortion == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
		{
			OutputFiles::m_logFile << "# Gains and rotations of distortion matrices are estimated as model parameters." << std::endl;
		}
		else if (m_typeOfDistortion == AnalysisControl::ESTIMATE_GAINS_ONLY)
		{
			OutputFiles::m_logFile << "# Gains of distortion matrices are estimated as model parameters." << std::endl;
		}
	}
	else
	{
		OutputFiles::m_logFile << "# Distortion matrices are NOT estimated as model parameters." << std::endl;
	}

	if (m_typeOfRoughningMatrix == AnalysisControl::USE_ELEMENTS_SHARE_FACES)
	{
		OutputFiles::m_logFile << "# Roughening matrix is created using shared faces of elements." << std::endl;
	}
	else if (m_typeOfRoughningMatrix == AnalysisControl::USER_DEFINED_ROUGHNING)
	{
		OutputFiles::m_logFile << "# Roughening matrix is created from user-defined roughning factor." << std::endl;
	}
	else if (m_typeOfRoughningMatrix == AnalysisControl::USE_RESISTIVITY_BLOCKS_SHARE_FACES)
	{
		OutputFiles::m_logFile << "# Roughening matrix is created using shared faces of resistivity blocks." << std::endl;
	}
	else if (m_typeOfRoughningMatrix == AnalysisControl::USE_ELEMENTS_SHARE_FACES_AREA_VOL_RATIO)
	{
		OutputFiles::m_logFile << "# Roughening matrix is created using shared faces of elements (weighting by area-volume ratio)." << std::endl;
	}
	else
	{
		OutputFiles::m_logFile << "Error : Number of parameter specifing the way of creating roughning matrix must be 0 , 1 or 2 !!" << std::endl;
		exit(1);
	}

	if (m_isRougheningMatrixOutputted)
	{
		OutputFiles::m_logFile << "# Roughening matrix is outputted." << std::endl;
	}

	if (m_iterationNumMax < m_iterationNumInit)
	{
		OutputFiles::m_logFile << "# Inital number of iteration must be less than or equal to the maximum number." << std::endl;
		exit(1);
	}

	OutputFiles::m_logFile << "# Trade-off parameter for resistivity value : " << m_tradeOffParameterForResistivityValue << " ." << std::endl;
	if (estimateDistortionMatrix())
	{
		if (m_typeOfDistortion == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
		{
			OutputFiles::m_logFile << "# Trade-off parameter for distortion strength : " << m_tradeOffParameterForDistortionMatrixComplexity << " ." << std::endl;
		}
		else if (m_typeOfDistortion == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
		{
			OutputFiles::m_logFile << "# Trade-off parameter for gains of distortion matrix : " << m_tradeOffParameterForDistortionGain << " ." << std::endl;
			OutputFiles::m_logFile << "# Trade-off parameter for rotations of distortion matrix : " << m_tradeOffParameterForDistortionRotation << " ." << std::endl;
		}
		else if (m_typeOfDistortion == AnalysisControl::ESTIMATE_GAINS_ONLY)
		{
			OutputFiles::m_logFile << "# Trade-off parameter for gains of distortion matrix : " << m_tradeOffParameterForDistortionGain << " ." << std::endl;
		}
	}

	if (m_CrossGradientInv)
	{
		OutputFiles::m_logFile << "# Trade-off parameter for cross-gradient : " << m_tradeOffParameterForCrossGradient << " ." << std::endl;
	}

	OutputFiles::m_logFile << "# Weighting factor of alpha (X,Y,Z) = (" << m_alphaWeight[0] << ", " << m_alphaWeight[1] << ", " << m_alphaWeight[2] << ") ." << std::endl;

	OutputFiles::m_logFile << "# Appraisal mode : " << appraisalModeLabel(m_appraisalMode) << " (" << m_appraisalMode << ")." << std::endl;
	if (isAppraisalEnabled())
	{
		OutputFiles::m_logFile << "# Appraisal random vectors : " << m_numRandomVectorsForAppraisal << "." << std::endl;
		OutputFiles::m_logFile << "# Appraisal checkpoints :";
		for (std::vector<int>::const_iterator itr = m_appraisalCheckpoints.begin(); itr != m_appraisalCheckpoints.end(); ++itr)
		{
			OutputFiles::m_logFile << " " << *itr;
		}
		OutputFiles::m_logFile << "." << std::endl;
		OutputFiles::m_logFile << "# Appraisal run-local sensitivity directory : " << m_appraisalInputSensitivityDirectory << "." << std::endl;
		OutputFiles::m_logFile << "# Appraisal output directory : " << m_appraisalOutputDirectory << "." << std::endl;
	}

	OutputFiles::m_logFile << "# Factor of inverse distance weighting : " << ptrResistivityBlock->getInverseDistanceWeightingFactor() << "." << std::endl;

	OutputFiles::m_logFile << "# Initial iteration number : " << m_iterationNumInit << "." << std::endl;

	OutputFiles::m_logFile << "# Maximum iteration number : " << m_iterationNumMax << "." << std::endl;

	OutputFiles::m_logFile << "# Threshold value for determining if objective functional decrease : " << m_thresholdValueForDecreasing << "." << std::endl;

	OutputFiles::m_logFile << "# Convergence criterion of inversion is that change ratios of objective function and its components are less than " << m_decreaseRatioForConvegence << " [%] ." << std::endl;

	if (m_stepLengthDampingFactorCur < 0.0 || m_stepLengthDampingFactorCur > 1.0)
	{
		OutputFiles::m_logFile << "Error : Initial factor of step-length damping " << m_stepLengthDampingFactorCur << " must be greater than or equal to zero and less than or equal to one." << std::endl;
		exit(1);
	}
	OutputFiles::m_logFile << "# Initial factor of step-length damping : " << m_stepLengthDampingFactorCur << "." << std::endl;

	if (m_stepLengthDampingFactorCur < 0.0 || m_stepLengthDampingFactorCur > 1.0)
	{
		OutputFiles::m_logFile << "Error : Minimum factor of step-length damping " << m_stepLengthDampingFactorCur << " must be greater than or equal to zero and less than or equal to one." << std::endl;
		exit(1);
	}
	if (m_stepLengthDampingFactorCur < m_stepLengthDampingFactorMin)
	{
		OutputFiles::m_logFile << "Error : Minimum factor of step-length damping must be less than or equal to the initial one." << std::endl;
		exit(1);
	}
	OutputFiles::m_logFile << "# Minimum factor of step-length damping : " << m_stepLengthDampingFactorMin << "." << std::endl;

	if (m_stepLengthDampingFactorCur < 0.0 || m_stepLengthDampingFactorCur > 1.0)
	{
		OutputFiles::m_logFile << "Error : Maximum factor of step-length damping " << m_stepLengthDampingFactorCur << " must be greater than or equal to zero and less than or equal to one." << std::endl;
		exit(1);
	}
	if (m_stepLengthDampingFactorCur > m_stepLengthDampingFactorMax)
	{
		OutputFiles::m_logFile << "Error : Maximum factor of step-length damping must be greater than or equal to the initial one." << std::endl;
		exit(1);
	}
	OutputFiles::m_logFile << "# Maximum factor of step-length damping : " << m_stepLengthDampingFactorMax << "." << std::endl;

	if (m_factorDecreasingStepLength < 0 || m_factorDecreasingStepLength > 1.0)
	{
		OutputFiles::m_logFile << "Error : Factors of step-length damping is  must be less than or equal to the initial one." << std::endl;
		exit(1);
	}
	OutputFiles::m_logFile << "# If residual increase, factor of step-length damping is muliplied by " << m_factorDecreasingStepLength << " times." << std::endl;

	OutputFiles::m_logFile << "# If residual decrease " << m_numOfIterIncreaseStepLength << " times in a row, factor of step-length damping is muliplied by " << m_factorIncreasingStepLength << " times." << std::endl;

	if (m_numCutbackMax < 0)
	{
		m_continueWithoutCutback = true;
		m_numCutbackMax = 0;
		OutputFiles::m_logFile << "# Continue iteration without retrials." << std::endl;
	}
	else
	{
		m_continueWithoutCutback = false;
		OutputFiles::m_logFile << "# Maximum number of retrials : " << m_numCutbackMax << "." << std::endl;
	}

	if (!getDirectoryOfOutOfCoreFilesForSensitivityMatrix().empty()) {
		OutputFiles::m_logFile << "# Directory of out-of-core files for the sensitivitry matrix: " << getDirectoryOfOutOfCoreFilesForSensitivityMatrix() << std::endl;
	}

	OutputFiles::m_logFile << "#==============================================================================" << std::endl;
}

// Calculate elapsed time
std::string AnalysisControl::outputElapsedTime() const
{

	std::ostringstream output;
	output << "( " << getElapsedTimeInSeconds() << " sec )";

	return output.str();
}

double AnalysisControl::getElapsedTimeInSeconds() const
{
	time_t curTime(NULL);
	time(&curTime);

	return difftime(curTime, m_startTime);
}

// Copy Residual Vector Of Data
void AnalysisControl::getResidualVectorOfDataThisPE(double *vector) const
{
	ObservedData *const ptrObservedData = ObservedData::getInstance();
	int numDataThisPE = ptrObservedData->getNumObservedDataThisPETotal();

	for (int iMdl = 0; iMdl < numDataThisPE; ++iMdl)
	{
		vector[iMdl] = m_residualVectorThisPE[iMdl];
	}
}

#ifdef _ANISOTOROPY
// Get type of anisotropy
int AnalysisControl::getTypeOfAnisotropy() const
{
	return m_typeOfAnisotropy;
}

// Get flag specifing whether anisotropy of conductivity is taken into account
bool AnalysisControl::isAnisotropyConsidered() const
{
	if (getTypeOfAnisotropy() == AnalysisControl::NO_ANISOTROPY)
	{
		return false;
	}
	else
	{
		return true;
	}
}
#endif

// Get pointer to the object of class MeshData
const MeshData *AnalysisControl::getPointerOfMeshData() const
{
	if (getPointerOfForward3D() == NULL)
	{
		OutputFiles::m_logFile << "Error : Pointer to the class Forward3D is NULL." << std::endl;
		exit(1);
	}
	return getPointerOfForward3D()->getPointerToMeshData();
}

// Get pointer to the object of class MeshDataBrickElement
const MeshDataBrickElement *AnalysisControl::getPointerOfMeshDataBrickElement() const
{
	if (m_ptrForward3DBrickElement0thOrder == NULL)
	{
		OutputFiles::m_logFile << "Error : m_ptrForward3DBrickElement0thOrder is NULL." << std::endl;
		exit(1);
	}
	return m_ptrForward3DBrickElement0thOrder->getPointerToMeshDataBrickElement();
}

// Get pointer to the object of class MeshDataTetraElement
const MeshDataTetraElement *AnalysisControl::getPointerOfMeshDataTetraElement() const
{
	if (m_ptrForward3DTetraElement0thOrder == NULL)
	{
		OutputFiles::m_logFile << "Error : m_ptrForward3DTetraElement0thOrder is NULL." << std::endl;
		exit(1);
	}
	return m_ptrForward3DTetraElement0thOrder->getPointerToMeshDataTetraElement();
}

// Get pointer to the object of class MeshDataNonConformingHexaElement
const MeshDataNonConformingHexaElement *AnalysisControl::getPointerOfMeshDataNonConformingHexaElement() const
{
	if (m_ptrForward3DNonConformingHexaElement0thOrder == NULL)
	{
		OutputFiles::m_logFile << "Error : m_ptrForward3DNonConformingHexaElement0thOrder is NULL." << std::endl;
		exit(1);
	}
	return m_ptrForward3DNonConformingHexaElement0thOrder->getPointerToMeshDataNonConformingHexaElement();
}

void AnalysisControl::seticut(int value)
{
	icut = value;
}

int AnalysisControl::geticut() const
{
	return icut;
}

// Return flag specifing whether sensitivity is calculated or not
bool AnalysisControl::doesCalculateSensitivity(const int iter) const
{

	return (m_iterationNumMax > iter) ? true : false;
}

// Get pointer to the object of class Forward3D
Forward3D *AnalysisControl::getPointerOfForward3D() const
{

	if (m_typeOfMesh == MeshData::HEXA)
	{

		if (m_ptrForward3DBrickElement0thOrder != NULL)
		{
			return static_cast<Forward3D *>(m_ptrForward3DBrickElement0thOrder);
		}
		else
		{
			OutputFiles::m_logFile << "Error : m_ptrForward3DBrickElement0thOrderv is NULL." << std::endl;
			exit(1);
		}
	}
	else if (m_typeOfMesh == MeshData::TETRA)
	{

		if (m_ptrForward3DTetraElement0thOrder != NULL)
		{
			return static_cast<Forward3D *>(m_ptrForward3DTetraElement0thOrder);
		}
		else
		{
			OutputFiles::m_logFile << "Error : m_ptrForward3DTetraElement0thOrder is NULL." << std::endl;
			exit(1);
		}
	}
	else if (m_typeOfMesh == MeshData::NONCONFORMING_HEXA)
	{

		if (m_ptrForward3DNonConformingHexaElement0thOrder != NULL)
		{
			return static_cast<Forward3D *>(m_ptrForward3DNonConformingHexaElement0thOrder);
		}
		else
		{
			OutputFiles::m_logFile << "Error : m_ptrForward3DNonConformingHexaElement0thOrder is NULL." << std::endl;
			exit(1);
		}
	}
	else
	{
		OutputFiles::m_logFile << "Error : Type of mesh is wrong !! : " << m_typeOfMesh << "." << std::endl;
		exit(1);
	}
	return NULL;
}
