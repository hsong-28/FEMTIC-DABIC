/* -------------------------------------------------------------------------------------------------------
 * FEMTIC-DABIC L-curve diagnostic member definitions split from AnalysisControl.cpp.
 * This file owns cross-iteration diagnostics for linearized L-curve selection
 * and accepted nonlinear forward-response checks.
 * ------------------------------------------------------------------------------------------------------- */
#include "AnalysisControl.h"

#include <cmath>
#include <fstream>

namespace {

const char* kLCurveNonlinearCheckFile = "lcurve_nonlinear_check_summary.csv";

bool fileHasContent(const char* const fileName)
{
	std::ifstream input(fileName);
	return input.good() && input.peek() != std::ifstream::traits_type::eof();
}

bool isLargeNonlinearMisfitError(const double ratio)
{
	return std::isfinite(ratio) && (ratio > 2.0 || ratio < 0.5);
}

}

void AnalysisControl::recordLCurveSelectionDiagnostics(
	const int iteration,
	const std::string& mode,
	const std::string& roughnessOperator,
	const double selectedAlpha,
	const double selectedPredictedDataMisfit,
	const double selectedPredictedRms,
	const double selectedModelRoughness,
	const double maxCurvature,
	const std::string& failureIndicators)
{
	m_hasLCurveSelectionDiagnostics = true;
	m_lCurveSelectionIteration = iteration;
	m_lCurveModeName = mode;
	m_lCurveRoughnessOperator = roughnessOperator;
	m_lCurveFailureIndicators = failureIndicators;
	m_lCurveSelectedAlpha = selectedAlpha;
	m_lCurveFinalAlpha = selectedAlpha;
	m_lCurveSelectedPredictedDataMisfit = selectedPredictedDataMisfit;
	m_lCurveSelectedPredictedRms = selectedPredictedRms;
	m_lCurveSelectedModelRoughness = selectedModelRoughness;
	m_lCurveMaxCurvature = maxCurvature;
}

void AnalysisControl::setLCurveFinalTradeOffParameterForDiagnostics(const double finalAlpha)
{
	if( m_hasLCurveSelectionDiagnostics ){
		m_lCurveFinalAlpha = finalAlpha;
	}
}

void AnalysisControl::appendLCurveNonlinearCheckDiagnostics(
	const int iteration,
	const int retrial,
	const double actualDataMisfit,
	const double actualRms,
	const double acceptedModelRoughness,
	const double stepLengthFactorUsed)
{
	if( !m_hasLCurveSelectionDiagnostics ||
		iteration != m_lCurveSelectionIteration + 1 ){
		return;
	}

	const double nonlinearMisfitDiff =
		actualDataMisfit - m_lCurveSelectedPredictedDataMisfit;
	const double nonlinearMisfitRatio =
		m_lCurveSelectedPredictedDataMisfit > 0.0 ?
		actualDataMisfit / m_lCurveSelectedPredictedDataMisfit :
		-1.0;
	const bool largeNonlinearError =
		isLargeNonlinearMisfitError(nonlinearMisfitRatio);

	std::string failureIndicators = m_lCurveFailureIndicators;
	if( largeNonlinearError ){
		if( !failureIndicators.empty() ){
			failureIndicators += "|";
		}
		failureIndicators += "LARGE_NONLINEAR_ERROR";
	}
	if( retrial > 0 ){
		if( !failureIndicators.empty() ){
			failureIndicators += "|";
		}
		failureIndicators += "CUTBACK_CHANGED_MODEL";
	}

	const bool writeHeader = !fileHasContent(kLCurveNonlinearCheckFile);
	std::ofstream output(kLCurveNonlinearCheckFile, std::ios::app);
	if( writeHeader ){
		output
			<< "iteration,retrial,mode,roughness_operator,selected_alpha,final_alpha,"
			<< "predicted_data_misfit,actual_data_misfit,nonlinear_misfit_diff,"
			<< "nonlinear_misfit_ratio,predicted_rms,actual_rms,"
			<< "selected_model_roughness,accepted_model_roughness,max_curvature,"
			<< "step_length_factor_used,cutback_changed_model,"
			<< "bound_clipping_changed_model,large_nonlinear_error,"
			<< "failure_indicators\n";
	}
	output
		<< iteration << ","
		<< retrial << ","
		<< m_lCurveModeName << ","
		<< m_lCurveRoughnessOperator << ","
		<< m_lCurveSelectedAlpha << ","
		<< m_lCurveFinalAlpha << ","
		<< m_lCurveSelectedPredictedDataMisfit << ","
		<< actualDataMisfit << ","
		<< nonlinearMisfitDiff << ","
		<< nonlinearMisfitRatio << ","
		<< m_lCurveSelectedPredictedRms << ","
		<< actualRms << ","
		<< m_lCurveSelectedModelRoughness << ","
		<< acceptedModelRoughness << ","
		<< m_lCurveMaxCurvature << ","
		<< stepLengthFactorUsed << ","
		<< (retrial > 0 ? "yes" : "no") << ","
		<< "not_measured,"
		<< (largeNonlinearError ? "yes" : "no") << ","
		<< failureIndicators << "\n";

	m_hasLCurveSelectionDiagnostics = false;
}
