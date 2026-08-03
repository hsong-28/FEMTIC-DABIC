/* -------------------------------------------------------------------------------------------------------
 * FEMTIC-DABIC nonlinear L-curve member definitions split from AnalysisControl.cpp.
 * This file owns actual-response diagnostics for nonlinear cubic-spline L-curve selection.
 * ------------------------------------------------------------------------------------------------------- */
#include "AnalysisControl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "mpi.h"
#include "LCurveCubicSpline.h"
#include "ObservedData.h"
#include "OutputFiles.h"
#include "ResistivityBlock.h"

namespace {

const char* kNonlinearLCurveTrialSummaryFile = "lcurve_nonlinear_trial_summary.csv";
const char* kNonlinearLCurvePhase1SummaryFile = "lcurve_nonlinear_phase1_summary.csv";
const char* kNonlinearLCurveSelectionSummaryFile = "lcurve_nonlinear_selection_summary.csv";
const char* kNonlinearLCurveActualSplinePrefix = "lcurve_nonlinear_actual";
const double kNonlinearLCurveRiseTolerance = 0.01;
const double kNonlinearLCurveFlatSearchLogTolerance = 0.02;
const double kNonlinearLCurveAdaptiveFactor = 1.5;
const double kNonlinearLCurveAlphaLowerBound = 0.1;
const double kNonlinearLCurveAlphaUpperBound = 100.0;
const int kNonlinearLCurveFlatSearchMaxEvaluations = 4;
const int kNonlinearLCurveMinRetainedPointsForSpline = 6;
const char* kNonlinearLCurveSupportSamplingStrategy = "largest_log_interval_geometric_midpoint";
const char* kNonlinearLCurveConfiguredSchedule = "configured_first_iteration";
const char* kNonlinearLCurveAdaptiveSchedule = "adaptive_local_previous_selected_alpha_factor_1p5";
const char* kNonlinearLCurveConfiguredFallbackSchedule = "configured_fallback";

struct NonlinearLCurveActualPoint {
	NonlinearLCurveActualPoint():
		alpha(-1.0),
		dataMisfit(-1.0),
		rms(-1.0),
		modelRoughness(-1.0),
		elapsedSec(0.0)
	{
	}

	NonlinearLCurveActualPoint(
		const double alphaIn,
		const double dataMisfitIn,
		const double rmsIn,
		const double modelRoughnessIn,
		const double elapsedSecIn):
		alpha(alphaIn),
		dataMisfit(dataMisfitIn),
		rms(rmsIn),
		modelRoughness(modelRoughnessIn),
		elapsedSec(elapsedSecIn)
	{
	}

	double alpha;
	double dataMisfit;
	double rms;
	double modelRoughness;
	double elapsedSec;
};

bool fileHasContent(const char* const fileName)
{
	std::ifstream input(fileName);
	return input.good() && input.peek() != std::ifstream::traits_type::eof();
}

bool isPositiveFinite(const double value)
{
	return std::isfinite(value) && value > 0.0;
}

bool almostSameAlpha(const double lhs, const double rhs)
{
	if (!isPositiveFinite(lhs) || !isPositiveFinite(rhs))
	{
		return false;
	}
	const double scale = std::max(std::fabs(lhs), std::fabs(rhs));
	return std::fabs(lhs - rhs) <= std::max(1.0e-10, scale * 1.0e-8);
}

void appendFailureIndicator(std::string& indicators, const std::string& value)
{
	if (value.empty())
	{
		return;
	}
	if (!indicators.empty())
	{
		indicators += "|";
	}
	indicators += value;
}

void appendScheduleReason(std::string& reasons, const std::string& value)
{
	if (value.empty())
	{
		return;
	}
	std::istringstream input(reasons);
	std::string token;
	while (std::getline(input, token, '|'))
	{
		if (token == value)
		{
			return;
		}
	}
	if (!reasons.empty())
	{
		reasons += "|";
	}
	reasons += value;
}

std::string lcurveModeName(const bool useLogLog, const bool useRootNorm)
{
	if (useLogLog && useRootNorm)
	{
		return "LOG_ROOT";
	}
	if (useLogLog && !useRootNorm)
	{
		return "LOG_SQUARED";
	}
	if (!useLogLog && useRootNorm)
	{
		return "LINEAR_ROOT";
	}
	return "LINEAR_SQUARED";
}

std::string joinAlphaList(const std::vector<double>& alphas)
{
	std::ostringstream output;
	for (std::vector<double>::size_type i = 0; i < alphas.size(); ++i)
	{
		if (i > 0)
		{
			output << "|";
		}
		output << std::setprecision(12) << alphas[i];
	}
	return output.str();
}

bool appendUniqueAlpha(std::vector<double>& alphas, const double alpha)
{
	if (!isPositiveFinite(alpha))
	{
		return false;
	}
	const double clippedAlpha =
		std::max(kNonlinearLCurveAlphaLowerBound, std::min(kNonlinearLCurveAlphaUpperBound, alpha));
	for (std::vector<double>::const_iterator itr = alphas.begin(); itr != alphas.end(); ++itr)
	{
		if (almostSameAlpha(*itr, clippedAlpha))
		{
			return false;
		}
	}
	alphas.push_back(clippedAlpha);
	return true;
}

void sortAlphasDescending(std::vector<double>& alphas)
{
	std::sort(alphas.begin(), alphas.end(), std::greater<double>());
}

std::vector<double> buildLocalAlphaSearchSeed(const double centerAlpha)
{
	std::vector<double> alphas;
	if (!isPositiveFinite(centerAlpha))
	{
		return alphas;
	}

	appendUniqueAlpha(alphas, centerAlpha * kNonlinearLCurveAdaptiveFactor);
	appendUniqueAlpha(alphas, centerAlpha);
	appendUniqueAlpha(alphas, centerAlpha / kNonlinearLCurveAdaptiveFactor);
	sortAlphasDescending(alphas);
	return alphas;
}

std::string joinActualPointAlphaList(const std::vector<LCurveCubicSpline::Point>& points)
{
	std::vector<double> alphas;
	alphas.reserve(points.size());
	for (std::vector<LCurveCubicSpline::Point>::const_iterator itr = points.begin();
		itr != points.end(); ++itr)
	{
		alphas.push_back(itr->alpha);
	}
	std::sort(alphas.begin(), alphas.end(), std::greater<double>());
	return joinAlphaList(alphas);
}

std::string roughnessOperatorName(const char* const regularizationLabel)
{
	const std::string label(regularizationLabel == NULL ? "" : regularizationLabel);
	if (label.find("Laplacian") != std::string::npos)
	{
		return "laplacian";
	}
	return "difference";
}

void appendNonlinearLCurveTrialHeaderIfNeeded()
{
	const bool writeHeader = !fileHasContent(kNonlinearLCurveTrialSummaryFile);
	if (!writeHeader)
	{
		return;
	}
	std::ofstream output(kNonlinearLCurveTrialSummaryFile, std::ios::app);
	output
		<< "iteration,phase,trial_index,alpha,scan_direction,roughness_operator,mode,"
		<< "actual_data_misfit,actual_rms,model_roughness,is_best_so_far,"
		<< "rms_rise_ratio,rise_tolerance,role,stop_reason,"
		<< "forward_response_only,sensitivity_calculated,elapsed_sec\n";
}

void appendNonlinearLCurvePhase1HeaderIfNeeded()
{
	const bool writeHeader = !fileHasContent(kNonlinearLCurvePhase1SummaryFile);
	if (!writeHeader)
	{
		return;
	}
	std::ofstream output(kNonlinearLCurvePhase1SummaryFile, std::ios::app);
	output
		<< "iteration,roughness_operator,mode,scan_direction,rise_tolerance,"
		<< "num_candidate_alpha,num_trials_run,phase1_best_alpha,phase1_best_actual_rms,"
		<< "tail_start_alpha,tail_start_actual_rms,flat_bracket_alpha_high,"
		<< "flat_bracket_alpha_low,retained_alpha,alpha_list_descending,"
		<< "phase1_status,failure_indicators,alpha_schedule_source,"
		<< "alpha_schedule_reason,alpha_schedule_center,effective_alpha\n";
}

void appendNonlinearLCurveSelectionHeaderIfNeeded()
{
	const bool writeHeader = !fileHasContent(kNonlinearLCurveSelectionSummaryFile);
	if (!writeHeader)
	{
		return;
	}
	std::ofstream output(kNonlinearLCurveSelectionSummaryFile, std::ios::app);
	output
		<< "iteration,roughness_operator,mode,rise_tolerance,"
		<< "flat_search_log_tolerance,flat_search_max_evaluations,"
		<< "phase1_status,phase2_status,phase3_status,phase2_evaluation_count,"
		<< "retained_support_evaluation_count,min_retained_points_for_spline,"
		<< "retained_point_count,support_sampling_strategy,"
		<< "phase1_best_alpha,phase1_best_actual_rms,"
		<< "tail_start_alpha,tail_start_actual_rms,"
		<< "flat_point_alpha,flat_point_actual_rms,flat_point_data_misfit,"
		<< "flat_point_model_roughness,retained_alpha,"
		<< "selected_corner_alpha,selected_corner_actual_data_misfit,"
		<< "selected_corner_model_roughness,max_curvature,curvature_contrast,"
		<< "input_points,filtered_points,failure_indicators\n";
}

void appendNonlinearLCurveTrialRow(
	const int iteration,
	const char* const phase,
	const int trialIndex,
	const NonlinearLCurveActualPoint& point,
	const char* const scanDirection,
	const std::string& roughnessOperator,
	const std::string& mode,
	const bool isBestSoFar,
	const double rmsRiseRatio,
	const char* const role,
	const char* const stopReason)
{
	std::ofstream output(kNonlinearLCurveTrialSummaryFile, std::ios::app);
	output
		<< iteration << ","
		<< phase << ","
		<< trialIndex << ","
		<< std::setprecision(12) << point.alpha << ","
		<< scanDirection << ","
		<< roughnessOperator << ","
		<< mode << ","
		<< std::setprecision(12) << point.dataMisfit << ","
		<< std::setprecision(12) << point.rms << ","
		<< std::setprecision(12) << point.modelRoughness << ","
		<< (isBestSoFar ? "yes" : "no") << ","
		<< std::setprecision(12) << rmsRiseRatio << ","
		<< kNonlinearLCurveRiseTolerance << ","
		<< role << ","
		<< stopReason << ","
		<< "yes,"
		<< "no,"
		<< std::setprecision(12) << point.elapsedSec << "\n";
}

bool addUniqueActualSplinePoint(
	std::vector<LCurveCubicSpline::Point>& points,
	const NonlinearLCurveActualPoint& point)
{
	if (!isPositiveFinite(point.alpha) ||
		!isPositiveFinite(point.dataMisfit) ||
		!isPositiveFinite(point.modelRoughness))
	{
		return false;
	}
	for (std::vector<LCurveCubicSpline::Point>::const_iterator itr = points.begin();
		itr != points.end(); ++itr)
	{
		if (almostSameAlpha(itr->alpha, point.alpha))
		{
			return false;
		}
	}
	points.push_back(LCurveCubicSpline::Point(
		point.alpha,
		point.dataMisfit,
		point.modelRoughness));
	return true;
}

}

// Run actual-response diagnostics for nonlinear cubic-spline L-curve selection.
void AnalysisControl::runNonlinearLCurveDiagnostics(
	const int iter,
	const char* regularizationLabel)
{
	const int myProcessID = getMyPE();
	if (m_ptrInversiondataspace == NULL)
	{
		OutputFiles::m_logFile << "Error : Nonlinear L-curve data-space trial inversion object is NULL." << std::endl;
		exit(1);
	}
	if (m_NumOF_TO <= 0)
	{
		OutputFiles::m_logFile << "Error : Nonlinear L-curve diagnostics require at least one candidate trade-off parameter." << std::endl;
		exit(1);
	}

	ObservedData *const ptrObservedData = ObservedData::getInstance();
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	const std::string roughnessOperator = roughnessOperatorName(regularizationLabel);
	const std::string mode = lcurveModeName(getloglog(), getnorm());
	const double originalTradeOffParameter = m_tradeOffParameterForResistivityValue;
	m_stopAfterNonlinearLCurveDiagnostics = false;

	std::vector<double> configuredAlphas;
	configuredAlphas.reserve(static_cast<std::vector<double>::size_type>(m_NumOF_TO));
	for (int i = 0; i < m_NumOF_TO; ++i)
	{
		configuredAlphas.push_back(get_ithTradeOffParameterForResistivityValue(i));
	}
	std::vector<double> candidateAlphas = configuredAlphas;
	std::string alphaScheduleSource = kNonlinearLCurveConfiguredSchedule;
	std::string alphaScheduleReason = "global_start_alpha_sqrt10";
	double alphaScheduleCenter = -1.0;
	bool useLocalAlphaSearch = false;
	if (iter > m_iterationNumInit)
	{
		alphaScheduleCenter = originalTradeOffParameter;
		const std::vector<double> adaptiveAlphas =
			buildLocalAlphaSearchSeed(alphaScheduleCenter);
		if (adaptiveAlphas.size() >= 2)
		{
			candidateAlphas = adaptiveAlphas;
			alphaScheduleSource = kNonlinearLCurveAdaptiveSchedule;
			alphaScheduleReason = "centered";
			useLocalAlphaSearch = true;
		}
		else
		{
			alphaScheduleSource = kNonlinearLCurveConfiguredFallbackSchedule;
			alphaScheduleReason = "configured_fallback";
		}
	}

	if (!ptrObservedData->supportsSelectedTrialForwardResponseCache())
	{
		OutputFiles::m_logFile
			<< "Error : Nonlinear L-curve diagnostics currently require the selected-trial "
			<< "forward-response cache so that every alpha is evaluated from the same "
			<< "base residual state." << std::endl;
		exit(1);
	}

	const double baseDataMisfitThisPE = ptrObservedData->calculateErrorSumOfSquaresThisPE();
	double baseDataMisfit = 0.0;
	MPI_Reduce(&baseDataMisfitThisPE, &baseDataMisfit, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&baseDataMisfit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	const int numDataThisPEForDiagnostics = ptrObservedData->getNumObservedDataThisPETotal();
	int numDataTotalForDiagnostics = 0;
	MPI_Reduce(
		&numDataThisPEForDiagnostics,
		&numDataTotalForDiagnostics,
		1,
		MPI_INT,
		MPI_SUM,
		0,
		MPI_COMM_WORLD);
	MPI_Bcast(&numDataTotalForDiagnostics, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (myProcessID == 0)
	{
		std::cout << " # Entering nonlinear cubic-spline L-curve actual-response diagnostics ("
				  << regularizationLabel << ")." << std::endl;
		std::cout << " # Nonlinear L-curve mode: " << mode
				  << "; alpha scan direction: high_to_low"
				  << "; rise tolerance: " << kNonlinearLCurveRiseTolerance
				  << "; Phase II max evaluations: " << kNonlinearLCurveFlatSearchMaxEvaluations
				  << std::endl;
		std::cout << " # Nonlinear L-curve effective alpha schedule: source = "
				  << alphaScheduleSource
				  << "; reason = " << alphaScheduleReason
				  << "; center alpha = " << alphaScheduleCenter
				  << "; alphas = " << joinAlphaList(candidateAlphas)
				  << std::endl;
	}
	OutputFiles::m_logFile
		<< "# Nonlinear L-curve actual-response diagnostics start."
		<< " Roughness operator : " << roughnessOperator
		<< ", mode : " << mode
		<< ", rise tolerance : " << kNonlinearLCurveRiseTolerance
		<< ", Phase II max evaluations : " << kNonlinearLCurveFlatSearchMaxEvaluations
		<< "." << std::endl;
	OutputFiles::m_logFile
		<< "# Nonlinear L-curve diagnostics use forward responses only after each trial update;"
		<< " they do not calculate EM-field derivatives or sensitivity matrices." << std::endl;
	OutputFiles::m_logFile
		<< "# Nonlinear L-curve effective alpha schedule. Source : "
		<< alphaScheduleSource
		<< ", reason : " << alphaScheduleReason
		<< ", center alpha : " << alphaScheduleCenter
		<< ", alphas : " << joinAlphaList(candidateAlphas)
		<< "." << std::endl;

	ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
	ptrObservedData->copyDistortionParamsCurToPWK2();
	ptrObservedData->cacheSelectedTrialForwardResponse();

	const auto evaluateTrial = [&](
		const char* const phaseLabel,
		const int trialIndex,
		const double trialAlpha) -> NonlinearLCurveActualPoint
	{
		ptrResistivityBlock->copyPWK2NotFixedToPWK1();
		ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
		ptrObservedData->copyDistortionParamsPWK2ToPWK1();
		ptrObservedData->copyDistortionParamsPWK1ToCur();
		ptrObservedData->restoreSelectedTrialForwardResponseCache();
		m_tradeOffParameterForResistivityValue = trialAlpha;

		MPI_Barrier(MPI_COMM_WORLD);
		const double trialStartSec = MPI_Wtime();
		if (myProcessID == 0)
		{
			std::cout << " # Nonlinear L-curve " << phaseLabel
					  << " trial " << trialIndex
					  << ": alpha = " << trialAlpha << std::endl;
		}
		OutputFiles::m_logFile
			<< "# Nonlinear L-curve " << phaseLabel
			<< " trial " << trialIndex
			<< ", alpha : " << trialAlpha << "." << std::endl;

		m_ptrInversiondataspace->inversionCalculation();
		calcForwardResponseForNonlinearLCurveTrial(iter, trialIndex, trialAlpha);

		const double dataMisfitThisPE = ptrObservedData->calculateErrorSumOfSquaresThisPE();
		double actualDataMisfit = 0.0;
		MPI_Reduce(&dataMisfitThisPE, &actualDataMisfit, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		const int numDataThisPE = ptrObservedData->getNumObservedDataThisPETotal();
		int numDataTotal = 0;
		MPI_Reduce(&numDataThisPE, &numDataTotal, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

		double actualRms = 0.0;
		double modelRoughness = 0.0;
		if (myProcessID == 0)
		{
			actualRms = numDataTotal > 0 ? std::sqrt(actualDataMisfit / static_cast<double>(numDataTotal)) : -1.0;
			if (useDifferenceFilter())
			{
				modelRoughness = ptrResistivityBlock->calcModelRoughnessForDifferenceFilter();
			}
			else
			{
				modelRoughness = ptrResistivityBlock->calcModelRoughnessForLaplacianFilter();
			}
		}
		MPI_Bcast(&actualRms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&modelRoughness, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&actualDataMisfit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		const double trialElapsedSec = MPI_Wtime() - trialStartSec;
		return NonlinearLCurveActualPoint(
			trialAlpha,
			actualDataMisfit,
			actualRms,
			modelRoughness,
			trialElapsedSec);
	};

	double bestActualRms = std::numeric_limits<double>::max();
	NonlinearLCurveActualPoint bestPoint;
	NonlinearLCurveActualPoint tailStartPoint;
	int numTrialsRun = 0;
	bool tailDetected = false;
	bool alphaListDescending = true;
	std::vector<double> retainedAlphas;
	std::vector<NonlinearLCurveActualPoint> retainedPhase1Points;
	int localAlphaExpansionCount = 0;

	if (myProcessID == 0)
	{
		appendNonlinearLCurveTrialHeaderIfNeeded();
	}

	double previousAlpha = std::numeric_limits<double>::max();
	std::vector<double>::size_type i = 0;
	while (i < candidateAlphas.size())
	{
		const double trialAlpha = candidateAlphas[i];
		if (i > 0 && trialAlpha >= previousAlpha)
		{
			alphaListDescending = false;
		}
		previousAlpha = trialAlpha;

		const NonlinearLCurveActualPoint point =
			evaluateTrial("Phase I", i + 1, trialAlpha);

		const double previousBestActualRms = bestActualRms;
		const bool isBestSoFar = point.rms < bestActualRms;
		const double rmsRiseRatio =
			previousBestActualRms < std::numeric_limits<double>::max() ?
			point.rms / previousBestActualRms :
			1.0;
		bool stopAfterThisTrial = false;
		const char* role = "retained";
		const char* stopReason = "not_stopped";
		if (isBestSoFar)
		{
			bestActualRms = point.rms;
			bestPoint = point;
		}
		else if (point.rms > bestActualRms * (1.0 + kNonlinearLCurveRiseTolerance))
		{
			tailDetected = true;
			tailStartPoint = point;
			stopAfterThisTrial = true;
			role = "tail_start";
			stopReason = "actual_rms_rise_detected";
		}
		if (!stopAfterThisTrial)
		{
			retainedAlphas.push_back(trialAlpha);
			retainedPhase1Points.push_back(point);
		}

		if (myProcessID == 0)
		{
			appendNonlinearLCurveTrialRow(
				iter,
				"phase1",
				i + 1,
				point,
				"high_to_low",
				roughnessOperator,
				mode,
				isBestSoFar,
				rmsRiseRatio,
				role,
				stopReason);

			std::cout << " # Nonlinear L-curve Phase I result: alpha = " << trialAlpha
					  << "; actual RMS = " << point.rms
					  << "; roughness = " << point.modelRoughness
					  << "; role = " << role << std::endl;
		}

		++numTrialsRun;
		int stopFlag = stopAfterThisTrial ? 1 : 0;
		MPI_Bcast(&stopFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (stopFlag != 0)
		{
			break;
		}
		if (useLocalAlphaSearch && i + 1 >= candidateAlphas.size())
		{
			const double lowestScheduledAlpha = candidateAlphas.empty() ? -1.0 : candidateAlphas.back();
			if (isPositiveFinite(lowestScheduledAlpha) &&
				lowestScheduledAlpha > kNonlinearLCurveAlphaLowerBound * (1.0 + 1.0e-8))
			{
				const double expandedAlpha =
					std::max(kNonlinearLCurveAlphaLowerBound, lowestScheduledAlpha / kNonlinearLCurveAdaptiveFactor);
				if (appendUniqueAlpha(candidateAlphas, expandedAlpha))
				{
					++localAlphaExpansionCount;
					appendScheduleReason(alphaScheduleReason, "expanded_low");
					if (myProcessID == 0)
					{
						std::cout << " # Nonlinear L-curve local alpha search expands to lower alpha = "
								  << expandedAlpha << std::endl;
						OutputFiles::m_logFile
							<< "# Nonlinear L-curve local alpha search expands to lower alpha : "
							<< expandedAlpha << "." << std::endl;
					}
				}
				else
				{
					appendScheduleReason(alphaScheduleReason, "expansion_duplicate_or_clipped");
				}
			}
			else
			{
				appendScheduleReason(alphaScheduleReason, "lower_bound_reached");
			}
		}
		++i;
	}

	std::string failureIndicators;
	if (!alphaListDescending)
	{
		appendFailureIndicator(failureIndicators, "ALPHA_LIST_NOT_STRICTLY_DESCENDING");
	}
	if (!tailDetected)
	{
		appendFailureIndicator(failureIndicators, "TAIL_NOT_DETECTED");
	}
	const std::string phase1Status = tailDetected ? "tail_detected" : "tail_not_detected";

	if (myProcessID == 0)
	{
		appendNonlinearLCurvePhase1HeaderIfNeeded();
		const double flatBracketAlphaHigh = tailDetected ? bestPoint.alpha : -1.0;
		const double flatBracketAlphaLow = tailDetected ? tailStartPoint.alpha : -1.0;

		std::ofstream output(kNonlinearLCurvePhase1SummaryFile, std::ios::app);
		output
			<< iter << ","
			<< roughnessOperator << ","
			<< mode << ","
			<< "high_to_low,"
			<< kNonlinearLCurveRiseTolerance << ","
			<< candidateAlphas.size() << ","
			<< numTrialsRun << ","
			<< std::setprecision(12) << bestPoint.alpha << ","
			<< std::setprecision(12) << bestPoint.rms << ","
			<< std::setprecision(12) << tailStartPoint.alpha << ","
			<< std::setprecision(12) << tailStartPoint.rms << ","
			<< std::setprecision(12) << flatBracketAlphaHigh << ","
			<< std::setprecision(12) << flatBracketAlphaLow << ","
			<< joinAlphaList(retainedAlphas) << ","
			<< (alphaListDescending ? "yes" : "no") << ","
			<< phase1Status << ","
			<< failureIndicators << ","
			<< alphaScheduleSource << ","
			<< alphaScheduleReason << ","
			<< std::setprecision(12) << alphaScheduleCenter << ","
			<< joinAlphaList(candidateAlphas) << "\n";

		std::cout << " # Nonlinear L-curve Phase I completed: status = "
				  << phase1Status << "; phase1 best alpha = " << bestPoint.alpha
				  << "; best actual RMS = " << bestPoint.rms
				  << "; local alpha expansions = " << localAlphaExpansionCount << std::endl;
		std::cout << " # Nonlinear L-curve final effective alpha schedule: reason = "
				  << alphaScheduleReason
				  << "; alphas = " << joinAlphaList(candidateAlphas) << std::endl;
		if (tailDetected)
		{
			std::cout << " # Nonlinear L-curve flat-point bracket for Phase II: high alpha = "
					  << flatBracketAlphaHigh << "; low alpha = " << flatBracketAlphaLow << std::endl;
		}
	}

	std::vector<NonlinearLCurveActualPoint> phase2Points;
	std::vector<NonlinearLCurveActualPoint> retainedSupportPoints;
	NonlinearLCurveActualPoint flatPoint = bestPoint;
	std::string phase2Status = "not_run";
	int phase2Evaluations = 0;
	int retainedSupportEvaluations = 0;

	if (tailDetected)
	{
		if (!isPositiveFinite(bestPoint.alpha) ||
			!isPositiveFinite(tailStartPoint.alpha) ||
			bestPoint.alpha <= tailStartPoint.alpha)
		{
			phase2Status = "invalid_bracket";
			appendFailureIndicator(failureIndicators, "INVALID_FLAT_POINT_BRACKET");
		}
		else
		{
			phase2Status = "completed";
			double xLower = std::log(tailStartPoint.alpha);
			double xUpper = std::log(bestPoint.alpha);
			const double goldenRatioConjugate = 0.6180339887498948482;

			const auto evaluatePhase2 = [&](
				const double logAlpha) -> NonlinearLCurveActualPoint
			{
				++phase2Evaluations;
				const double trialAlpha = std::exp(logAlpha);
				const NonlinearLCurveActualPoint point =
					evaluateTrial("Phase II", numTrialsRun + phase2Evaluations, trialAlpha);
				const bool isBestSoFar = point.rms < flatPoint.rms;
				const double rmsRiseRatio =
					isPositiveFinite(flatPoint.rms) ? point.rms / flatPoint.rms : 1.0;
				if (isBestSoFar)
				{
					flatPoint = point;
				}
				phase2Points.push_back(point);

				if (myProcessID == 0)
				{
					appendNonlinearLCurveTrialRow(
						iter,
						"phase2",
						numTrialsRun + phase2Evaluations,
						point,
						"bracketed_log_alpha",
						roughnessOperator,
						mode,
						isBestSoFar,
						rmsRiseRatio,
						isBestSoFar ? "flat_point_candidate" : "flat_search",
						"not_stopped");

					std::cout << " # Nonlinear L-curve Phase II result: alpha = " << trialAlpha
							  << "; actual RMS = " << point.rms
							  << "; roughness = " << point.modelRoughness
							  << "; role = " << (isBestSoFar ? "flat_point_candidate" : "flat_search")
							  << std::endl;
				}
				return point;
			};

			double x1 = xUpper - goldenRatioConjugate * (xUpper - xLower);
			double x2 = xLower + goldenRatioConjugate * (xUpper - xLower);
			NonlinearLCurveActualPoint point1 = evaluatePhase2(x1);
			NonlinearLCurveActualPoint point2 = evaluatePhase2(x2);
			while (phase2Evaluations < kNonlinearLCurveFlatSearchMaxEvaluations &&
				std::fabs(xUpper - xLower) > kNonlinearLCurveFlatSearchLogTolerance)
			{
				if (point1.rms > point2.rms)
				{
					xLower = x1;
					x1 = x2;
					point1 = point2;
					x2 = xLower + goldenRatioConjugate * (xUpper - xLower);
					point2 = evaluatePhase2(x2);
				}
				else
				{
					xUpper = x2;
					x2 = x1;
					point2 = point1;
					x1 = xUpper - goldenRatioConjugate * (xUpper - xLower);
					point1 = evaluatePhase2(x1);
				}
			}
		}
	}

	const auto rebuildRetainedActualSplinePoints = [&](
		std::vector<LCurveCubicSpline::Point>& points)
	{
		points.clear();
		for (std::vector<NonlinearLCurveActualPoint>::const_iterator itr = retainedPhase1Points.begin();
			itr != retainedPhase1Points.end(); ++itr)
		{
			if (itr->alpha + std::max(1.0e-10, std::fabs(flatPoint.alpha) * 1.0e-8) >= flatPoint.alpha)
			{
				addUniqueActualSplinePoint(points, *itr);
			}
		}
		for (std::vector<NonlinearLCurveActualPoint>::const_iterator itr = phase2Points.begin();
			itr != phase2Points.end(); ++itr)
		{
			if (itr->alpha + std::max(1.0e-10, std::fabs(flatPoint.alpha) * 1.0e-8) >= flatPoint.alpha)
			{
				addUniqueActualSplinePoint(points, *itr);
			}
		}
		for (std::vector<NonlinearLCurveActualPoint>::const_iterator itr = retainedSupportPoints.begin();
			itr != retainedSupportPoints.end(); ++itr)
		{
			if (itr->alpha + std::max(1.0e-10, std::fabs(flatPoint.alpha) * 1.0e-8) >= flatPoint.alpha)
			{
				addUniqueActualSplinePoint(points, *itr);
			}
		}
		addUniqueActualSplinePoint(points, flatPoint);
	};

	std::vector<LCurveCubicSpline::Point> retainedActualSplinePoints;
	rebuildRetainedActualSplinePoints(retainedActualSplinePoints);

	while (tailDetected &&
		phase2Status == "completed" &&
		retainedActualSplinePoints.size() < static_cast<std::vector<LCurveCubicSpline::Point>::size_type>(
			kNonlinearLCurveMinRetainedPointsForSpline))
	{
		std::vector<double> sortedAlphas;
		sortedAlphas.reserve(retainedActualSplinePoints.size());
		for (std::vector<LCurveCubicSpline::Point>::const_iterator itr = retainedActualSplinePoints.begin();
			itr != retainedActualSplinePoints.end(); ++itr)
		{
			if (isPositiveFinite(itr->alpha))
			{
				sortedAlphas.push_back(itr->alpha);
			}
		}
		std::sort(sortedAlphas.begin(), sortedAlphas.end());
		sortedAlphas.erase(
			std::unique(sortedAlphas.begin(), sortedAlphas.end(), almostSameAlpha),
			sortedAlphas.end());
		if (sortedAlphas.size() < 2)
		{
			break;
		}

		double lowerAlphaForSupport = -1.0;
		double upperAlphaForSupport = -1.0;
		double widestLogInterval = -1.0;
		for (std::vector<double>::size_type iAlpha = 0; iAlpha + 1 < sortedAlphas.size(); ++iAlpha)
		{
			if (sortedAlphas[iAlpha] + std::max(1.0e-10, std::fabs(flatPoint.alpha) * 1.0e-8) < flatPoint.alpha)
			{
				continue;
			}
			const double logInterval = std::log(sortedAlphas[iAlpha + 1]) - std::log(sortedAlphas[iAlpha]);
			if (logInterval > widestLogInterval)
			{
				widestLogInterval = logInterval;
				lowerAlphaForSupport = sortedAlphas[iAlpha];
				upperAlphaForSupport = sortedAlphas[iAlpha + 1];
			}
		}
		if (!isPositiveFinite(lowerAlphaForSupport) ||
			!isPositiveFinite(upperAlphaForSupport) ||
			widestLogInterval <= kNonlinearLCurveFlatSearchLogTolerance)
		{
			break;
		}

		++retainedSupportEvaluations;
		const double supportAlpha = std::sqrt(lowerAlphaForSupport * upperAlphaForSupport);
		const NonlinearLCurveActualPoint point =
			evaluateTrial(
				"Phase III retained-support",
				numTrialsRun + phase2Evaluations + retainedSupportEvaluations,
				supportAlpha);
		const bool supportImprovesFlatPoint = point.rms < flatPoint.rms;
		if (supportImprovesFlatPoint)
		{
			flatPoint = point;
		}
		retainedSupportPoints.push_back(point);

		if (myProcessID == 0)
		{
			appendNonlinearLCurveTrialRow(
				iter,
				"phase3_support",
				numTrialsRun + phase2Evaluations + retainedSupportEvaluations,
				point,
				"retained_log_midpoint",
				roughnessOperator,
				mode,
				supportImprovesFlatPoint,
				isPositiveFinite(flatPoint.rms) ? point.rms / flatPoint.rms : 1.0,
				supportImprovesFlatPoint ? "retained_support_improved_flat" : "retained_support",
				"not_stopped");

			std::cout << " # Nonlinear L-curve Phase III retained-support result: alpha = "
					  << supportAlpha << "; actual RMS = " << point.rms
					  << "; roughness = " << point.modelRoughness
					  << "; role = "
					  << (supportImprovesFlatPoint ? "retained_support_improved_flat" : "retained_support")
					  << std::endl;
		}

		rebuildRetainedActualSplinePoints(retainedActualSplinePoints);
	}

	std::string phase3Status = "not_run";
	LCurveCubicSpline::Result splineResult;
	if (tailDetected && phase2Status == "completed")
	{
		if (retainedActualSplinePoints.size() < static_cast<std::vector<LCurveCubicSpline::Point>::size_type>(
			kNonlinearLCurveMinRetainedPointsForSpline))
		{
			phase3Status = "insufficient_retained_points";
			appendFailureIndicator(failureIndicators, "INSUFFICIENT_RETAINED_POINTS_FOR_SPLINE");
		}
		else
		{
			if (myProcessID == 0)
			{
				LCurveCubicSpline::Options options;
				options.useLogLog = getloglog();
				options.useRootNorm = getnorm();
				options.diagnosticPrefix = kNonlinearLCurveActualSplinePrefix;
				options.roughnessOperator = roughnessOperator;

				LCurveCubicSpline splineSelector;
				splineResult = splineSelector.selectAlpha(
					retainedActualSplinePoints,
					baseDataMisfit,
					iter,
					options);
				if (splineResult.selected)
				{
					phase3Status = "selected";
				}
				else
				{
					phase3Status = "spline_selection_failed";
					appendFailureIndicator(failureIndicators, splineResult.failureIndicators);
				}
			}
		}
	}

	if (myProcessID == 0)
	{
		appendNonlinearLCurveSelectionHeaderIfNeeded();
		std::ofstream output(kNonlinearLCurveSelectionSummaryFile, std::ios::app);
		output
			<< iter << ","
			<< roughnessOperator << ","
			<< mode << ","
			<< kNonlinearLCurveRiseTolerance << ","
			<< kNonlinearLCurveFlatSearchLogTolerance << ","
			<< kNonlinearLCurveFlatSearchMaxEvaluations << ","
			<< phase1Status << ","
			<< phase2Status << ","
			<< phase3Status << ","
			<< phase2Evaluations << ","
			<< retainedSupportEvaluations << ","
			<< kNonlinearLCurveMinRetainedPointsForSpline << ","
			<< retainedActualSplinePoints.size() << ","
			<< kNonlinearLCurveSupportSamplingStrategy << ","
			<< std::setprecision(12) << bestPoint.alpha << ","
			<< std::setprecision(12) << bestPoint.rms << ","
			<< std::setprecision(12) << tailStartPoint.alpha << ","
			<< std::setprecision(12) << tailStartPoint.rms << ","
			<< std::setprecision(12) << flatPoint.alpha << ","
			<< std::setprecision(12) << flatPoint.rms << ","
			<< std::setprecision(12) << flatPoint.dataMisfit << ","
			<< std::setprecision(12) << flatPoint.modelRoughness << ","
			<< joinActualPointAlphaList(retainedActualSplinePoints) << ","
			<< std::setprecision(12) << splineResult.selectedAlpha << ","
			<< std::setprecision(12) << splineResult.selectedPredictedDataMisfit << ","
			<< std::setprecision(12) << splineResult.selectedModelRoughness << ","
			<< std::setprecision(12) << splineResult.maxCurvature << ","
			<< std::setprecision(12) << splineResult.curvatureContrast << ","
			<< splineResult.inputPointCount << ","
			<< splineResult.filteredPointCount << ","
			<< failureIndicators << "\n";

		std::cout << " # Nonlinear L-curve Phase II completed: status = "
				  << phase2Status << "; flat alpha = " << flatPoint.alpha
				  << "; flat actual RMS = " << flatPoint.rms
				  << "; Phase II evaluations = " << phase2Evaluations << std::endl;
		std::cout << " # Nonlinear L-curve retained-support evaluations = "
				  << retainedSupportEvaluations
				  << "; minimum retained points for spline = "
				  << kNonlinearLCurveMinRetainedPointsForSpline
				  << "; support strategy = "
				  << kNonlinearLCurveSupportSamplingStrategy << std::endl;
		std::cout << " # Nonlinear L-curve retained actual points for spline: "
				  << joinActualPointAlphaList(retainedActualSplinePoints) << std::endl;
		if (splineResult.selected)
		{
			std::cout << " # Nonlinear L-curve retained actual spline corner alpha = "
					  << splineResult.selectedAlpha
					  << "; actual data misfit = " << splineResult.selectedPredictedDataMisfit
					  << "; model roughness = " << splineResult.selectedModelRoughness << std::endl;
		}
		else
		{
			std::cout << " # Nonlinear L-curve retained actual spline corner not selected: "
					  << phase3Status << "; indicators = " << failureIndicators << std::endl;
		}
	}

	int selectedFlag = 0;
	double selectedAlpha = -1.0;
	double selectedActualDataMisfit = -1.0;
	double selectedActualRms = -1.0;
	double selectedModelRoughness = -1.0;
	double selectedMaxCurvature = -1.0;
	if (myProcessID == 0 && splineResult.selected)
	{
		selectedFlag = 1;
		selectedAlpha = splineResult.selectedAlpha;
		selectedActualDataMisfit = splineResult.selectedPredictedDataMisfit;
		selectedActualRms =
			numDataTotalForDiagnostics > 0 ?
			std::sqrt(selectedActualDataMisfit / static_cast<double>(numDataTotalForDiagnostics)) :
			-1.0;
		selectedModelRoughness = splineResult.selectedModelRoughness;
		selectedMaxCurvature = splineResult.maxCurvature;
		recordLCurveSelectionDiagnostics(
			iter,
			mode,
			roughnessOperator,
			selectedAlpha,
			selectedActualDataMisfit,
			selectedActualRms,
			selectedModelRoughness,
			selectedMaxCurvature,
			failureIndicators);
		setLCurveFinalTradeOffParameterForDiagnostics(selectedAlpha);
	}
	MPI_Bcast(&selectedFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&selectedAlpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&selectedActualDataMisfit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&selectedActualRms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&selectedModelRoughness, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&selectedMaxCurvature, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	ptrResistivityBlock->copyPWK2NotFixedToPWK1();
	ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
	ptrObservedData->copyDistortionParamsPWK2ToPWK1();
	ptrObservedData->copyDistortionParamsPWK1ToCur();
	ptrObservedData->restoreSelectedTrialForwardResponseCache();

	if (selectedFlag != 0)
	{
		m_tradeOffParameterForResistivityValue = selectedAlpha;
		if (myProcessID == 0)
		{
			std::cout << " # Final nonlinear L-curve trade-off parameter : "
					  << m_tradeOffParameterForResistivityValue
					  << " (accepted model update will follow)." << std::endl;
		}
	}
	else
	{
		m_tradeOffParameterForResistivityValue = originalTradeOffParameter;
		m_stopAfterNonlinearLCurveDiagnostics = true;
	}

	OutputFiles::m_logFile
		<< "# Nonlinear cubic-spline L-curve diagnostics completed. "
		<< "Phase II flat-point search and retained actual spline-corner selection are enabled; "
		<< (selectedFlag != 0 ?
			"the selected alpha is passed to the accepted data-space model update." :
			"no selected alpha is available, so the inversion loop is stopped.")
		<< std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	return;
}
