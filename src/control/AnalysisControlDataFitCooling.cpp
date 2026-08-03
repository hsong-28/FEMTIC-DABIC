/* -------------------------------------------------------------------------------------------------------
 * FEMTIC-DABIC data-fit-bracketed cooling member definitions.
 * This file owns the standalone initial-alpha scan and its forward-only trial boundary.
 * ------------------------------------------------------------------------------------------------------- */
#include "AnalysisControl.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "DataFitCoolingPolicy.h"
#include "Forward3D.h"
#include "ObservedData.h"
#include "OutputFiles.h"
#include "ResistivityBlock.h"
#include "mpi.h"

namespace {

const double kInitialBracketLog10Span = 0.25;
const int kMaximumInitialBracketEvaluations = 32;
const char* kCoolingTrialSummaryFile = "cooling_trial_summary.csv";

bool fileHasContent(const char* const fileName)
{
	std::ifstream input(fileName, std::ios::binary | std::ios::ate);
	return input.is_open() && input.tellg() > 0;
}

}

// Append one actual forward trial to the maintained cooling diagnostic CSV
void AnalysisControl::appendDataFitCoolingTrialSummary(
	const int iteration,
	const int trialIndex,
	const char* const alphaSource,
	const double trialAlpha,
	const double acceptedAlpha,
	const double nextAlpha,
	const double previousRms,
	const double trialRms,
	const bool coolingTriggered,
	const double stepLength,
	const int cutbackCount,
	const char* const acceptanceStatus,
	const char* const terminationReason) const
{
	if (getMyPE() != 0)
	{
		return;
	}
	const bool writeHeader = !fileHasContent(kCoolingTrialSummaryFile);
	std::ofstream output(kCoolingTrialSummaryFile, std::ios::app);
	if (!output.is_open())
	{
		OutputFiles::m_logFile << "Error : Cannot open " << kCoolingTrialSummaryFile << "." << std::endl;
		exit(1);
	}
	if (writeHeader)
	{
		output
			<< "iteration,trial_index,alpha_source,initial_alpha,initial_log10_alpha,trial_log10_alpha,bracket_log10_span,trial_alpha,accepted_alpha,next_alpha,minimum_alpha,previous_rms,trial_rms,relative_rms_decrease,initial_rms_decrease_threshold,cooling_trigger_threshold,cooling_factor,cooling_triggered,cooling_count,step_length,cutback_count,trial_acceptance_status,termination_reason\n";
	}
	output << std::setprecision(12)
		<< iteration << ","
		<< trialIndex << ","
		<< alphaSource << ","
		<< m_dataFitCoolingInitialAlpha << ","
		<< std::log10(m_dataFitCoolingInitialAlpha) << ","
		<< std::log10(trialAlpha) << ","
		<< kInitialBracketLog10Span << ","
		<< trialAlpha << ","
		<< acceptedAlpha << ","
		<< nextAlpha << ","
		<< m_dataFitCoolingMinimumAlpha << ","
		<< previousRms << ","
		<< trialRms << ","
		<< DataFitCoolingPolicy::relativeRmsDecrease(previousRms, trialRms) << ","
		<< m_dataFitCoolingInitialRmsDecreaseThreshold << ","
		<< m_dataFitCoolingTriggerThreshold << ","
		<< m_dataFitCoolingFactor << ","
		<< (coolingTriggered ? "yes" : "no") << ","
		<< m_dataFitCoolingCount << ","
		<< stepLength << ","
		<< cutbackCount << ","
		<< acceptanceStatus << ","
		<< terminationReason << "\n";
}

// Return whether the standalone data-fit cooling method is active
bool AnalysisControl::isDataFitCoolingMode() const
{
	return m_inversionMethod == Inversion::DATA_FIT_COOLING_DATA_SPECE &&
		m_typeOfTradeOffParam == AnalysisControl::TO_DATA_FIT_COOLING;
}

// Calculate the global RMS represented by the current response state
double AnalysisControl::calculateCurrentGlobalRms() const
{
	ObservedData* const ptrObservedData = ObservedData::getInstance();
	const double localMisfit = ptrObservedData->calculateErrorSumOfSquaresThisPE();
	const int localDataCount = ptrObservedData->getNumObservedDataThisPETotal();
	double globalMisfit = 0.0;
	int globalDataCount = 0;
	MPI_Allreduce(&localMisfit, &globalMisfit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&localDataCount, &globalDataCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if (globalDataCount <= 0 || !std::isfinite(globalMisfit) || globalMisfit < 0.0)
	{
		OutputFiles::m_logFile << "Error : Cannot calculate a finite data-fit cooling RMS." << std::endl;
		exit(1);
	}
	return std::sqrt(globalMisfit / static_cast<double>(globalDataCount));
}

// Evaluate one cooling candidate using a full forward response without sensitivity work
void AnalysisControl::runDataFitCoolingTrial(
	const double alpha,
	const int trialIndex,
	const char* const alphaSource)
{
	ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData* const ptrObservedData = ObservedData::getInstance();
	Forward3D* const ptrForward3D = getPointerOfForward3D();
	if (m_ptrInversion == NULL || ptrForward3D == NULL)
	{
		OutputFiles::m_logFile << "Error : Data-fit cooling trial dependencies are not initialized." << std::endl;
		exit(1);
	}

	ptrResistivityBlock->copyPWK2NotFixedToPWK1();
	ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
	ptrObservedData->copyDistortionParamsPWK2ToPWK1();
	ptrObservedData->copyDistortionParamsPWK1ToCur();
	ptrObservedData->restoreSelectedTrialForwardResponseCache();
	m_tradeOffParameterForResistivityValue = alpha;

	if (getMyPE() == 0)
	{
		std::cout << " # Data-fit cooling " << alphaSource
			<< " trial " << trialIndex << ": alpha = " << alpha << std::endl;
	}
	OutputFiles::m_logFile
		<< "# Data-fit cooling trial. Source : " << alphaSource
		<< ", trial : " << trialIndex
		<< ", alpha : " << alpha
		<< ", sensitivity_in_trial : no." << std::endl;

	m_ptrInversion->inversionCalculation();

	const int numFrequencies = ptrObservedData->getNumOfFrequenciesCalculatedByThisPE();
	for (int iFreq = 0; iFreq < numFrequencies; ++iFreq)
	{
		const double frequency = ptrObservedData->getValuesOfFrequenciesCalculatedByThisPE(iFreq);
		for (int iPol = 0; iPol < 2; ++iPol)
		{
			ptrForward3D->forwardCalculation(frequency, iPol);
			ptrObservedData->calculateEMFieldOfAllStations(ptrForward3D, frequency, iPol, iFreq);
		}
		ptrObservedData->calculateResponseFunctionOfAllStations(iFreq);
	}
	if (!m_holdMemoryForwardSolver)
	{
		ptrForward3D->releaseMemoryOfMatrixAndSolver();
	}

	m_dataFitCoolingTrialRms = calculateCurrentGlobalRms();
	OutputFiles::m_logFile
		<< "# Data-fit cooling trial result. Source : " << alphaSource
		<< ", trial : " << trialIndex
		<< ", alpha : " << alpha
		<< ", actual RMS : " << m_dataFitCoolingTrialRms << "." << std::endl;
}

// Select the largest tested alpha that produces a meaningful RMS decrease
bool AnalysisControl::runInitialDataFitCoolingBracket()
{
	if (!isDataFitCoolingMode())
	{
		OutputFiles::m_logFile << "Error : Initial data-fit cooling bracket called outside method 6/type 5." << std::endl;
		exit(1);
	}

	ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData* const ptrObservedData = ObservedData::getInstance();
	m_dataFitCoolingPreviousAcceptedRms = calculateCurrentGlobalRms();
	ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
	ptrObservedData->copyDistortionParamsCurToPWK2();
	ptrObservedData->cacheSelectedTrialForwardResponse();
	if (getMyPE() == 0)
	{
		std::cout << " # Data-fit cooling initial bracket: alpha_start = "
			<< m_dataFitCoolingInitialAlpha
			<< "; log10(alpha_start) = " << std::log10(m_dataFitCoolingInitialAlpha)
			<< "; log10 span = " << kInitialBracketLog10Span
			<< "; required relative RMS decrease = " << m_dataFitCoolingInitialRmsDecreaseThreshold
			<< "; minimum alpha = " << m_dataFitCoolingMinimumAlpha << std::endl;
	}
	OutputFiles::m_logFile
		<< "# Data-fit cooling initial bracket parameters. Alpha start : "
		<< m_dataFitCoolingInitialAlpha
		<< ", log10(alpha start) : " << std::log10(m_dataFitCoolingInitialAlpha)
		<< ", log10 span : " << kInitialBracketLog10Span
		<< ", required relative RMS decrease : " << m_dataFitCoolingInitialRmsDecreaseThreshold
		<< ", minimum alpha : " << m_dataFitCoolingMinimumAlpha
		<< ", maximum bracket evaluations : " << kMaximumInitialBracketEvaluations
		<< "." << std::endl;

	const auto restoreBaseState = [&]()
	{
		ptrResistivityBlock->copyPWK2NotFixedToPWK1();
		ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
		ptrObservedData->copyDistortionParamsPWK2ToPWK1();
		ptrObservedData->copyDistortionParamsPWK1ToCur();
		ptrObservedData->restoreSelectedTrialForwardResponseCache();
	};
	const auto isSuccessfulTrial = [&]()
	{
		return DataFitCoolingPolicy::isMeaningfulRmsDecrease(
			m_dataFitCoolingPreviousAcceptedRms,
			m_dataFitCoolingTrialRms,
			m_dataFitCoolingInitialRmsDecreaseThreshold);
	};
	const auto stopInitialSearch = [&](const char* const reason)
	{
		restoreBaseState();
		m_tradeOffParameterForResistivityValue = m_dataFitCoolingInitialAlpha;
		OutputFiles::m_logFile
			<< "# Data-fit cooling initial bracket stopped. termination_reason : "
			<< reason << "." << std::endl;
		return false;
	};

	double selectedAlpha = -1.0;
	double selectedRms = -1.0;
	const double startLog10Alpha = std::log10(m_dataFitCoolingInitialAlpha);
	double log10Alpha = startLog10Alpha;
	const double minimumLog10Alpha = std::log10(m_dataFitCoolingMinimumAlpha);
	double alpha = std::pow(10.0, log10Alpha);
	int trialIndex = 0;
	runDataFitCoolingTrial(alpha, trialIndex, "initial_bracket");
	const double startRms = m_dataFitCoolingTrialRms;
	const bool initialTrialSucceeded = isSuccessfulTrial();
	bool mustExpandUpward = false;
	const double firstNextLog10Alpha = initialTrialSucceeded
		? DataFitCoolingPolicy::nextHigherBracketLog10Alpha(
			startLog10Alpha, kInitialBracketLog10Span)
		: DataFitCoolingPolicy::nextLowerBracketLog10Alpha(
			startLog10Alpha, minimumLog10Alpha, kInitialBracketLog10Span);
	appendDataFitCoolingTrialSummary(
		m_iterationNumCurrent, trialIndex, "initial_bracket", alpha,
		initialTrialSucceeded ? alpha : -1.0,
		std::pow(10.0, firstNextLog10Alpha),
		m_dataFitCoolingPreviousAcceptedRms, m_dataFitCoolingTrialRms,
		false, m_stepLengthDampingFactorCur, 0,
		initialTrialSucceeded ? "candidate" : "rejected",
		initialTrialSucceeded ? "search_upward" : "probe_downward");

	if (initialTrialSucceeded)
	{
		selectedAlpha = alpha;
		selectedRms = m_dataFitCoolingTrialRms;
		mustExpandUpward = true;
	}
	else
	{
		double previousRms = startRms;
		bool reverseToUpward = false;
		bool firstDownwardTrial = true;
		while (selectedAlpha <= 0.0)
		{
			if (log10Alpha <= minimumLog10Alpha + 1.0e-12 ||
				trialIndex + 1 >= kMaximumInitialBracketEvaluations)
			{
				return stopInitialSearch("no_acceptable_initial_alpha");
			}
			const double nextLog10Alpha = DataFitCoolingPolicy::nextLowerBracketLog10Alpha(
				log10Alpha, minimumLog10Alpha, kInitialBracketLog10Span);
			alpha = std::pow(10.0, nextLog10Alpha);
			++trialIndex;
			const char* const source = firstDownwardTrial ? "probe_downward" : "search_downward";
			runDataFitCoolingTrial(alpha, trialIndex, source);
			if (isSuccessfulTrial())
			{
				selectedAlpha = alpha;
				selectedRms = m_dataFitCoolingTrialRms;
				appendDataFitCoolingTrialSummary(
					m_iterationNumCurrent, trialIndex, source, alpha, alpha, alpha,
					m_dataFitCoolingPreviousAcceptedRms, m_dataFitCoolingTrialRms,
					false, m_stepLengthDampingFactorCur, 0,
					"candidate", "lower_success_found");
				break;
			}
			const bool reverse = DataFitCoolingPolicy::shouldReverseInitialBracketSearch(
				previousRms, m_dataFitCoolingTrialRms);
			const double nextAlpha = reverse
				? std::pow(10.0, DataFitCoolingPolicy::nextHigherBracketLog10Alpha(
					startLog10Alpha, kInitialBracketLog10Span))
				: std::pow(10.0, DataFitCoolingPolicy::nextLowerBracketLog10Alpha(
					nextLog10Alpha, minimumLog10Alpha, kInitialBracketLog10Span));
			appendDataFitCoolingTrialSummary(
				m_iterationNumCurrent, trialIndex, source, alpha, -1.0,
				nextAlpha,
				m_dataFitCoolingPreviousAcceptedRms, m_dataFitCoolingTrialRms,
				false, m_stepLengthDampingFactorCur, 0, "rejected",
				reverse ? "reverse_to_upward" : "search_downward");
			if (reverse)
			{
				reverseToUpward = true;
				break;
			}
			previousRms = m_dataFitCoolingTrialRms;
			log10Alpha = nextLog10Alpha;
			firstDownwardTrial = false;
		}

		if (reverseToUpward)
		{
			log10Alpha = startLog10Alpha;
			previousRms = startRms;
			bool firstUpwardTrial = true;
			while (selectedAlpha <= 0.0)
			{
				if (trialIndex + 1 >= kMaximumInitialBracketEvaluations)
				{
					return stopInitialSearch("no_acceptable_initial_alpha");
				}
				log10Alpha = DataFitCoolingPolicy::nextHigherBracketLog10Alpha(
					log10Alpha, kInitialBracketLog10Span);
				alpha = std::pow(10.0, log10Alpha);
				++trialIndex;
				const char* const source = firstUpwardTrial ? "reverse_to_upward" : "search_upward";
				runDataFitCoolingTrial(alpha, trialIndex, source);
				if (isSuccessfulTrial())
				{
					selectedAlpha = alpha;
					selectedRms = m_dataFitCoolingTrialRms;
					mustExpandUpward = true;
					appendDataFitCoolingTrialSummary(
						m_iterationNumCurrent, trialIndex, source, alpha, alpha, alpha,
						m_dataFitCoolingPreviousAcceptedRms, m_dataFitCoolingTrialRms,
						false, m_stepLengthDampingFactorCur, 0,
						"candidate", "upper_success_found");
					break;
				}
				const bool upwardTurned = DataFitCoolingPolicy::shouldReverseInitialBracketSearch(
					previousRms, m_dataFitCoolingTrialRms);
				const double nextAlpha = upwardTurned ? alpha : std::pow(
					10.0, DataFitCoolingPolicy::nextHigherBracketLog10Alpha(
						log10Alpha, kInitialBracketLog10Span));
				appendDataFitCoolingTrialSummary(
					m_iterationNumCurrent, trialIndex, source, alpha, -1.0, nextAlpha,
					m_dataFitCoolingPreviousAcceptedRms, m_dataFitCoolingTrialRms,
					false, m_stepLengthDampingFactorCur, 0, "rejected",
					upwardTurned ? "upward_turning_point_before_success" : "search_upward");
				if (upwardTurned)
				{
					return stopInitialSearch("no_acceptable_initial_alpha_near_local_minimum");
				}
				previousRms = m_dataFitCoolingTrialRms;
				firstUpwardTrial = false;
			}
		}
	}

	while (mustExpandUpward)
	{
		if (trialIndex + 1 >= kMaximumInitialBracketEvaluations)
		{
			appendDataFitCoolingTrialSummary(
				m_iterationNumCurrent, trialIndex, "search_upward", selectedAlpha,
				selectedAlpha, selectedAlpha, m_dataFitCoolingPreviousAcceptedRms,
				selectedRms, false, m_stepLengthDampingFactorCur, 0,
				"candidate", "upper_bracket_not_found");
			return stopInitialSearch("upper_bracket_not_found");
		}
		log10Alpha = DataFitCoolingPolicy::nextHigherBracketLog10Alpha(
			std::log10(selectedAlpha), kInitialBracketLog10Span);
		alpha = std::pow(10.0, log10Alpha);
		++trialIndex;
		runDataFitCoolingTrial(alpha, trialIndex, "search_upward");
		if (!isSuccessfulTrial())
		{
			appendDataFitCoolingTrialSummary(
				m_iterationNumCurrent, trialIndex, "search_upward", alpha,
				selectedAlpha, selectedAlpha, m_dataFitCoolingPreviousAcceptedRms,
				m_dataFitCoolingTrialRms, false, m_stepLengthDampingFactorCur, 0,
				"rejected", "upper_boundary_found");
			mustExpandUpward = false;
		}
		else
		{
			selectedAlpha = alpha;
			selectedRms = m_dataFitCoolingTrialRms;
			appendDataFitCoolingTrialSummary(
				m_iterationNumCurrent, trialIndex, "search_upward", alpha,
				selectedAlpha, selectedAlpha, m_dataFitCoolingPreviousAcceptedRms,
				m_dataFitCoolingTrialRms, false, m_stepLengthDampingFactorCur, 0,
				"candidate", "search_upward");
		}
	}

	restoreBaseState();
	++trialIndex;
	runDataFitCoolingTrial(selectedAlpha, trialIndex, "selected_alpha_replay");
	if (!isSuccessfulTrial())
	{
		appendDataFitCoolingTrialSummary(
			m_iterationNumCurrent, trialIndex, "selected_alpha_replay", selectedAlpha,
			-1.0, selectedAlpha, m_dataFitCoolingPreviousAcceptedRms,
			m_dataFitCoolingTrialRms, false, m_stepLengthDampingFactorCur, 0,
			"rejected", "selected_alpha_replay_failed");
		restoreBaseState();
		m_tradeOffParameterForResistivityValue = m_dataFitCoolingInitialAlpha;
		return false;
	}

	m_dataFitCoolingHasSelectedAlpha = true;
	m_dataFitCoolingPersistentAlpha = selectedAlpha;
	m_dataFitCoolingCurrentUpdateIteration = m_iterationNumCurrent;
	m_dataFitCoolingCurrentAlphaSource = "initial_bracket_selected_alpha";
	m_tradeOffParameterForResistivityValue = selectedAlpha;
	ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
	ptrObservedData->copyDistortionParamsCurToPWK1();
	ptrObservedData->cacheSelectedTrialForwardResponse();
	appendDataFitCoolingTrialSummary(
		m_iterationNumCurrent, trialIndex, "selected_alpha_replay", selectedAlpha,
		selectedAlpha, selectedAlpha, m_dataFitCoolingPreviousAcceptedRms,
		m_dataFitCoolingTrialRms, false, m_stepLengthDampingFactorCur, 0,
		"accepted", "initial_alpha_selected");
	OutputFiles::m_logFile
		<< "# Data-fit cooling initial bracket accepted. Alpha : " << selectedAlpha
		<< ", bracket RMS : " << selectedRms
		<< ", replay RMS : " << m_dataFitCoolingTrialRms
		<< ", previous RMS : " << m_dataFitCoolingPreviousAcceptedRms << "." << std::endl;
	return true;
}

// Select the next model using full-step alpha cooling without sensitivity work
bool AnalysisControl::runPersistentDataFitCoolingAlpha()
{
	if (!m_dataFitCoolingHasSelectedAlpha || m_dataFitCoolingPersistentAlpha <= 0.0)
	{
		OutputFiles::m_logFile << "Error : Data-fit cooling persistent alpha is not initialized." << std::endl;
		exit(1);
	}

	ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData* const ptrObservedData = ObservedData::getInstance();
	ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
	ptrObservedData->copyDistortionParamsCurToPWK2();
	ptrObservedData->cacheSelectedTrialForwardResponse();
	m_stepLengthDampingFactorCur = 1.0;
	m_dataFitCoolingCurrentUpdateIteration = m_iterationNumCurrent;
	double trialAlpha = m_dataFitCoolingPersistentAlpha;
	for (int retryIndex = 0; retryIndex <= m_numCutbackMax; ++retryIndex)
	{
		const std::string alphaSource = retryIndex == 0
			? m_dataFitCoolingCurrentAlphaSource
			: "alpha_only_full_step_retry";
		OutputFiles::m_logFile
			<< "# Data-fit cooling model update trial. Iteration : " << m_iterationNumCurrent
			<< ", alpha_source : " << alphaSource
			<< ", retry : " << retryIndex
			<< ", step_length : 1"
			<< ", alpha : " << trialAlpha << "." << std::endl;
		runDataFitCoolingTrial(trialAlpha, retryIndex, alphaSource.c_str());

		if (DataFitCoolingPolicy::isRmsImproved(
			m_dataFitCoolingPreviousAcceptedRms,
			m_dataFitCoolingTrialRms))
		{
			m_dataFitCoolingPersistentAlpha = trialAlpha;
			m_tradeOffParameterForResistivityValue = trialAlpha;
			m_dataFitCoolingCurrentAlphaSource = alphaSource;
			ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
			ptrObservedData->copyDistortionParamsCurToPWK1();
			ptrObservedData->cacheSelectedTrialForwardResponse();
			appendDataFitCoolingTrialSummary(
				m_iterationNumCurrent, retryIndex, alphaSource.c_str(),
				trialAlpha, trialAlpha, trialAlpha,
				m_dataFitCoolingPreviousAcceptedRms, m_dataFitCoolingTrialRms,
				retryIndex > 0, 1.0, 0, "accepted", "full_step_model_selected");
			OutputFiles::m_logFile
				<< "# Data-fit cooling full-step alpha accepted. Iteration : "
				<< m_iterationNumCurrent << ", retry : " << retryIndex
				<< ", alpha : " << trialAlpha
				<< ", RMS : " << m_dataFitCoolingTrialRms << "." << std::endl;
			return true;
		}

		const double nextAlpha = DataFitCoolingPolicy::cooledAlpha(
			trialAlpha,
			m_dataFitCoolingFactor,
			m_dataFitCoolingMinimumAlpha);
		const bool canRetry = retryIndex < m_numCutbackMax && nextAlpha < trialAlpha;
		appendDataFitCoolingTrialSummary(
			m_iterationNumCurrent, retryIndex, alphaSource.c_str(),
			trialAlpha, -1.0, nextAlpha,
			m_dataFitCoolingPreviousAcceptedRms, m_dataFitCoolingTrialRms,
			canRetry, 1.0, 0, "rejected",
			canRetry ? "alpha_only_full_step_retry" : "no_acceptable_full_step_alpha");
		if (!canRetry)
		{
			ptrResistivityBlock->copyPWK2NotFixedToPWK1();
			ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
			ptrObservedData->copyDistortionParamsPWK2ToPWK1();
			ptrObservedData->copyDistortionParamsPWK1ToCur();
			ptrObservedData->restoreSelectedTrialForwardResponseCache();
			m_tradeOffParameterForResistivityValue = m_dataFitCoolingPersistentAlpha;
			OutputFiles::m_logFile
				<< "# Data-fit cooling stopped. termination_reason : no_acceptable_full_step_alpha."
				<< std::endl;
			return false;
		}

		++m_dataFitCoolingCount;
		trialAlpha = nextAlpha;
	}
	return false;
}

// Record an evaluated update and apply cooling only after acceptance
void AnalysisControl::applyAcceptedDataFitCoolingDecision(
	const int responseIteration,
	const int cutbackCount,
	const double currentRms,
	const double stepLengthUsed,
	const bool accepted,
	const bool terminating,
	const char* const terminationReason)
{
	if (!m_dataFitCoolingHasSelectedAlpha)
	{
		return;
	}
	const double alphaUsed = m_tradeOffParameterForResistivityValue;
	if (!accepted)
	{
		appendDataFitCoolingTrialSummary(
			m_dataFitCoolingCurrentUpdateIteration,
			cutbackCount,
			m_dataFitCoolingCurrentAlphaSource.c_str(),
			alphaUsed,
			-1.0,
			alphaUsed,
			m_dataFitCoolingPreviousAcceptedRms,
			currentRms,
			false,
			stepLengthUsed,
			cutbackCount,
			"rejected",
			terminationReason);
		m_stepLengthDampingFactorCur = 1.0;
		m_stopAfterDataFitCooling = true;
		OutputFiles::m_logFile
			<< "# Data-fit cooling response rejected. Update iteration : "
			<< m_dataFitCoolingCurrentUpdateIteration
			<< ", response iteration : " << responseIteration
			<< ", alpha : " << alphaUsed
			<< ", RMS : " << currentRms
			<< ", next action : stop_without_step_cutback." << std::endl;
		return;
	}

	bool coolingTriggered = false;
	double nextAlpha = alphaUsed;
	if (!terminating && DataFitCoolingPolicy::shouldCool(
		m_dataFitCoolingPreviousAcceptedRms,
		currentRms,
		m_dataFitCoolingTriggerThreshold))
	{
		nextAlpha = DataFitCoolingPolicy::cooledAlpha(
			alphaUsed,
			m_dataFitCoolingFactor,
			m_dataFitCoolingMinimumAlpha);
		coolingTriggered = nextAlpha < alphaUsed;
		if (coolingTriggered)
		{
			++m_dataFitCoolingCount;
		}
	}

	appendDataFitCoolingTrialSummary(
		m_dataFitCoolingCurrentUpdateIteration,
		cutbackCount,
		m_dataFitCoolingCurrentAlphaSource.c_str(),
		alphaUsed,
		alphaUsed,
		nextAlpha,
		m_dataFitCoolingPreviousAcceptedRms,
		currentRms,
		coolingTriggered,
		stepLengthUsed,
		cutbackCount,
		"accepted",
		terminationReason);

	OutputFiles::m_logFile
		<< "# Data-fit cooling accepted response. Update iteration : "
		<< m_dataFitCoolingCurrentUpdateIteration
		<< ", response iteration : " << responseIteration
		<< ", alpha used : " << alphaUsed
		<< ", previous RMS : " << m_dataFitCoolingPreviousAcceptedRms
		<< ", current RMS : " << currentRms
		<< ", cooling triggered : " << (coolingTriggered ? "yes" : "no")
		<< ", next alpha : " << nextAlpha << "." << std::endl;
	if (getMyPE() == 0)
	{
		std::cout << " # Data-fit cooling decision: alpha used = " << alphaUsed
			<< "; previous RMS = " << m_dataFitCoolingPreviousAcceptedRms
			<< "; current RMS = " << currentRms
			<< "; cooling = " << (coolingTriggered ? "yes" : "no")
			<< "; next alpha = " << nextAlpha << std::endl;
	}

	m_dataFitCoolingPreviousAcceptedRms = currentRms;
	m_dataFitCoolingPersistentAlpha = nextAlpha;
	m_dataFitCoolingCurrentAlphaSource = coolingTriggered ? "cooled_alpha" : "persistent_alpha";
}
