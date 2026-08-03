/* -------------------------------------------------------------------------------------------------------
 * FEMTIC-DABIC forward-computation member definitions split from AnalysisControl.cpp.
 * This file owns forward-response cache dispatch and sensitivity calculation flow.
 * ------------------------------------------------------------------------------------------------------- */
#include "AnalysisControl.h"

#include <cstdlib>
#include <string>

#include "Forward3D.h"
#include "ObservedData.h"
#include "OutputFiles.h"

namespace {

const char* kLogStartForwardFieldForSensitivityPrefix =
	"# Start forward-field calculation for sensitivity. Frequency : ";
const char* kLogStartForwardCalculationPrefix =
	"# Start forward calculation. Frequency : ";
const char* kLogStationEMFieldsForSensitivity =
	"# Calculate station EM fields for sensitivity while keeping cached response functions.";
const char* kLogCalculateResponseFunctions =
	"# Calculate response functions. ";
const char* kLogCalculateSensitivityFromCachedResponses =
	"# Calculate sensitivity matrix from forward-field derivatives and cached responses. ";
const char* kLogStartNonlinearLCurveTrialForwardPrefix =
	"# Start nonlinear L-curve trial forward-response calculation. Frequency : ";
const char* kLogCalculateNonlinearLCurveTrialResponses =
	"# Calculate nonlinear L-curve trial response functions without derivative or sensitivity work. ";

}

// Calculate forward computation
void AnalysisControl::calcForwardComputation(
	const int iter,
	const bool reuseSelectedTrialForwardResponse)
{

	Forward3D *ptrForward3D = getPointerOfForward3D();
	if (ptrForward3D == NULL)
	{
		OutputFiles::m_logFile << "Error : Pointer to the class Forward3D is NULL." << std::endl;
		exit(1);
	}

	if (m_ptrInversion == NULL)
	{
		OutputFiles::m_logFile << "Error : m_ptrInversion is NULL." << std::endl;
		exit(1);
	}

	ObservedData *const pObservedData = ObservedData::getInstance();
	const bool calculateSensitivity = doesCalculateSensitivity(iter);
	if (reuseSelectedTrialForwardResponse && calculateSensitivity && canUseSelectedTrialForwardStateCache(iter))
	{
		OutputFiles::m_logFile << "Error : Selected-trial forward-state cache is not implemented." << std::endl;
		exit(1);
	}

	const int numOfFrequenciesCalculatedByThisPE = pObservedData->getNumOfFrequenciesCalculatedByThisPE();
	for (int ifreq = 0; ifreq < numOfFrequenciesCalculatedByThisPE; ++ifreq)
	{

		// const int ifreq = m_IDsOfFrequenciesCalculatedByThisPE[ifreq];
		const double frquencyValue = pObservedData->getValuesOfFrequenciesCalculatedByThisPE(ifreq);

		for (int iPol = 0; iPol < 2; ++iPol)
		{

			std::string polarizationName;
			if (iPol == 0)
			{
				polarizationName = "Ex-polarization";
			}
			else
			{
				polarizationName = "Ey-polarization";
			}

			OutputFiles::m_logFile << "#================================================================================================" << std::endl;
			if (reuseSelectedTrialForwardResponse)
			{
				OutputFiles::m_logFile << kLogStartForwardFieldForSensitivityPrefix << frquencyValue << " [Hz], Polarization : " << polarizationName << std::endl;
			}
			else
			{
				OutputFiles::m_logFile << kLogStartForwardCalculationPrefix << frquencyValue << " [Hz], Polarization : " << polarizationName << std::endl;
			}
			OutputFiles::m_logFile << "#================================================================================================" << std::endl;

			ptrForward3D->forwardCalculation(frquencyValue, iPol);

			if (!reuseSelectedTrialForwardResponse || calculateSensitivity)
			{
				if (reuseSelectedTrialForwardResponse && calculateSensitivity)
				{
					OutputFiles::m_logFile << kLogStationEMFieldsForSensitivity << std::endl;
				}
				pObservedData->calculateEMFieldOfAllStations(ptrForward3D, frquencyValue, iPol, ifreq);
			}

			if (calculateSensitivity)
			{
				m_ptrInversion->calculateDerivativesOfEMField(ptrForward3D, frquencyValue, iPol);
			}
		}

		if (!reuseSelectedTrialForwardResponse)
		{
			OutputFiles::m_logFile << "#==============================================================================" << std::endl;
			OutputFiles::m_logFile << kLogCalculateResponseFunctions << outputElapsedTime() << std::endl;
			pObservedData->calculateResponseFunctionOfAllStations(ifreq);
		}
		if (calculateSensitivity)
		{
			if (reuseSelectedTrialForwardResponse)
			{
				OutputFiles::m_logFile << "#==============================================================================" << std::endl;
				OutputFiles::m_logFile << kLogCalculateSensitivityFromCachedResponses << outputElapsedTime() << std::endl;
			}
			m_ptrInversion->calculateSensitivityMatrix(ifreq, frquencyValue);
		}
		OutputFiles::m_logFile << "#==============================================================================" << std::endl;
	}

	OutputFiles::m_logFile << "# Release memory of coefficient matrix and sparse solver. " << outputElapsedTime() << std::endl;
	if (!m_holdMemoryForwardSolver)
	{ // Release memory of sparse solver
		ptrForward3D->releaseMemoryOfMatrixAndSolver();
	}
}

// Calculate forward responses for one nonlinear L-curve trial without derivative or sensitivity work
void AnalysisControl::calcForwardResponseForNonlinearLCurveTrial(
	const int iter,
	const int trialIndex,
	const double tradeOffParameter)
{

	Forward3D *ptrForward3D = getPointerOfForward3D();
	if (ptrForward3D == NULL)
	{
		OutputFiles::m_logFile << "Error : Pointer to the class Forward3D is NULL." << std::endl;
		exit(1);
	}

	ObservedData *const pObservedData = ObservedData::getInstance();
	const int numOfFrequenciesCalculatedByThisPE = pObservedData->getNumOfFrequenciesCalculatedByThisPE();
	for (int ifreq = 0; ifreq < numOfFrequenciesCalculatedByThisPE; ++ifreq)
	{

		const double frquencyValue = pObservedData->getValuesOfFrequenciesCalculatedByThisPE(ifreq);

		for (int iPol = 0; iPol < 2; ++iPol)
		{

			std::string polarizationName;
			if (iPol == 0)
			{
				polarizationName = "Ex-polarization";
			}
			else
			{
				polarizationName = "Ey-polarization";
			}

			OutputFiles::m_logFile << "#================================================================================================" << std::endl;
			OutputFiles::m_logFile << kLogStartNonlinearLCurveTrialForwardPrefix
								   << frquencyValue << " [Hz], Polarization : " << polarizationName
								   << ", iteration : " << iter
								   << ", trial : " << trialIndex
								   << ", trade-off : " << tradeOffParameter << std::endl;
			OutputFiles::m_logFile << "#================================================================================================" << std::endl;

			ptrForward3D->forwardCalculation(frquencyValue, iPol);
			pObservedData->calculateEMFieldOfAllStations(ptrForward3D, frquencyValue, iPol, ifreq);
		}

		OutputFiles::m_logFile << "#==============================================================================" << std::endl;
		OutputFiles::m_logFile << kLogCalculateNonlinearLCurveTrialResponses << outputElapsedTime() << std::endl;
		pObservedData->calculateResponseFunctionOfAllStations(ifreq);
		OutputFiles::m_logFile << "#==============================================================================" << std::endl;
	}

	OutputFiles::m_logFile << "# Release memory of coefficient matrix and sparse solver. " << outputElapsedTime() << std::endl;
	if (!m_holdMemoryForwardSolver)
	{
		ptrForward3D->releaseMemoryOfMatrixAndSolver();
	}
}

// Return whether selected-trial forward-state cache can replace forward-field and sensitivity-state work
bool AnalysisControl::canUseSelectedTrialForwardStateCache(const int iter) const
{
	(void)iter;
	return false;
}

// Return whether selected-trial forward-response cache can replace forward response calculation
bool AnalysisControl::canUseSelectedTrialForwardResponseCache(const int iter) const
{
	const ObservedData *const pObservedData = ObservedData::getInstance();

	if (iter <= m_iterationNumInit)
	{
		return false;
	}
	if (m_typeOfTradeOffParam != AnalysisControl::TO_ABIC_LS)
	{
		return false;
	}
	if (m_isOutput2DResult)
	{
		return false;
	}
	if (doesOutputToVTK(AnalysisControl::OUTPUT_ELECTRIC_FIELD_VECTORS_TO_VTK) ||
		doesOutputToVTK(AnalysisControl::OUTPUT_MAGNETIC_FIELD_VECTORS_TO_VTK) ||
		doesOutputToVTK(AnalysisControl::OUTPUT_CURRENT_DENSITY) ||
		doesOutputToVTK(AnalysisControl::OUTPUT_SENSITIVITY) ||
		doesOutputToVTK(AnalysisControl::OUTPUT_SENSITIVITY_DENSITY))
	{
		return false;
	}
	if (!pObservedData->supportsSelectedTrialForwardResponseCache())
	{
		return false;
	}
	return pObservedData->hasSelectedTrialForwardResponseCache();
}
