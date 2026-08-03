/* -------------------------------------------------------------------------------------------------------
 * FEMTIC-DABIC getter definitions split from AnalysisControl.cpp.
 * This file must not own parser, line-search, forward, or convergence behavior.
 * ------------------------------------------------------------------------------------------------------- */
#include "AnalysisControl.h"

#include <assert.h>

// Get inversion method
bool AnalysisControl::OCCAMinversion() const
{
	return m_OCCAMinversion;
}

// Get inversion method
bool AnalysisControl::ABICinversion() const
{
	return m_ABICinversion;
}

// Get inversion method
bool AnalysisControl::MinNormInv() const
{
	return m_MinNormInv;
}

bool AnalysisControl::LMdamping() const
{
	return m_Levenberg_Marquardt;
}

double AnalysisControl::getDampingofLM() const
{
	return m_dampingof_LM;
}

double AnalysisControl::getstepcutOCC() const
{
	return m_stepcutOCC;
}

// Get damping factor for resistivity value
double AnalysisControl::getTradeOffParameterForMinNorm() const
{
	return m_tradeOffParameterForMinNorm;
}

// Dieno2023
// Get damping factor for cross gradient
double AnalysisControl::getTradeOffParameterForCrossGradient() const
{
	return m_tradeOffParameterForCrossGradient;
}

// Get small value for Cross-Gradient
double AnalysisControl::getSmallvalueforCrossGradient() const
{
	return m_smallvalueForCrossGradient;
}

// Get type of boundary condition at the bottom of the model
int AnalysisControl::getBoundaryConditionBottom() const
{
	return m_boundaryConditionBottom;
}

// Get order of finite element
int AnalysisControl::getOrderOfFiniteElement() const
{
	return m_orderOfFiniteElement;
}

// Get process ID
int AnalysisControl::getMyPE() const
{
	return m_myPE;
}

// Get total number of processes
int AnalysisControl::getTotalPE() const
{
	return m_totalPE;
}

// Get total number of threads
int AnalysisControl::getNumThreads() const
{
	return m_numThreads;
}

double AnalysisControl::getSmallValueOfMinimumSupport() const
{
	return m_smallvauleOfMinimumSupport;
}

double AnalysisControl::getSmallValueOfMinimumGradientSupport() const
{
	return m_smallvauleOfMinimumGradientSupport;
}

// Get flag specifing either incore or out-of-core version of PARDISO is used
int AnalysisControl::getModeOfPARDISO() const
{
	return m_modeOfPARDISO;
}

// Get flag specifing the way of numbering of edges or nodess
int AnalysisControl::getNumberingMethod() const
{
	return m_numberingMethod;
}

// Get flag specifing whether the results of 2D forward calculations are outputed
bool AnalysisControl::getIsOutput2DResult() const
{
	return m_isOutput2DResult;
}

// Get current iteration number
int AnalysisControl::getIterationNumInit() const
{
	return m_iterationNumInit;
}

// Get current iteration number
int AnalysisControl::getIterationNumCurrent() const
{
	return m_iterationNumCurrent;
}

// Get maximum iteration number
int AnalysisControl::getIterationNumMax() const
{
	return m_iterationNumMax;
}

// Get member variable specifing which backward or forward element is used for calculating EM field
const AnalysisControl::UseBackwardOrForwardElement AnalysisControl::getUseBackwardOrForwardElement() const
{
	return m_useBackwardOrForwardElement;
}

// Get whether the specified parameter is outputed to VTK file
bool AnalysisControl::doesOutputToVTK(const int paramID) const
{
	if (m_outputParametersForVis.find(paramID) == m_outputParametersForVis.end())
	{
		return false;
	}
	else
	{
		return true;
	}
}

// Get damping factor for resistivity value
double AnalysisControl::getTradeOffParameterForResistivityValue() const
{
	return m_tradeOffParameterForResistivityValue;
}

int AnalysisControl::getNumTO() const
{
	return m_NumOF_TO;
}

bool AnalysisControl::getloglog() const
{
	return m_lCurveUseLogLog;
}

bool AnalysisControl::getnorm() const
{
	return m_lCurveUseRootNorm;
}

double AnalysisControl::get_ithTradeOffParameterForResistivityValue(const int ito) const
{
	assert(ito >= 0);
	if (m_tradeOffParameters != NULL)
	{
		assert(ito < m_NumOF_TO);
		return m_tradeOffParameters[ito];
	}
	assert(ito == 0);
	return m_tradeOffParameterForResistivityValue;
}

// Get data misfit
double AnalysisControl::getdatamisfit() const
{
	return m_datamisfit;
}

// Get trade-off parameter for distortion matrix complexity
double AnalysisControl::getTradeOffParameterForDistortionMatrixComplexity() const
{
	assert(m_typeOfDistortion == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE);
	return m_tradeOffParameterForDistortionMatrixComplexity;
}

// Get trade-off parameter for gains of distortion matrix
double AnalysisControl::getTradeOffParameterForGainsOfDistortionMatrix() const
{
	assert(m_typeOfDistortion == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS || m_typeOfDistortion == AnalysisControl::ESTIMATE_GAINS_ONLY);
	return m_tradeOffParameterForDistortionGain;
}

// Get trade-off parameter for rotations of distortion matrix
double AnalysisControl::getTradeOffParameterForRotationsOfDistortionMatrix() const
{
	assert(m_typeOfDistortion == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS);
	return m_tradeOffParameterForDistortionRotation;
}

// Get current factor of step length damping
double AnalysisControl::getStepLengthDampingFactorCur() const
{
	return m_stepLengthDampingFactorCur;
}

// Get maximum number of cutbacks.
int AnalysisControl::getNumCutbackMax() const
{
	return m_numCutbackMax;
}

// Get flag whether memory of solver is held after forward calculation
bool AnalysisControl::holdMemoryForwardSolver() const
{
	return m_holdMemoryForwardSolver;
}

// Get flag whether using Cross-Gradient
bool AnalysisControl::runCG() const
{
	return m_CrossGradientInv;
}

// Get type of mesh
int AnalysisControl::getTypeOfMesh() const
{
	return m_typeOfMesh;
}

// Get flag specifing whether distortion matrix is estimated or not
bool AnalysisControl::estimateDistortionMatrix() const
{
	return (m_typeOfDistortion != AnalysisControl::NO_DISTORTION);
}

// Get type of galvanic distortion
int AnalysisControl::getTypeOfDistortion() const
{
	return m_typeOfDistortion;
}

// Get flag specifing the way of creating roughning matrix
int AnalysisControl::geTypeOfRoughningMatrix() const
{
	return m_typeOfRoughningMatrix;
}

// Get type of the electric field used to calculate response functions
int AnalysisControl::getTypeOfElectricField() const
{
	return m_typeOfElectricField;
}

// Flag specifing whether type of the electric field of each site is specified indivisually
bool AnalysisControl::isTypeOfElectricFieldSetIndivisually() const
{
	return m_isTypeOfElectricFieldSetIndivisually;
}

// Tyep of owner element of observation sites
int AnalysisControl::getTypeOfOwnerElement() const
{
	return m_typeOfOwnerElement;
}

// Flag specifing whether the type of owner element of each site is specified indivisually
bool AnalysisControl::isTypeOfOwnerElementSetIndivisually() const
{
	return m_isTypeOfOwnerElementSetIndivisually;
}

// Get division number of right-hand sides at solve phase in forward calculation
int AnalysisControl::getDivisionNumberOfMultipleRHSInForward() const
{
	return m_divisionNumberOfMultipleRHSInForward;
}

// Get division number of right-hand sides at solve phase in inversion
int AnalysisControl::getDivisionNumberOfMultipleRHSInInversion() const
{
	return m_divisionNumberOfMultipleRHSInInversion;
}

// Get weighting factor of alpha
double AnalysisControl::getAlphaWeight(const int iDir) const
{
	assert(iDir >= 0 && iDir < 3);
	return m_alphaWeight[iDir];
}

// Get flag specifing whether the cofficient matrix of the normal equation is positive definite or not
bool AnalysisControl::getPositiveDefiniteNormalEqMatrix() const
{
	return m_positiveDefiniteNormalEqMatrix;
}

// Get flag specifing whether output file for paraview is binary or ascii
bool AnalysisControl::writeBinaryFormat() const
{
	return m_binaryOutput;
}

int AnalysisControl::getDegreeOfLpMinimumNorm() const
{
	return m_degreeOfLpMinimumNorm;
}

// Get lower limit of the difference of log10(rho) for Lp optimization
double AnalysisControl::getLowerLimitOfDifflog10RhoForLpMinimumNorm() const
{
	return m_lowerLimitOfDifflog10RhoForLpMinimumNorm;
}

double AnalysisControl::getUpperLimitOfDifflog10RhoForLpMinimumNorm() const
{
	return m_upperLimitOfDifflog10RhoForLpMinimumNorm;
}

// Get inversion method
int AnalysisControl::getInversionMethod() const
{
	return m_inversionMethod;
}

// Get flag specifing whether observation point is moved to the horizontal center of the element including it
int AnalysisControl::getIsObsLocMovedToCenter() const
{
	return m_isObsLocMovedToCenter;
}

// Get option about treatment of apparent resistivity & phase
int AnalysisControl::getApparentResistivityAndPhaseTreatmentOption() const
{
	return m_apparentResistivityAndPhaseTreatmentOption;
}

// Get flag specifing whether roughening matrix is outputed
bool AnalysisControl::getIsRougheningMatrixOutputted() const
{
	return m_isRougheningMatrixOutputted;
}

// Get type of data space algorithm
int AnalysisControl::getTypeOfDataSpaceAlgorithm() const
{
	return m_typeOfDataSpaceAlgorithm;
}

// Get flag specifing whether Lp optimization with difference filter is used
bool AnalysisControl::useDifferenceFilter() const
{
	return m_useDifferenceFilter;
}

// Get degree of Lp optimization
int AnalysisControl::getDegreeOfLpOptimization() const
{
	return m_degreeOfLpOptimization;
}

// Get residual updated or not
int AnalysisControl::getresidualupdate() const
{
	return m_residualupdated;
}

// Get type of Cross-Gradient operator
int AnalysisControl::gettypeofCG() const
{
	return m_typeOfCG;
}

// Get lower limit of the difference of log10(rho) for Lp optimization
double AnalysisControl::getLowerLimitOfDifflog10RhoForLpOptimization() const
{
	return m_lowerLimitOfDifflog10RhoForLpOptimization;
}

// Get upper limit of the difference of log10(rho) for Lp optimization
double AnalysisControl::getUpperLimitOfDifflog10RhoForLpOptimization() const
{
	return m_upperLimitOfDifflog10RhoForLpOptimization;
}

// Get maximum iteration number of IRWLS for Lp optimization
int AnalysisControl::getMaxIterationIRWLSForLpOptimization() const
{
	return m_maxIterationIRWLSForLpOptimization;
}

// Get threshold value for deciding convergence about IRWLS for Lp optimization
double AnalysisControl::getThresholdIRWLSForLpOptimization() const
{
	return m_thresholdIRWLSForLpOptimization;
}

// Get directory of out-of-core files for the sensitivitry matrix
std::string AnalysisControl::getDirectoryOfOutOfCoreFilesForSensitivityMatrix() const
{
	return m_directoryOfOutOfCoreFilesForSensitivityMatrix;
}

int AnalysisControl::getAppraisalMode() const
{
	return m_appraisalMode;
}

bool AnalysisControl::isAppraisalEnabled() const
{
	return m_appraisalMode != AnalysisControl::APPRAISAL_DISABLED;
}

int AnalysisControl::getNumRandomVectorsForAppraisal() const
{
	return m_numRandomVectorsForAppraisal;
}

const std::vector<int>& AnalysisControl::getAppraisalCheckpoints() const
{
	return m_appraisalCheckpoints;
}

std::string AnalysisControl::getAppraisalInputSensitivityDirectory() const
{
	return m_appraisalInputSensitivityDirectory;
}

std::string AnalysisControl::getAppraisalOutputDirectory() const
{
	return m_appraisalOutputDirectory;
}

bool AnalysisControl::writeLegacyAppraisalDsdkFiles() const
{
	return m_writeLegacyAppraisalDsdkFiles;
}
