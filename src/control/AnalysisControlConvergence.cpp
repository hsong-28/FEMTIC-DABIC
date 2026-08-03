/* -------------------------------------------------------------------------------------------------------
 * FEMTIC-DABIC convergence member definitions split from AnalysisControl.cpp.
 * This file owns convergence predicates, CNV row dispatch, and step-length handoff.
 * ------------------------------------------------------------------------------------------------------- */
#include "AnalysisControl.h"

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "mpi.h"
#include "ObservedData.h"
#include "OutputFiles.h"
#include "ResistivityBlock.h"

// Return whether convergence data should be written to the CNV file
bool AnalysisControl::shouldWriteConvergenceDataToCnv(const int iterCur) const
{
	return iterCur == 0 || iterCur > getIterationNumInit();
}

// Exit if the CNV file is not open before convergence-data output
void AnalysisControl::ensureCnvFileIsOpenForConvergence() const
{
	if (!OutputFiles::m_cnvFile.is_open())
	{
		OutputFiles::m_logFile << "Error : CNV file has not been opened." << std::endl;
		exit(1);
	}
}

// Adjust factor of step length damping and output convergence data to cnv file
AnalysisControl::ConvergenceBehaviors AnalysisControl::adjustStepLengthDampingFactor(const int iterCur, const int iCutbackCur)
{

	// Get process ID
	// MPI_Comm_rank( MPI_COMM_WORLD, &myProcessID );
	const int myProcessID = getMyPE();

	ObservedData *const pObservedData = ObservedData::getInstance();

	// const double dataMisfit = pObservedData->calculateErrorSumOfSquares();
	double dataMisfitThisPE = pObservedData->calculateErrorSumOfSquaresThisPE();

#ifdef _DEBUG_WRITE
	std::cout << "PE dataMisfitThisPE : " << myProcessID << " " << dataMisfitThisPE << std::endl; // For debug
#endif

	double dataMisfit(0.0);
	MPI_Reduce(&dataMisfitThisPE, &dataMisfit, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	m_datamisfit = dataMisfit;

#ifdef _DEBUG_WRITE
	if (myProcessID == 0)
	{															 // Zero process only ---------------------
		std::cout << "dataMisfit = " << dataMisfit << std::endl; // For debug
	}
#endif

	int iynConverged(0);
	int iynGoNextIteration(0);

	int numDataThisPE = pObservedData->getNumObservedDataThisPETotal();
	int numDataTotal(0);
	MPI_Reduce(&numDataThisPE, &numDataTotal, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	double rms = 0.0;
	double modelRoughness = 0.0;
	const double stepLengthDampingFactorUsed = m_stepLengthDampingFactorCur;

	if (myProcessID == 0)
	{ // Zero process only ---------------------

		rms = sqrt(dataMisfit / static_cast<double>(numDataTotal));
		m_rmsPre = rms;
		double modelNorm(0.0);
		double CrossGradient(0.0);
		if (useDifferenceFilter())
		{
			modelRoughness = (ResistivityBlock::getInstance())->calcModelRoughnessForDifferenceFilter();
		}
		else
		{
			modelRoughness = (ResistivityBlock::getInstance())->calcModelRoughnessForLaplacianFilter();
		}
		modelNorm = (ResistivityBlock::getInstance())->calModelNormLog10();
		const double modelRoughnessMultipliedByAlphaAlpha = modelRoughness * pow(m_tradeOffParameterForResistivityValue, 2);

		double objectFunctionalCur(0.0);
		double obj(0.0);
		obj = dataMisfit + modelRoughnessMultipliedByAlphaAlpha;
		// if (m_CrossGradientInv)
		// {

		// 	const double CrossGradientMultipliedByGammaGammma = CrossGradient * pow(m_tradeOffParameterForCrossGradient, 2);
		// }
		if (m_typeOfTradeOffParam == AnalysisControl::TO_Fixed)
		{
			objectFunctionalCur = dataMisfit + modelRoughnessMultipliedByAlphaAlpha;
		}
		else
		{
			objectFunctionalCur = dataMisfit;
		}
		m_objPre = obj;
		if (iterCur == getIterationNumInit() + 1)
		{
			m_abicpre[0] = m_abic[0] + 1.0;
		}
		if (iterCur == m_iterationNumInit)
		{
			std::cout << " --------------------------------------------------- " << std::endl;
			std::cout << " Rough outcomes of current iteration: " << std::endl;
			std::cout << " iterCur = " << iterCur << std::endl;
			std::cout << " stepLength = " << m_stepLengthDampingFactorCur << std::endl;
			std::cout << " rms = " << rms << std::endl;
			std::cout << " dataMisfit = " << dataMisfit << std::endl;
			std::cout << " modelRoughness = " << modelRoughness << std::endl;
			std::cout << " modelNorm = " << modelNorm << std::endl;
			if (m_typeOfTradeOffParam == AnalysisControl::TO_Fixed)
			{
				std::cout << " objectFunctionalCur = " << objectFunctionalCur << std::endl;
			}
			else
			{
				std::cout << " objectFunctionalCur = " << obj << std::endl;
			}
			if (m_CrossGradientInv)
			{
				CrossGradient = (ResistivityBlock::getInstance())->calcCrossGradient();
				std::cout << " Cross-Gradient = " << CrossGradient << std::endl;
			}
			std::cout << " --------------------------------------------------- " << std::endl;
		}
		else
		{
			std::cout << " --------------------------------------------------- " << std::endl;
			std::cout << " Rough outcomes of current iteration: " << std::endl;
			std::cout << " iterCur = " << iterCur << std::endl;
			std::cout << " stepLength = " << m_stepLengthDampingFactorCur << std::endl;
			std::cout << " Tradeoffparameter = " << m_tradeOffParameterForResistivityValue << std::endl;
			std::cout << " rms = " << rms << std::endl;
			std::cout << " dataMisfit = " << dataMisfit << std::endl;
			std::cout << " modelRoughness = " << modelRoughness << std::endl;
			std::cout << " modelNorm = " << modelNorm << std::endl;
			if (m_typeOfTradeOffParam == AnalysisControl::TO_ABIC_LS)
			{
				std::cout << " modelUpdatedmean = " << m_updatedmean << std::endl;
				std::cout << " ABIC = " << m_abic[0] << std::endl;
			}
			if (m_CrossGradientInv)
			{
				CrossGradient = (ResistivityBlock::getInstance())->calcCrossGradient();
				std::cout << " Cross-Gradient = " << CrossGradient << std::endl;
			}
			if (iCutbackCur > 0)
			{
				std::cout << " objectFunctionalCur = " << obj << std::endl;
				std::cout << " objectFunctionalPre = " << m_objPreiter << std::endl;
			}
			std::cout << " --------------------------------------------------- " << std::endl;
		}

		double distortionMatrixNorm = -1.0;
		double normOfGains = -1.0;
		double normOfRotations = -1.0;

		//----------------------------------------
		// Output convergence data to cnv file
		//----------------------------------------
		if (shouldWriteConvergenceDataToCnv(iterCur))
		{
			ensureCnvFileIsOpenForConvergence();

			if ((AnalysisControl::getInstance())->ABICinversion())
			{
				if (m_CrossGradientInv)
				{
					if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
					{
						distortionMatrixNorm = pObservedData->calculateSumSquareOfDistortionMatrixComplexity();
						objectFunctionalCur += m_tradeOffParameterForDistortionMatrixComplexity * m_tradeOffParameterForDistortionMatrixComplexity * distortionMatrixNorm;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionMatrixComplexity
											   << std::setw(15) << std::scientific << m_tradeOffParameterForCrossGradient
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << distortionMatrixNorm
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << CrossGradient
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << m_abic[0]
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
					{
						normOfGains = pObservedData->calculateSumSquareOfDistortionMatrixGains();
						objectFunctionalCur += m_tradeOffParameterForDistortionGain * m_tradeOffParameterForDistortionGain * normOfGains;
						normOfRotations = pObservedData->calculateSumSquareOfDistortionMatrixRotations();
						objectFunctionalCur += m_tradeOffParameterForDistortionRotation * m_tradeOffParameterForDistortionRotation * normOfRotations;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionGain
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionRotation
											   << std::setw(15) << std::scientific << m_tradeOffParameterForCrossGradient
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << normOfGains
											   << std::setw(15) << std::scientific << normOfRotations
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << CrossGradient
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << m_abic[0]
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
					{
						normOfGains = pObservedData->calculateSumSquareOfDistortionMatrixGains();
						objectFunctionalCur += m_tradeOffParameterForDistortionGain * m_tradeOffParameterForDistortionGain * normOfGains;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionGain
											   << std::setw(15) << std::scientific << m_tradeOffParameterForCrossGradient
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << normOfGains
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << CrossGradient
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << m_abic[0]
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else
					{
						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForCrossGradient
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << CrossGradient
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << m_abic[0]
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
				}
				else
				{
					if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
					{
						distortionMatrixNorm = pObservedData->calculateSumSquareOfDistortionMatrixComplexity();
						objectFunctionalCur += m_tradeOffParameterForDistortionMatrixComplexity * m_tradeOffParameterForDistortionMatrixComplexity * distortionMatrixNorm;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionMatrixComplexity
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << distortionMatrixNorm
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << m_abic[0]
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
					{
						normOfGains = pObservedData->calculateSumSquareOfDistortionMatrixGains();
						objectFunctionalCur += m_tradeOffParameterForDistortionGain * m_tradeOffParameterForDistortionGain * normOfGains;
						normOfRotations = pObservedData->calculateSumSquareOfDistortionMatrixRotations();
						objectFunctionalCur += m_tradeOffParameterForDistortionRotation * m_tradeOffParameterForDistortionRotation * normOfRotations;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionGain
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionRotation
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << normOfGains
											   << std::setw(15) << std::scientific << normOfRotations
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << m_abic[0]
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
					{
						normOfGains = pObservedData->calculateSumSquareOfDistortionMatrixGains();
						objectFunctionalCur += m_tradeOffParameterForDistortionGain * m_tradeOffParameterForDistortionGain * normOfGains;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionGain
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << normOfGains
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << m_abic[0]
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else
					{
						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << m_abic[0]
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
				}
			}
			else
			{
				if (m_CrossGradientInv)
				{
					if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
					{
						distortionMatrixNorm = pObservedData->calculateSumSquareOfDistortionMatrixComplexity();
						objectFunctionalCur += m_tradeOffParameterForDistortionMatrixComplexity * m_tradeOffParameterForDistortionMatrixComplexity * distortionMatrixNorm;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionMatrixComplexity
											   << std::setw(15) << std::scientific << m_tradeOffParameterForCrossGradient
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << distortionMatrixNorm
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << CrossGradient
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
					{
						normOfGains = pObservedData->calculateSumSquareOfDistortionMatrixGains();
						objectFunctionalCur += m_tradeOffParameterForDistortionGain * m_tradeOffParameterForDistortionGain * normOfGains;
						normOfRotations = pObservedData->calculateSumSquareOfDistortionMatrixRotations();
						objectFunctionalCur += m_tradeOffParameterForDistortionRotation * m_tradeOffParameterForDistortionRotation * normOfRotations;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionGain
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionRotation
											   << std::setw(15) << std::scientific << m_tradeOffParameterForCrossGradient
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << normOfGains
											   << std::setw(15) << std::scientific << normOfRotations
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << CrossGradient
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
					{
						normOfGains = pObservedData->calculateSumSquareOfDistortionMatrixGains();
						objectFunctionalCur += m_tradeOffParameterForDistortionGain * m_tradeOffParameterForDistortionGain * normOfGains;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionGain
											   << std::setw(15) << std::scientific << m_tradeOffParameterForCrossGradient
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << normOfGains
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << CrossGradient
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else
					{
						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForCrossGradient
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << CrossGradient
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
				}
				else
				{
					if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
					{
						distortionMatrixNorm = pObservedData->calculateSumSquareOfDistortionMatrixComplexity();
						objectFunctionalCur += m_tradeOffParameterForDistortionMatrixComplexity * m_tradeOffParameterForDistortionMatrixComplexity * distortionMatrixNorm;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionMatrixComplexity
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << distortionMatrixNorm
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
					{
						normOfGains = pObservedData->calculateSumSquareOfDistortionMatrixGains();
						objectFunctionalCur += m_tradeOffParameterForDistortionGain * m_tradeOffParameterForDistortionGain * normOfGains;
						normOfRotations = pObservedData->calculateSumSquareOfDistortionMatrixRotations();
						objectFunctionalCur += m_tradeOffParameterForDistortionRotation * m_tradeOffParameterForDistortionRotation * normOfRotations;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionGain
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionRotation
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << normOfGains
											   << std::setw(15) << std::scientific << normOfRotations
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
					{
						normOfGains = pObservedData->calculateSumSquareOfDistortionMatrixGains();
						objectFunctionalCur += m_tradeOffParameterForDistortionGain * m_tradeOffParameterForDistortionGain * normOfGains;

						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_tradeOffParameterForDistortionGain
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << normOfGains
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
					else
					{
						OutputFiles::m_cnvFile.precision(4);
						OutputFiles::m_cnvFile << std::setw(10) << iterCur << std::setw(10) << iCutbackCur
											   << std::setw(15) << std::scientific << m_tradeOffParameterForResistivityValuePre
											   << std::setw(15) << std::scientific << m_stepLengthDampingFactorCur
											   << std::setw(15) << std::scientific << modelRoughness
											   << std::setw(15) << std::scientific << dataMisfit
											   << std::setw(15) << std::scientific << rms
											   << std::setw(15) << std::scientific << m_updatedmean
											   << std::setw(15) << std::scientific << objectFunctionalCur
											   << std::endl;
					}
				}
			}
		}

		// Perform convergence test
		if (m_typeOfTradeOffParam == AnalysisControl::TO_ABIC_LS)
		{
			if (rms <= 1.01 * m_tolreq)
			{
				m_ABICconverage = true;
			}
		}

		const bool dataFitCoolingTargetReached =
			isDataFitCoolingMode() && m_dataFitCoolingHasSelectedAlpha && rms <= m_tolreq;
		// Perform convergence test
		if (dataFitCoolingTargetReached || checkConvergence(objectFunctionalCur, iterCur))
		{ // SONG240412-2RatiosForConverage
			iynConverged = 1;
		}
		//----------------------------------------
		// Adjust factor of step length damping
		//----------------------------------------
		if (iterCur <= getIterationNumInit() || m_continueWithoutCutback)
		{ // First iteration or no cutback

			m_objectFunctionalPre = objectFunctionalCur;
			m_objPre = obj;
			m_objPreiter = obj;
			m_abicpre = m_abic;
			iynGoNextIteration = 1; // Go next iteration
			m_dataMisfitPre = dataMisfit;
		}
		else
		{
			const double stepLengthDampingFactorPre = m_stepLengthDampingFactorCur;

			if ((objectFunctionalCur < m_objectFunctionalPre - m_thresholdValueForDecreasing && m_dataMisfitPre > dataMisfit) ||
				m_OCCAMsmoothing || m_ABICconverage)
			{
				// Value of objective functional decrease from the one of previous iteration
				if (m_ABICinversion)
				{
					if (m_abicpre[0] > m_abic[0])
					{
						OutputFiles::m_logFile << "m_dataMisfitPre: " << m_dataMisfitPre << std::endl;
						OutputFiles::m_logFile << "m_dataMisfitCur: " << dataMisfit << std::endl;
						iynGoNextIteration = 1; // Go next iteration

						if (++m_numConsecutiveIterFunctionalDecreasing >= m_numOfIterIncreaseStepLength)
						{
							m_stepLengthDampingFactorCur *= m_factorIncreasingStepLength; // Increase factor of step length damping
						}
						m_objectFunctionalPre2 = m_objectFunctionalPre; // SONG240412-2RatiosForConverage
						m_objectFunctionalPre = objectFunctionalCur;
						m_abicpre = m_abic;
						m_objPre = obj;
						m_objPreiter = obj;

						m_dataMisfitPre = dataMisfit;
						m_modelRoughnessPre = modelRoughness;
						m_normOfDistortionMatrixDifferencesPre = distortionMatrixNorm;
						m_normOfGainsPre = normOfGains;
						m_normOfRotationsPre = normOfRotations;
					}
					else
					{
						// Value of objective functional increase from the one of previous iteration
						std::cout << " # m_abicPre: " << m_abicpre[0] << "  <  " << "m_abicCur: " << m_abic[0] << std::endl;
						std::cout << " # Cutting the stepsize... " << std::endl;
						iynGoNextIteration = 0;										  // Not go next iteration
						m_numConsecutiveIterFunctionalDecreasing = 0;				  // reset value
						m_stepLengthDampingFactorCur *= m_factorDecreasingStepLength; // Decreaes factor of step length damping
					}
				}
				else
				{
					OutputFiles::m_logFile << "m_dataMisfitPre: " << m_dataMisfitPre << std::endl;
					OutputFiles::m_logFile << "m_dataMisfitCur: " << dataMisfit << std::endl;
					iynGoNextIteration = 1; // Go next iteration

					if (++m_numConsecutiveIterFunctionalDecreasing >= m_numOfIterIncreaseStepLength)
					{
						m_stepLengthDampingFactorCur *= m_factorIncreasingStepLength; // Increase factor of step length damping
					}
					m_objectFunctionalPre2 = m_objectFunctionalPre; // SONG240412-2RatiosForConverage
					m_objectFunctionalPre = objectFunctionalCur;
					m_abicpre = m_abic;
					m_objPre = obj;
					m_objPreiter = obj;

					m_dataMisfitPre = dataMisfit;
					m_modelRoughnessPre = modelRoughness;
					m_normOfDistortionMatrixDifferencesPre = distortionMatrixNorm;
					m_normOfGainsPre = normOfGains;
					m_normOfRotationsPre = normOfRotations;
				}
			}
			else
			{
				// Value of objective functional increase from the one of previous iteration
				std::cout << " # m_dataMisfitPre: " << m_dataMisfitPre << "  <  " << "m_dataMisfitCur: " << dataMisfit << std::endl;
				if (isDataFitCoolingMode())
				{
					std::cout << " # Data-fit cooling keeps full step and does not perform step cutback." << std::endl;
				}
				else if (m_dataMisfitPre <= dataMisfit || m_abicpre[0] <= m_abic[0])
				{
					std::cout << " # Cutting the stepsize... " << std::endl;
				}

				iynGoNextIteration = 0; // Not go next iteration

				m_numConsecutiveIterFunctionalDecreasing = 0; // reset value

				if (isDataFitCoolingMode())
				{
					m_stepLengthDampingFactorCur = 1.0;
				}
				else
				{
					m_stepLengthDampingFactorCur *= m_factorDecreasingStepLength; // Decreaes factor of step length damping
				}
			}

			if (m_stepLengthDampingFactorCur < m_stepLengthDampingFactorMin)
			{ // Reach minimum value
				m_stepLengthDampingFactorCur = m_stepLengthDampingFactorMin;
			}

			if (m_stepLengthDampingFactorCur > m_stepLengthDampingFactorMax)
			{ // Reach maximum value
				m_stepLengthDampingFactorCur = m_stepLengthDampingFactorMax;
			}

			const double threshold = 1.0E-8;
			if (fabs(m_stepLengthDampingFactorCur - stepLengthDampingFactorPre) > threshold)
			{

				OutputFiles::m_logFile << "# Factor of step length damping change from " << stepLengthDampingFactorPre << " to " << m_stepLengthDampingFactorCur << "." << std::endl;
			}
		}
	} //--------------------------------------------------------------

	MPI_Bcast(&m_rmsPre, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&iynConverged, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&iynGoNextIteration, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_objectFunctionalPre, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(m_abicpre.data(), m_abicpre.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_objPre, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_dataMisfitPre, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_modelRoughnessPre, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_normOfDistortionMatrixDifferencesPre, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_normOfGainsPre, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_normOfRotationsPre, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_stepLengthDampingFactorCur, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_numConsecutiveIterFunctionalDecreasing, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (isDataFitCoolingMode() && m_dataFitCoolingHasSelectedAlpha)
	{
		const bool accepted = iynConverged == 1 || iynGoNextIteration == 1;
		const bool targetReached = accepted && m_rmsPre <= m_tolreq;
		const char* const terminationReason = !accepted
			? "selected_full_step_response_not_reproduced"
			: (targetReached
				? "target_rms_reached"
				: (iynConverged == 1 ? "convergence_reached" : "continue"));
		applyAcceptedDataFitCoolingDecision(
			iterCur,
			iCutbackCur,
			m_rmsPre,
			stepLengthDampingFactorUsed,
			accepted,
			iynConverged == 1,
			terminationReason);
	}

#ifdef _DEBUG_WRITE
	std::cout << "PE iynConverged : " << myProcessID << " " << iynConverged << std::endl;														  // For debug
	std::cout << "PE iynGoNextIteration : " << myProcessID << " " << iynGoNextIteration << std::endl;											  // For debug
	std::cout << "PE m_objectFunctionalPre : " << myProcessID << " " << m_objectFunctionalPre << std::endl;										  // For debug
	std::cout << "PE m_dataMisfitPre : " << myProcessID << " " << m_dataMisfitPre << std::endl;													  // For debug
	std::cout << "PE m_modelRoughnessPre : " << myProcessID << " " << m_modelRoughnessPre << std::endl;											  // For debug
	std::cout << "PE m_normOfDistortionMatrixDifferencesPre : " << myProcessID << " " << m_normOfDistortionMatrixDifferencesPre << std::endl;	  // For debug
	std::cout << "PE m_normOfGainsPre : " << myProcessID << " " << m_normOfGainsPre << std::endl;												  // For debug
	std::cout << "PE m_normOfRotationsPre : " << myProcessID << " " << m_normOfRotationsPre << std::endl;										  // For debug
	std::cout << "PE m_stepLengthDampingFactorCur : " << myProcessID << " " << m_stepLengthDampingFactorCur << std::endl;						  // For debug
	std::cout << "PE m_numConsecutiveIterFunctionalDecreasing : " << myProcessID << " " << m_numConsecutiveIterFunctionalDecreasing << std::endl; // For debug
#endif

	if (myProcessID == 0 && (iynConverged == 1 || iynGoNextIteration == 1))
	{
		appendLCurveNonlinearCheckDiagnostics(
			iterCur,
			iCutbackCur,
			dataMisfit,
			rms,
			modelRoughness,
			stepLengthDampingFactorUsed);
	}

	if (iynConverged == 1)
	{
		return AnalysisControl::INVERSIN_CONVERGED;
	}
	else
	{
		if (iynGoNextIteration == 1)
		{
			return AnalysisControl::GO_TO_NEXT_ITERATION;
		}
		else
		{
			return AnalysisControl::DURING_RETRIALS;
		}
	}
}

bool AnalysisControl::checkConvergence(const double objectFunctionalCur, const int iter)
{

	const double criterion = m_decreaseRatioForConvegence * 0.01;

	if (m_ABICconverage)
	{
		return true;
	}

	if (iter <= (getIterationNumInit() + 1))
	{ // SONG240412-2RatiosForConverage
		if (m_objectFunctionalPre - objectFunctionalCur > 0.0 &&
			fabs(m_objectFunctionalPre - objectFunctionalCur) < 0.01 * m_objectFunctionalPre * criterion)
		{
			OutputFiles::m_cnvFile << "Iteration : " << iter << "." << std::endl;
			OutputFiles::m_cnvFile << "Coveraged : " << "1" << "." << std::endl;
			/*	std::cout << " Iteration : " << iter << "." << std::endl;
				std::cout << " Coveraged : " << "1" << "." << std::endl;*/
			return true;
		}
	}
	else
	{ // SONG240412-2RatiosForConverage
		if (m_objectFunctionalPre - objectFunctionalCur > 0.0 &&
			(m_objectFunctionalPre2 - m_objectFunctionalPre) < (m_objectFunctionalPre2 * criterion) &&
			(m_objectFunctionalPre - objectFunctionalCur) < (m_objectFunctionalPre * criterion))
		{
			OutputFiles::m_logFile << "Iteration : " << iter << "." << std::endl;
			OutputFiles::m_logFile << "Coveraged : " << "2" << "." << std::endl;
			OutputFiles::m_logFile << "m_objectFunctionalPre2 : " << m_objectFunctionalPre2 << "." << std::endl;
			OutputFiles::m_logFile << "m_objectFunctionalPre : " << m_objectFunctionalPre << "." << std::endl;
			OutputFiles::m_logFile << "objectFunctionalCur : " << objectFunctionalCur << "." << std::endl;
			return true;
		}
	}
	return false;
}

// Perform convergence test
bool AnalysisControl::checkConvergence(const double objectFunctionalCur, const double dataMisft, const double modelRoughness,
									   const double normDist1, const double normDist2)
{

	const double criterion = m_decreaseRatioForConvegence * 0.01;

	if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
	{
		if ((m_objectFunctionalPre - objectFunctionalCur) > 0.0 &&
			fabs(m_objectFunctionalPre - objectFunctionalCur) < m_objectFunctionalPre * criterion &&
			fabs(m_dataMisfitPre - dataMisft) < m_dataMisfitPre * criterion &&
			fabs(m_modelRoughnessPre - modelRoughness) < m_modelRoughnessPre * criterion &&
			fabs(m_normOfDistortionMatrixDifferencesPre - normDist1) < m_normOfDistortionMatrixDifferencesPre * criterion)
		{
			return true;
		}
	}
	else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
	{
		if ((m_objectFunctionalPre - objectFunctionalCur) > 0.0 &&
			fabs(m_objectFunctionalPre - objectFunctionalCur) < m_objectFunctionalPre * criterion &&
			fabs(m_dataMisfitPre - dataMisft) < m_dataMisfitPre * criterion &&
			fabs(m_modelRoughnessPre - modelRoughness) < m_modelRoughnessPre * criterion &&
			fabs(m_normOfGainsPre - normDist1) < m_normOfGainsPre * criterion &&
			fabs(m_normOfRotationsPre - normDist2) < m_normOfRotationsPre * criterion)
		{
			return true;
		}
	}
	else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
	{
		if ((m_objectFunctionalPre - objectFunctionalCur) > 0.0 &&
			fabs(m_objectFunctionalPre - objectFunctionalCur) < m_objectFunctionalPre * criterion &&
			fabs(m_dataMisfitPre - dataMisft) < m_dataMisfitPre * criterion &&
			fabs(m_modelRoughnessPre - modelRoughness) < m_modelRoughnessPre * criterion &&
			fabs(m_normOfGainsPre - normDist1) < m_normOfGainsPre * criterion)
		{
			return true;
		}
	}
	else
	{
		if ((m_objectFunctionalPre - objectFunctionalCur) > 0.0 &&
			fabs(m_objectFunctionalPre - objectFunctionalCur) < m_objectFunctionalPre * criterion &&
			fabs(m_dataMisfitPre - dataMisft) < m_dataMisfitPre * criterion &&
			fabs(m_modelRoughnessPre - modelRoughness) < m_modelRoughnessPre * criterion)
		{
			return true;
		}
	}
	return false;
}
