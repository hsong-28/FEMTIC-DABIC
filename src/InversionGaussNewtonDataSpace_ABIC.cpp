//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Han Song (c) 2025
//
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
#include "ObservedData.h"
#include "AnalysisControl.h"
#include "Forward3D.h"
#include "OutputFiles.h"
#include "ResistivityBlock.h"
#include "InversionGaussNewtonDataSpace_ABIC.h"
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#include "mpi.h"
#include <iomanip>

// Default constructer
InversionGaussNewtonDataSpace_ABIC::InversionGaussNewtonDataSpace_ABIC() : Inversion()
{
}

// constructer
InversionGaussNewtonDataSpace_ABIC::InversionGaussNewtonDataSpace_ABIC(const int nModel, const int nData)
	: Inversion(nModel, nData)
{
}

// Destructer
InversionGaussNewtonDataSpace_ABIC::~InversionGaussNewtonDataSpace_ABIC()
{
}

// Perform inversion
void InversionGaussNewtonDataSpace_ABIC::inversionCalculation()
{

	const bool useBLAS = true;
	// Get process ID and total process number
	const AnalysisControl *const ptrAnalysisControl = AnalysisControl::getInstance();
	const int myProcessID = ptrAnalysisControl->getMyPE();
	const int numProcessTotal = ptrAnalysisControl->getTotalPE();

#ifdef _DEBUG_WRITE
	std::ostringstream oss;
	oss << "debug_" << myProcessID << ".txt";
	std::ofstream fout;
	fout.open(oss.str().c_str());
#endif

	ObservedData *const ptrObservedData = ObservedData::getInstance();
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	const int nBlkNotFixed = ptrResistivityBlock->getNumResistivityBlockNotFixed();
	const int numModel = getNumberOfModel();
	const int nFreqThisPE = ptrObservedData->getNumOfFrequenciesCalculatedByThisPE();

	OutputFiles::m_logFile << "# Number of model : " << numModel << std::endl;
	OutputFiles::m_logFile << "# Number of frequencies of this PE : " << nFreqThisPE << std::endl;

	//---------------------------------
	// Calculate constraining matrix
	//---------------------------------

	OutputFiles::m_logFile << "# Calculate constraining matrix. " << ptrAnalysisControl->outputElapsedTime() << std::endl;
	RougheningMatrix constrainingMatrix; //for making RTR matrix

	// std::cout << "calcConstrainingMatrix" << std::endl;
	if (ptrAnalysisControl->useDifferenceFilter())
	{
		calcConstrainingMatrixForDifferenceFilter(constrainingMatrix); // α is included
	}
	else
	{
		calcConstrainingMatrix(constrainingMatrix);
	}
	//-----------------------------------------------
	// Calculate vector of model roughness
	//-----------------------------------------------
	// modelVector : m
	// dVector : RTRm = [R]T * [ b - [R]*m ]
	//-----------------------------------------------
	double *dVector = nullptr;
	if (myProcessID == 0)
	{
		double *modelVector = new double[numModel];
		double *modelNormVector = new double[numModel];
		ptrResistivityBlock->copyResistivityValuesNotFixedPreToVectorLog10(modelVector); 
		ptrObservedData->copyDistortionParamsNotFixedPreToVector(&modelVector[nBlkNotFixed]); //!!
#ifdef _DEBUG_WRITE
		std::cout << "Model roughness : " << constrainingMatrix.calcModelRoughness(modelVector) << std::endl;
#endif
		// if (ptrAnalysisControl->petrophysicalconstraint()) {
		const int numRows = constrainingMatrix.getNumRows();
		double *vectorRxm = new double[numRows];
		constrainingMatrix.calcVectorOfModelRoughness(modelVector, vectorRxm); //- [R]*m
		delete[] modelVector;
		if (dVector != nullptr)
		{
			delete[] dVector;
			dVector = nullptr;
		}
		dVector = new double[numModel];
		// dVector = - [R]T * [R]*m, α2 is included
		constrainingMatrix.calcMatrixVectorProductUsingTransposedMatrix(vectorRxm, dVector);
		delete[] vectorRxm;

		ptrResistivityBlock->calModelNormVectorLog10(modelNormVector); // α2 * β2 * （log10(m) - log10(mr)）
		//
		for (int iMdl = 0; iMdl < numModel; ++iMdl)
		{
			dVector[iMdl] += modelNormVector[iMdl];
		}
		delete[] modelNormVector;

	}

	//----------------------------------------------------------------------------------------
	// Make [R]T[R] matrix, where [R] is a constraining matrix
	//----------------------------------------------------------------------------------------
	// std::cout << "Make [R]T[R] matrix" << std::endl;
	OutputFiles::m_logFile << "# Make [R]T[R] matrix." << ptrAnalysisControl->outputElapsedTime() << std::endl;
	DoubleSparseSquareSymmetricMatrix RTRMatrix;
	if (ptrResistivityBlock->getFlagAddSmallValueToDiagonals())
	{
		// Add small value to diagonals of  [R]T[R] matrix
		constrainingMatrix.makeRTRMatrix(RTRMatrix, ptrResistivityBlock->getSmallValueAddedToDiagonals());
	}
	else
	{
		constrainingMatrix.makeRTRMatrix(RTRMatrix);
	}
	constrainingMatrix.releaseMemory();

	//----------------------------------
	// Initialization
	//----------------------------------
	std::ostringstream oocHeaderName;
	oocHeaderName << "ooc_temp_inv_3D_PE" << myProcessID;
	RTRMatrix.initializeMatrixSolver(oocHeaderName.str(), ptrAnalysisControl->getModeOfPARDISO());

	//----------------------------------
	// Analysis
	//----------------------------------
	OutputFiles::m_logFile << "# Analyse constraining matrix." << ptrAnalysisControl->outputElapsedTime() << std::endl;
	RTRMatrix.analysisPhaseMatrixSolver();

	//----------------------------------
	// Factorization
	//----------------------------------
	OutputFiles::m_logFile << "# Factorize constraining matrix." << ptrAnalysisControl->outputElapsedTime() << std::endl;
	RTRMatrix.factorizationPhaseMatrixSolver();

	//------------------------------------------------------------------------------------------------------
	// Calculate residual vector and sensitivity matrix multiplied by inverse of the constraning matrix
	//------------------------------------------------------------------------------------------------------
	int numDataThisPE = ptrObservedData->getNumObservedDataThisPETotal();
	OutputFiles::m_logFile << "# Number of data of this PE : " << numDataThisPE << std::endl;
#ifdef _DEBUG_WRITE
	for (int i = 0; i < numProcessTotal; ++i)
	{
		fout << "PE numDataThisPE : " << myProcessID << " " << numDataThisPE << std::endl;
	}
#endif
	double *residualVectorThisPE = new double[numDataThisPE];
	ptrAnalysisControl->getResidualVectorOfDataThisPE(residualVectorThisPE);
	OutputFiles::m_logFile << "# Get Residual Vector Of Data this PE : " << numDataThisPE << std::endl;

#ifdef _DEBUG_WRITE
	for (int i = 0; i < numDataThisPE; ++i)
	{
		fout << "PE i residualVectorThisPE[i] " << myProcessID << " " << i << " " << residualVectorThisPE[i] << std::endl;
	}
#endif

	double *vectorJTxResidialLocal = new double[numModel];
	for (int iMdl = 0; iMdl < numModel; ++iMdl)
	{
		vectorJTxResidialLocal[iMdl] = 0.0; // Zero clear
	}
	int offset(0);
	for (int iFreq = 0; iFreq < nFreqThisPE; ++iFreq)
	{
		assert(offset == ptrObservedData->getNumObservedDataThisPEAccumulated(iFreq)); // m_numObservedDataThisPEAccumulated[0]=0;
		const int freqID = ptrObservedData->getIDsOfFrequenciesCalculatedByThisPE(iFreq);

		std::ostringstream fileName;
		fileName << "sensMatFreq" << freqID;
		OutputFiles::m_logFile << "# Read sensitivity matrix from " << fileName.str() << "." << ptrAnalysisControl->outputElapsedTime() << std::endl;
		int numDataThisFreq(0);
		int numModelTemp(0);
		double *sensitivityMatrix = nullptr;
		readSensitivityMatrix(fileName.str(), numDataThisFreq, numModelTemp, sensitivityMatrix);
		if (numDataThisFreq != ptrObservedData->getNumObservedDataThisPE(iFreq))
		{
			OutputFiles::m_logFile << "Error : Number of data written in out-of-core file is unequal to the internal one !!" << std::endl;
			exit(1);
		}
		if (numModel != numModelTemp)
		{
			OutputFiles::m_logFile << "Error : Number of model written in out-of-core file is unequal to the internal one !!" << std::endl;
			exit(1);
		}

#ifdef _DEBUG_WRITE
		fout << "read sensitivity matrix at inversionCalculation. iFreq = " << iFreq << std::endl;
		fout << "sensitivityMatrix" << std::endl;
		for (int idat = 0; idat < numDataThisFreq; ++idat)
		{
			for (int imdl = 0; imdl < numModel; ++imdl)
			{
				fout << "idat, imdl, val " << idat + offset << " " << imdl << " " << sensitivityMatrix[imdl + idat * numModel] << std::endl;
			}
		}
#endif
		//-----------------------------------------------
		// Matrix-vector product
		//-----------------------------------------------
		if (useBLAS)
		{
			CBLAS_ORDER order = CblasRowMajor;
			CBLAS_TRANSPOSE trans = CblasTrans;
			MKL_INT m = static_cast<MKL_INT>(numDataThisFreq);
			MKL_INT n = static_cast<MKL_INT>(numModel);
			double alpha = 1.0;
			double beta = 1.0;
			MKL_INT lda = n;
			MKL_INT incx = 1.0;
			MKL_INT incy = 1.0;
			cblas_dgemv(order, trans, m, n, alpha, sensitivityMatrix, lda, &residualVectorThisPE[offset], incx, beta, vectorJTxResidialLocal, incy);
		}
		else
		{
			const long long numModel_64 = static_cast<long long>(numModel);
			const long long offset_64 = static_cast<long long>(offset);
			const long long nDataEnd = offset_64 + static_cast<long long>(numDataThisFreq);
			for (long long iDat = offset_64; iDat < nDataEnd; ++iDat)
			{
				const double val = residualVectorThisPE[iDat];
				const long long offsetTemp = (iDat - offset_64) * numModel_64;
				for (long long iMdl = 0; iMdl < numModel_64; ++iMdl)
				{
					vectorJTxResidialLocal[iMdl] += sensitivityMatrix[iMdl + offsetTemp] * val;
				}
			}
		}
		//-----------------------------------------------

		//-------------------------------------------------------------------------------------
		// Calculate sensitivity matrix multiplied by inverse of constraining matrix
		//-------------------------------------------------------------------------------------
		// sensitivityMatrix : JT => inv[RTR]*[J]T
		//-------------------------------------------------------------------------------------
		OutputFiles::m_logFile << "# Multiply transposed sensitivity matrix by inverse of constraining matrix. " << ptrAnalysisControl->outputElapsedTime() << std::endl;
		const long long numComps = static_cast<long long>(numDataThisFreq) * static_cast<long long>(numModel);
		double *sensitivityMatrixTemp = new double[numComps];

		int numDivRhs = ptrAnalysisControl->getDivisionNumberOfMultipleRHSInInversion(); // numDivRhs=1;
		assert(numDivRhs > 0);
		if (numDivRhs > numDataThisFreq)
		{
			OutputFiles::m_logFile << "Warning : Division number of right-hand sides ( " << numDivRhs << " ) is greater than the total number of right-hand sides ( " << numDataThisFreq << " )." << std::endl;
			OutputFiles::m_logFile << "          Thus, the division number is set to be the total number of right-hand sides." << std::endl;
			numDivRhs = numDataThisFreq;
		}

		const int numRHSDividedWithoutOdds = numDataThisFreq / numDivRhs; // numRHSDividedWithoutOdds=numDataThisFreq;
		const int numAdds = numDataThisFreq % numDivRhs;				  // numAdds=0;
		long long iRhsStart = 0;
		for (int iDiv = 0; iDiv < numDivRhs; ++iDiv)
		{
			const int numRHSDivided = iDiv < numAdds ? numRHSDividedWithoutOdds + 1 : numRHSDividedWithoutOdds;
			OutputFiles::m_logFile << "# Solve phase is performed simultaneously for " << numRHSDivided << " right-hand sides" << ptrAnalysisControl->outputElapsedTime() << std::endl;
			const long long index = static_cast<long long>(numModel) * iRhsStart;
			RTRMatrix.solvePhaseMatrixSolver(numRHSDivided, &sensitivityMatrix[index], &sensitivityMatrixTemp[index]); // Solve
			iRhsStart += numRHSDivided;
		}

		//------------------------------------------------------------------------------------
		// Write sensitivity matrix multiplied by inverse of constraining matrix
		//------------------------------------------------------------------------------------
		fileName << "Mod";
		FILE *fp = nullptr;
		fp = fopen(fileName.str().c_str(), "wb");
		if (fp == nullptr)
		{
			OutputFiles::m_logFile << "File open error !! : " << fileName.str() << std::endl;
			exit(1);
		}
		fwrite(&numDataThisFreq, sizeof(int), 1, fp);
		fwrite(&numModel, sizeof(int), 1, fp);
		fwrite(sensitivityMatrixTemp, sizeof(double), numComps, fp);
		fclose(fp);
		//------------------------------------------------------------------------------------

		delete[] sensitivityMatrix;
		delete[] sensitivityMatrixTemp;

		offset += numDataThisFreq; // Add data number
	} //

	delete[] residualVectorThisPE;
	if (myProcessID != 0)
	{
		// PE = 0 uses constraining matrix later
		RTRMatrix.releaseMemory();
	}

	//-------------------------------------------------
	// Merge vectors
	//-------------------------------------------------
	double *vectorJTxResidialGlobal = nullptr;
	if (myProcessID == 0)
	{
		vectorJTxResidialGlobal = new double[numModel];
	}
	// MPI_Allreduce( vectorJTxResidialLocal, vectorJTxResidialGlobal, numModel, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
	MPI_Reduce(vectorJTxResidialLocal, vectorJTxResidialGlobal, numModel, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	delete[] vectorJTxResidialLocal;

#ifdef _DEBUG_WRITE
	if (myProcessID == 0)
	{
		for (int iMdl = 0; iMdl < numModel; ++iMdl)
		{
			fout << "iMdl, vectorJTxResidialGlobal : " << iMdl << " " << vectorJTxResidialGlobal[iMdl] << std::endl;
		}
	}
#endif

	//-------------------------------------------------
	// Calculate Inv[ [R]T*[R] ]*d vector
	//-------------------------------------------------
	double *vectorInvRTRd = new double[numModel];
	if (myProcessID == 0)
	{ //----- Processs ID = 0 Only ----->>>>>
		//------------------------------------------
		// vectorJTxResidialGlobal : [J]T*rd
		// dVector : RTRm => RTRm + [J]T*rd
		//------------------------------------------
		for (int iMdl = 0; iMdl < numModel; ++iMdl)
		{
			dVector[iMdl] += vectorJTxResidialGlobal[iMdl];
		}
		delete[] vectorJTxResidialGlobal;

#ifdef _DEBUG_WRITE
		for (int iMdl = 0; iMdl < numModel; ++iMdl)
		{
			fout << "iMdl, dVector : " << iMdl << " " << dVector[iMdl] << std::endl;
		}
#endif

		//-------------------------------------------------
		// dVector : RTRm + [J]T*rd
		// vectorInvRTRd : inv[ [R]T*[R] ] * [ RTRm + [J]T*rd ]
		//-------------------------------------------------
		RTRMatrix.solvePhaseMatrixSolver(1, dVector, vectorInvRTRd);

#ifdef _DEBUG_WRITE
		for (int iMdl = 0; iMdl < numModel; ++iMdl)
		{
			fout << "iMdl, vectorInvRTRd : " << iMdl << " " << vectorInvRTRd[iMdl] << std::endl;
		}
#endif
	} //---- Processs ID = 0 Only -----<<<<<
	MPI_Bcast(vectorInvRTRd, numModel, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//-------------------------------------------------
	// Calculate right-hand side vector
	//-------------------------------------------------
	OutputFiles::m_logFile << "# Calculate right-hand side vector. " << ptrAnalysisControl->outputElapsedTime() << std::endl;
	double *rhsVectorLocal = new double[numDataThisPE];
	for (int i = 0; i < numDataThisPE; ++i)
	{
		rhsVectorLocal[i] = 0.0; // Zero clear
	}
	offset = 0;
	for (int iFreq = 0; iFreq < nFreqThisPE; ++iFreq)
	{
		assert(offset == ptrObservedData->getNumObservedDataThisPEAccumulated(iFreq));

		const int freqID = ptrObservedData->getIDsOfFrequenciesCalculatedByThisPE(iFreq);

		std::ostringstream fileName;
		fileName << "sensMatFreq" << freqID;

		OutputFiles::m_logFile << "# Read sensitivity matrix from " << fileName.str() << "." << ptrAnalysisControl->outputElapsedTime() << std::endl;
		int numDataThisFreq(0);
		int numModelTemp(0);
		double *sensitivityMatrix = nullptr;
		readSensitivityMatrix(fileName.str(), numDataThisFreq, numModelTemp, sensitivityMatrix);
		//(0)=delt(d0)/delt(m0),(1)=delt(d0)/delt(m1),(2)delt(d0)/delt(m2),...,(M-1)delt(d0)/delt(mM)

		if (numDataThisFreq != ptrObservedData->getNumObservedDataThisPE(iFreq))
		{
			OutputFiles::m_logFile << "Error : Number of data written in out-of-core file is unequal to the internal one !!" << std::endl;
			exit(1);
		}
		if (numModel != numModelTemp)
		{
			OutputFiles::m_logFile << "Error : Number of model written in out-of-core file is unequal to the internal one !!" << std::endl;
			exit(1);
		}

		if (useBLAS)
		{
			CBLAS_ORDER order = CblasRowMajor;
			CBLAS_TRANSPOSE trans = CblasNoTrans;
			MKL_INT m = static_cast<MKL_INT>(numDataThisFreq);
			MKL_INT n = static_cast<MKL_INT>(numModel);
			double alpha = 1.0;
			double beta = 0.0;
			MKL_INT lda = n;
			MKL_INT incx = 1.0;
			MKL_INT incy = 1.0;
			cblas_dgemv(order, trans, m, n, alpha, sensitivityMatrix, lda, vectorInvRTRd, incx, beta, &rhsVectorLocal[offset], incy);
		}
		else
		{
			long long iDat(0);
			long long offsetTemp(0);
			double work(0.0);
			long long iMdl(0);
			const long long numDataThisFreq_64 = static_cast<long long>(numDataThisFreq);
			const long long numModel_64 = static_cast<long long>(numModel);
			const long long offset_64 = static_cast<long long>(offset);
#ifdef _USE_OMP
#pragma omp parallel for default(shared) private(iDat, offsetTemp, work, iMdl)
#endif
			for (iDat = 0; iDat < numDataThisFreq_64; ++iDat)
			{
				offsetTemp = iDat * numModel_64;
				work = 0.0;
				for (iMdl = 0; iMdl < numModel_64; ++iMdl)
				{
					work += sensitivityMatrix[iMdl + offsetTemp] * vectorInvRTRd[iMdl];
				}
				rhsVectorLocal[offset_64 + iDat] = work;
			}
		}

		delete[] sensitivityMatrix;

		offset += numDataThisFreq; // Add data number
	}
	delete[] vectorInvRTRd;

#ifdef _DEBUG_WRITE
	for (int i = 0; i < numDataThisPE; ++i)
	{
		fout << "i rhsVectorLocal[i] " << i << " " << rhsVectorLocal[i] << std::endl;
	}
#endif

	//------------------------------------------------------------
	// Make array for garhering
	//------------------------------------------------------------
	int *numDataLocal = nullptr;
	if (myProcessID == 0)
	{
		numDataLocal = new int[numProcessTotal];
	}

	// MPI_Allgather( &numDataThisPE, 1, MPI_INT, numDataLocal, 1, MPI_INT, MPI_COMM_WORLD );
	MPI_Gather(&numDataThisPE, 1, MPI_INT, numDataLocal, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int *numDataAccumulated = nullptr;
	int numDataTotal(-1);
	if (myProcessID == 0)
	{
#ifdef _DEBUG_WRITE
		for (int i = 0; i < numProcessTotal; ++i)
		{
			fout << "PE i numDataLocal[i] : " << myProcessID << " " << i << " " << numDataLocal[i] << std::endl;
		}
#endif
		numDataAccumulated = new int[numProcessTotal + 1];
		numDataAccumulated[0] = 0;
		for (int i = 0; i < numProcessTotal; ++i)
		{
			numDataAccumulated[i + 1] = numDataAccumulated[i] + numDataLocal[i];
		}

		numDataTotal = numDataAccumulated[numProcessTotal];
		OutputFiles::m_logFile << "# Number of total data  : " << numDataTotal << std::endl;

#ifdef _DEBUG_WRITE
		for (int i = 0; i < numProcessTotal + 1; ++i)
		{
			fout << "PE i numDataAccumulated[i] : " << myProcessID << " " << i << " " << numDataAccumulated[i] << std::endl;
		}
		fout << "PE numDataTotal : " << myProcessID << " " << numDataTotal << std::endl;
#endif
	}

	//------------------------------------------------------------
	// Merge right-hands-side vector
	//------------------------------------------------------------
	double *rhsVectorGlobal = nullptr;
	if (myProcessID == 0)
	{
		rhsVectorGlobal = new double[numDataTotal];
	}
	MPI_Gatherv(rhsVectorLocal, numDataThisPE, MPI_DOUBLE, rhsVectorGlobal, numDataLocal, numDataAccumulated, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	delete[] rhsVectorLocal;

#ifdef _DEBUG_WRITE
	if (myProcessID == 0)
	{
		fout << "RHS vector" << std::endl;
		for (int i = 0; i < numDataTotal; ++i)
		{
			fout << "i rhsVectorGlobal[i] " << i << " " << rhsVectorGlobal[i] << std::endl;
		}
	}
#endif

	//------------------------------------------------------------
	// Calculate coefficient matrix
	//------------------------------------------------------------
	double *matrixToBeInverted = nullptr;
	if (myProcessID == 0)
	{ //---- Processs ID = 0 Only ----->>>>>
		OutputFiles::m_logFile << "# Calculate coefficient matrix. " << ptrAnalysisControl->outputElapsedTime() << std::endl;
		const long long int numDataTotal_64 = static_cast<long long int>(numDataTotal);
		double *matrix = new double[static_cast<long long>(numDataTotal) * static_cast<long long>(numDataTotal)];
#ifdef _DEBUG_WRITE
		for (long long i = 0; i < static_cast<long long>(numDataTotal) * static_cast<long long>(numDataTotal); ++i)
		{
			matrix[i] = -9999.999;
		}
#endif

		//--------------------------------------------------------------
		// Read sensitivity matrix and calculate matrix-matrix product
		//--------------------------------------------------------------
		const int nFreq = ptrObservedData->getTotalNumberOfDifferenetFrequencies();
		int offsetRows(0);
		for (int ifreqLeft = 0; ifreqLeft < nFreq; ++ifreqLeft)
		{
			std::ostringstream fileNameLeft;
			fileNameLeft << "sensMatFreq" << ifreqLeft;
			OutputFiles::m_logFile << "# Read sensitivity matrix from " << fileNameLeft.str() << "." << ptrAnalysisControl->outputElapsedTime() << std::endl;

			int numDataThisFreqLeft(0);
			int numModelTempLeft(0);
			double *sensitivityMatrixLeft = nullptr;
			readSensitivityMatrix(fileNameLeft.str(), numDataThisFreqLeft, numModelTempLeft, sensitivityMatrixLeft);
			if (numModel != numModelTempLeft)
			{
				OutputFiles::m_logFile << "Error : Number of model written in out-of-core file is unequal to the internal one !!" << std::endl;
				exit(1);
			}

			int offsetCols = offsetRows;
			for (int ifreqRight = ifreqLeft; ifreqRight < nFreq; ++ifreqRight)
			{
				std::ostringstream fileNameRight;
				fileNameRight << "sensMatFreq" << ifreqRight << "Mod";
				OutputFiles::m_logFile << "# Read sensitivity matrix from " << fileNameRight.str() << "." << ptrAnalysisControl->outputElapsedTime() << std::endl;

				int numDataThisFreqRight(0);
				int numModelTempRight(0);
				double *sensitivityMatrixRight = nullptr;
				readSensitivityMatrix(fileNameRight.str(), numDataThisFreqRight, numModelTempRight, sensitivityMatrixRight);
				if (numModel != numModelTempRight)
				{
					OutputFiles::m_logFile << "Error : Number of model written in out-of-core file is unequal to the internal one !!" << std::endl;
					exit(1);
				}
				//-----------------------------
				//--- Matrix-Matrix product ---
				//-----------------------------
				if (useBLAS)
				{
					double *result = new double[static_cast<long long>(numDataThisFreqLeft) * static_cast<long long>(numDataThisFreqRight)];
					MKL_INT m = static_cast<MKL_INT>(numDataThisFreqLeft);
					MKL_INT n = static_cast<MKL_INT>(numDataThisFreqRight);
					MKL_INT k = static_cast<MKL_INT>(numModel);
					MKL_INT lda = k;
					MKL_INT ldb = k;
					MKL_INT ldc = n;
					double alpha = 1.0;
					double beta = 0.0;
					CBLAS_ORDER order = CblasRowMajor;
					CBLAS_TRANSPOSE transA = CblasNoTrans;
					CBLAS_TRANSPOSE transB = CblasTrans;
					cblas_dgemm(order, transA, transB, m, n, k, alpha, sensitivityMatrixLeft, lda, sensitivityMatrixRight, ldb, beta, result, ldc);
					for (long long irow = 0; irow < numDataThisFreqLeft; ++irow)
					{
						const long long row = irow + static_cast<long long>(offsetRows);
						for (long long icol = 0; icol < numDataThisFreqRight; ++icol)
						{
							const long long col = icol + static_cast<long long>(offsetCols);
							matrix[row * static_cast<long long>(numDataTotal) + col] = result[irow * static_cast<long long>(numDataThisFreqRight) + icol];
						}
					}
					delete[] result;
				}
				else
				{
					int iDatLeft(0);
					int iDatRight(0);
					int offsetLeft(0);
					int offsetRight(0);
					int index(0);
					double work(0.0);
					long long iMdl(0);
					const long long numDataThisFreqLeft_64 = static_cast<long long>(numDataThisFreqLeft);
					const long long numDataThisFreqRight_64 = static_cast<long long>(numDataThisFreqRight);
					const long long numModel_64 = static_cast<long long>(numModel);
					const long long numDataTotal_64 = static_cast<long long>(numDataTotal);
					const long long offsetRows_64 = static_cast<long long>(offsetRows);
					const long long offsetCols_64 = static_cast<long long>(offsetCols);
#ifdef _USE_OMP
#pragma omp parallel for default(shared) private(iDatLeft, iDatRight, offsetLeft, offsetRight, index, work, iMdl)
#endif
					for (iDatLeft = 0; iDatLeft < numDataThisFreqLeft_64; ++iDatLeft)
					{
						offsetLeft = iDatLeft * numModel_64;
						for (iDatRight = 0; iDatRight < numDataThisFreqRight_64; ++iDatRight)
						{
							offsetRight = iDatRight * numModel_64;
							work = 0.0;
							for (iMdl = 0; iMdl < numModel_64; ++iMdl)
							{
								work += sensitivityMatrixLeft[offsetLeft + iMdl] * sensitivityMatrixRight[offsetRight + iMdl];
							}
							index = (iDatLeft + offsetRows_64) * numDataTotal_64 + (iDatRight + offsetCols_64);
							matrix[index] = work;
						}
					}
				}
				//-----------------------------
				delete[] sensitivityMatrixRight;
				offsetCols += numDataThisFreqRight; // Increment offset of column number
			}
			delete[] sensitivityMatrixLeft;
			offsetRows += numDataThisFreqLeft; // Increment offset of row number
		}

		//----------------------------------------------
		// Add unit matrix //
		//----------------------------------------------
		for (long long row = 0; row < numDataTotal_64; ++row)
		{
			const long long col = row;
			matrix[row * numDataTotal_64 + col] += 1.0;
		}

		const long long numElemsOfCoefficientMatrixTotal = (numDataTotal_64) * ((numDataTotal_64) + 1) / 2;
		OutputFiles::m_logFile << "# Total number of elements in coefficient matrix : " << numElemsOfCoefficientMatrixTotal << std::endl;
		matrixToBeInverted = new double[numElemsOfCoefficientMatrixTotal];

		//----------------------------------------------
		// Copy only upper triangle components
		//----------------------------------------------
		long long index(0);
		// std::ofstream outfilematrix("matrix_values1.txt", std::ios::out); //
		for (long long row = 0; row < numDataTotal; ++row)
		{
			for (long long col = row; col < numDataTotal; ++col)
			{
				matrixToBeInverted[index] = matrix[row * numDataTotal + col];
				// outfilematrix << row <<" "<< col << " "<< matrixToBeInverted[index] << "\n"; //
				++index;
			}
		}

#ifdef _DEBUG_WRITE
		for (int i = 0; i < numElemsOfCoefficientMatrixTotal; ++i)
		{
			fout << "i, matrixToBeInverted : " << i << " " << matrixToBeInverted[i] << std::endl;
		}
#endif
		delete[] matrix;

		//----------------------------------------------
		// Numerical factorization with lapack
		//----------------------------------------------
		OutputFiles::m_logFile << "# Start numerical factorization for normal equation. " << ptrAnalysisControl->outputElapsedTime() << std::endl;

		const long long int numModel_64 = static_cast<long long int>(numModel);
		const bool positiveDefinite = ptrAnalysisControl->getPositiveDefiniteNormalEqMatrix();
		long long int *ipiv = NULL;
		if (!positiveDefinite)
		{
			ipiv = new long long int[numDataTotal_64];
		}

		long long int ierr(0);
		if (positiveDefinite)
		{
			ierr = LAPACKE_dpptrf(LAPACK_COL_MAJOR, 'L', numDataTotal_64, matrixToBeInverted);
		}
		else
		{
			ierr = LAPACKE_dsptrf(LAPACK_COL_MAJOR, 'L', numDataTotal_64, matrixToBeInverted, ipiv);
		}

		if (ierr > 0)
		{
			OutputFiles::m_logFile << "Error : Matrix is singular. ierr = " << ierr << std::endl;
			exit(1);
		}
		else if (ierr < 0)
		{
			OutputFiles::m_logFile << "Error : " << -ierr << "-th parameter has illegal value." << std::endl;
			exit(1);
		}

		// Calculate determinant from block diagonal matrix D (matrixToBeInverted after "LAPACKE_dsptrf")
		// https://netlib.org/lapack/explore-html/d8/d08/group__hptrf_ga9adb8a8020b104eaaf72c2aad7861a8c.html
		double determinant = 0.0; // Store ln of determinant for numerical stability
		int tem = 0;
		// std::cout << " # calculate the determinant of matrix(I + J inv(RTR)JT) ... " << std::endl;
		if (positiveDefinite)
		// LAPACKE_dpptrf : A = [L] [L]T
		// Det(A) = pow(Det(L),2.0)
		// ln(Det(A)) = 2 * ln(Det(L))
		{
			for (int i = 0; i < numDataTotal_64; ++i)
			{
				determinant += std::log(std::fabs(matrixToBeInverted[tem])); // ln of diagonal element
			}
			determinant = determinant * 2.0;
		}
		else
		{
		// LAPACKE_dsptrf: A = [L] [D] [L]T
		// The diagonal elements of L are 1; therefore, Det(A) = Det(D), where D is a diagonal (or block diagonal) matrix.
		// The elements of D overwrite the diagonal elements of matrix A.
			for (int i = 0; i < numDataTotal_64; ++i)
			{
				if (ipiv[i] > 0)
				{
					// 1x1 block
					determinant += std::log(std::fabs(matrixToBeInverted[tem])); // ln of diagonal element
					tem += (numDataTotal_64 - i);								 // Move to next diagonal element
				}
				else
				{
					// 2x2 block in D (symmetric, stored in lower triangle)
					int n = numDataTotal_64;
					// Extract elements of D's 2x2 block
					double a = matrixToBeInverted[tem];			  // D[i][i] (overwrites A[i][i])
					double b = matrixToBeInverted[tem + 1];		  // D[i+1][i] (overwrites A[i+1][i], symmetric to D[i][i+1])
					double d = matrixToBeInverted[tem + (n - i)]; // D[i+1][i+1] (overwrites A[i+1][i+1], start of column i+1)

					// Determinant of 2x2 symmetric block: ad - b²
					double detBlock = a * d - b * b;
					determinant += std::log(std::fabs(detBlock));

					// Update tem: skip all elements in column i and i+1
					// Column i has (n - i) elements, column i+1 has (n - (i+1)) elements
					tem += (n - i) + (n - i - 1);

					// Skip next index as 2x2 block covers i and i+1
					i++; // Increment i to avoid reprocessing i+1
				}
			}
		}
		// std::cout << " # determinant = " << determinant << std::endl;
		setdeterminant(determinant);

		//----------------------------------------------
		// Solver linear equation with lapack
		//----------------------------------------------
		OutputFiles::m_logFile << "# Start solve phase for normal equation. " << ptrAnalysisControl->outputElapsedTime() << std::endl;
		const long long int nrhs = 1;
		const long long int ldb = numDataTotal_64;
		if (positiveDefinite)
		{
			ierr = LAPACKE_dpptrs(LAPACK_COL_MAJOR, 'L', numDataTotal_64, nrhs, matrixToBeInverted, rhsVectorGlobal, ldb);
		}
		else
		{
			ierr = LAPACKE_dsptrs(LAPACK_COL_MAJOR, 'L', numDataTotal_64, nrhs, matrixToBeInverted, ipiv, rhsVectorGlobal, ldb);
		}

		if (ierr < 0)
		{
			OutputFiles::m_logFile << "Error : " << -ierr << "-th parameter has illegal value." << std::endl;
			exit(1);
		}

		if (matrixToBeInverted != NULL)
		{
			delete[] matrixToBeInverted;
			matrixToBeInverted = NULL;
		}
		if (!positiveDefinite)
		{
			delete[] ipiv;
			ipiv = NULL;
		}

#ifdef _DEBUG_WRITE
		for (int iDat = 0; iDat < numDataTotal; ++iDat)
		{
			fout << "iDat, rhsVectorGlobal : " << iDat << " " << rhsVectorGlobal[iDat] << std::endl;
		}
#endif

	} //----- Treadted by only PE 0 ------------------<<<<<<<<<<<<<<<<<<<<<<<<

	//-----------------------------------------------------------------
	// Scatter result vector
	//-----------------------------------------------------------------
	double *dataVectorThisPE = new double[numDataThisPE];
	MPI_Scatterv(rhsVectorGlobal, numDataLocal, numDataAccumulated, MPI_DOUBLE, dataVectorThisPE, numDataThisPE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rhsVectorGlobal != nullptr)
	{
		delete[] rhsVectorGlobal;
		rhsVectorGlobal = nullptr;
	}
	if (numDataLocal != nullptr)
	{
		delete[] numDataLocal;
		numDataLocal = nullptr;
	}
	if (numDataAccumulated != nullptr)
	{
		delete[] numDataAccumulated;
		numDataAccumulated = nullptr;
	}

#ifdef _DEBUG_WRITE
	for (int iDat = 0; iDat < numDataThisPE; ++iDat)
	{
		fout << "iDat, dataVectorThisPE : " << iDat << " " << dataVectorThisPE[iDat] << std::endl;
	}
#endif

	//----------------------------------------------------------------------------------------------------------------
	// Matrix-vector product
	//----------------------------------------------------------------------------------------------------------------
	// dataVectorThisPE : inv[ [I] + [J] * inv[ [R]T*[R] ] *[J]T ] * [J] * inv[ [R]T*[R] ] * [ RTRm + [J]T*rd ]
	// workVector : [J]T * inv[ [I] + [J] * inv[ [R]T*[R] ] *[J]T ] * [J] * inv[ [R]T*[R] ] * [ RTRm + [J]T*rd ]
	//----------------------------------------------------------------------------------------------------------------
	double *workVector = new double[numModel];
	for (int iMdl = 0; iMdl < numModel; ++iMdl)
	{
		workVector[iMdl] = 0.0; // Zero clear
	}
	offset = 0;
	for (int iFreq = 0; iFreq < nFreqThisPE; ++iFreq)
	{
		assert(offset == ptrObservedData->getNumObservedDataThisPEAccumulated(iFreq));
		const int freqID = ptrObservedData->getIDsOfFrequenciesCalculatedByThisPE(iFreq);
		std::ostringstream fileName;
		fileName << "sensMatFreq" << freqID;
		OutputFiles::m_logFile << "# Read sensitivity matrix from " << fileName.str() << "." << ptrAnalysisControl->outputElapsedTime() << std::endl;
		int numDataThisFreq(0);
		int numModelTemp(0);
		double *sensitivityMatrix = nullptr;
		readSensitivityMatrix(fileName.str(), numDataThisFreq, numModelTemp, sensitivityMatrix);
		if (numDataThisFreq != ptrObservedData->getNumObservedDataThisPE(iFreq))
		{
			OutputFiles::m_logFile << "Error : Number of data written in out-of-core file is unequal to the internal one !!" << std::endl;
			exit(1);
		}
		if (numModel != numModelTemp)
		{
			OutputFiles::m_logFile << "Error : Number of model written in out-of-core file is unequal to the internal one !!" << std::endl;
			exit(1);
		}
		//-----------------------------------------------
		// Matrix-vector product
		//-----------------------------------------------
		if (useBLAS)
		{
			CBLAS_ORDER order = CblasRowMajor;
			CBLAS_TRANSPOSE trans = CblasTrans;
			MKL_INT m = static_cast<MKL_INT>(numDataThisFreq);
			MKL_INT n = static_cast<MKL_INT>(numModel);
			double alpha = 1.0;
			double beta = 1.0;
			MKL_INT lda = n;
			MKL_INT incx = 1.0;
			MKL_INT incy = 1.0;
			cblas_dgemv(order, trans, m, n, alpha, sensitivityMatrix, lda, &dataVectorThisPE[offset], incx, beta, workVector, incy);
		}
		else
		{
			const long long numModel_64 = static_cast<long long>(numModel);
			const long long offset_64 = static_cast<long long>(offset);
			const long long nDataEnd = offset_64 + static_cast<long long>(numDataThisFreq);
			for (long long iDat = offset_64; iDat < nDataEnd; ++iDat)
			{
				const long long offsetTemp = (iDat - offset_64) * numModel_64;
				const double val = dataVectorThisPE[iDat];
				for (long long iMdl = 0; iMdl < numModel_64; ++iMdl)
				{
					workVector[iMdl] += sensitivityMatrix[iMdl + offsetTemp] * val;
				}
			}
		}
		//-----------------------------------------------
		delete[] sensitivityMatrix;
		offset += numDataThisFreq; // Add data number
	}

	if (dataVectorThisPE != nullptr)
	{
		delete[] dataVectorThisPE;
		dataVectorThisPE = nullptr;
	}

#ifdef _DEBUG_WRITE
	for (int iMdl = 0; iMdl < numModel; ++iMdl)
	{
		fout << "iMdl, workVector : " << iMdl << " " << workVector[iMdl] << std::endl;
	}
#endif

	double *workVectorSum = nullptr;
	if (myProcessID == 0)
	{
		workVectorSum = new double[numModel];
	}
	MPI_Reduce(workVector, workVectorSum, numModel, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (workVector != nullptr)
	{
		delete[] workVector;
		workVector = nullptr;
	}

#ifdef _DEBUG_WRITE
	if (myProcessID == 0)
	{
		for (int iMdl = 0; iMdl < numModel; ++iMdl)
		{
			fout << "iMdl, workVectorSum : " << iMdl << " " << workVectorSum[iMdl] << std::endl;
		}
	}
#endif
	double *resultVector = new double[numModel];
	if (myProcessID == 0)
	{
		//-------------------------------------------------------------------------------------------------------------------
		// workVectorSum : [J]T * inv[ [I] + [J] * inv[ [R]T*[R] ] *[J]T ] * [J] * inv[ [R]T*[R] ] * [ RTRm + [J]T*rd ]
		// dVector : RTRm + [J]T*rd
		//   => [ RTRm + [J]T*rd ] - [J]T * inv[ [I] + [J] * inv[ [R]T*[R] ] *[J]T ] * [J] * inv[ [R]T*[R] ] * [ RTRm + [J]T*rd ]
		//-------------------------------------------------------------------------------------------------------------------
		for (int iMdl = 0; iMdl < numModel; ++iMdl)
		{
			dVector[iMdl] -= workVectorSum[iMdl];
		}
		delete[] workVectorSum;
		//-------------------------------------------------------------------------------------------------------------------

#ifdef _DEBUG_WRITE
		for (int iMdl = 0; iMdl < numModel; ++iMdl)
		{
			fout << "iMdl, dVector : " << iMdl << " " << dVector[iMdl] << std::endl;
		}
#endif
		//-------------------------------------------------------------------------------------------------------------------
		// dVector : [ RTRm + [J]T*rd ] - [J]T * inv[ [I] + [J] * inv[ [R]T*[R] ] *[J]T ] * [J] * inv[ [R]T*[R] ] * [ RTRm + [J]T*rd ]
		// resultVector : inv[ [R]T*[R] ] * [ RTRm + [J]T*rd ] - inv[ [R]T*[R] ] * [J]T * inv[ [I] + [J] * inv[ [R]T*[R] ] *[J]T ] * [J] * inv[ [R]T*[R] ] * [ RTRm + [J]T*rd ]
		//-------------------------------------------------------------------------------------------------------------------
		RTRMatrix.solvePhaseMatrixSolver(1, dVector, resultVector);
		RTRMatrix.releaseMemory();
		delete[] dVector;
		//-------------------------------------------------------------------------------------------------------------------
	}
	MPI_Bcast(resultVector, numModel, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef _DEBUG_WRITE
	for (int iMdl = 0; iMdl < numModel; ++iMdl)
	{
		fout << "iMdl, resultVector : " << iMdl << " " << resultVector[iMdl] << std::endl;
	}
#endif
	//=================================================================
	// Calculate model increments
	//=================================================================
	OutputFiles::m_logFile << "# Calculate model increments. " << ptrAnalysisControl->outputElapsedTime() << std::endl;

	//=================================================================
	// Update resistivity values
	//=================================================================
	ptrResistivityBlock->calctResistivityUpdatedFullFromLog10ResistivityIncres(resultVector);
	ptrResistivityBlock->updateResistivityValues();

	ptrObservedData->calcDistortionParamsUpdatedFullFromIncrements(&resultVector[nBlkNotFixed]);
	ptrObservedData->updateDistortionParams();
	double mupdatedratio = ptrResistivityBlock->calctResistivityUpdatedratio();

	delete[] resultVector;
	//=================================================================
	// Delete old out-of-core files
	//=================================================================
	for (int ifreq = 0; ifreq < nFreqThisPE; ++ifreq)
	{
		const int freqID = ptrObservedData->getIDsOfFrequenciesCalculatedByThisPE(ifreq);
		std::ostringstream fileName;
		fileName << "sensMatFreq" << freqID;
		fileName << "Mod";
		if (remove(fileName.str().c_str()) != 0)
		{
			OutputFiles::m_logFile << "Error : Fail to delete " << fileName.str() << std::endl;
			exit(1);
		}
	}
	//=================================================================
	// Synchronize
	//=================================================================
	MPI_Barrier(MPI_COMM_WORLD);

	calcForwardComputation_ABIC();
	double dataMisfitThisPE = ptrObservedData->calculateErrorSumOfSquaresThisPE();
	double dataMisfit(0.0);
	double modelroughness(0.0);
	double modelnorm(0.0);
	double obj(0.0);
	double abic(0.0);
	double abicd(0.0);
	double deter(0.0);
	double rmsc(0.0);
	double rmsu(0.0);
	std::vector<double> abicVec(2);
	MPI_Reduce(&dataMisfitThisPE, &dataMisfit, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (myProcessID == 0)
	{
		rmsc = sqrt(dataMisfit / static_cast<double>(numDataTotal));
		if (ptrAnalysisControl->useDifferenceFilter())
		{
			modelroughness = (ResistivityBlock::getInstance())->calcModelRoughnessForDifferenceFilter();
		}
		else
		{
			modelroughness = (ResistivityBlock::getInstance())->calcModelRoughnessForLaplacianFilter();
		}

		deter = getdeterminant();
		modelnorm = (ResistivityBlock::getInstance())->calModelNormLog10();
		obj = dataMisfit + pow(ptrAnalysisControl->getTradeOffParameterForResistivityValue(), 2.0) * modelroughness +  pow(ptrAnalysisControl->getTradeOffParameterForResistivityValue(), 2.0) * pow(ptrAnalysisControl->getTradeOffParameterForMinNorm(), 2.0) * modelnorm;
		rmsu = sqrt(obj / (static_cast<double>(numDataTotal) - 1));
		abicd = numDataTotal * std::log(2.0 * 3.1415926 * (obj / numDataTotal));
		abic = abicd + deter + numDataTotal;
		std::cout << " # Trade-off: " << ptrAnalysisControl->getTradeOffParameterForResistivityValue() << ";  step_size: " << ptrAnalysisControl->getStepLengthDampingFactorCur() << ";  obj_func: " << obj << ";  rms: " << rmsc << ";  rms_u: " << rmsu << ";  m_updated: " << mupdatedratio << ";  ABIC: " << abic << std::endl;
		abicVec[0] = abic;
		abicVec[1] = rmsc;

		OutputFiles::m_cnvFileitr.precision(6);
		OutputFiles::m_cnvFileitr << std::setw(15) << std::scientific << ptrAnalysisControl->geticut()
								  << std::setw(15) << std::scientific << ptrAnalysisControl->getTradeOffParameterForResistivityValue()
								  << std::setw(15) << std::scientific << obj
								  << std::setw(15) << std::scientific << dataMisfit
								  << std::setw(15) << std::scientific << modelroughness
								  << std::setw(15) << std::scientific << modelnorm
								  << std::setw(15) << std::scientific << rmsc
								  << std::setw(15) << std::scientific << rmsu
								  << std::setw(15) << std::scientific << mupdatedratio
								  << std::setw(15) << std::scientific << abic
								  << std::endl;
	}
	MPI_Bcast(abicVec.data(), abicVec.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	setabic(abicVec);
	setmratio(mupdatedratio);
}

// Read sensitivity matrix
void InversionGaussNewtonDataSpace_ABIC::readSensitivityMatrix(const std::string &fileName, int &numData, int &numModel, double *&sensitivityMatrix) const
{

	FILE *fp = fopen(fileName.c_str(), "rb");
	if (fp == nullptr)
	{
		OutputFiles::m_logFile << "File open error !! : " << fileName << std::endl;
		exit(1);
	}

	fread(&numData, sizeof(int), 1, fp);
	fread(&numModel, sizeof(int), 1, fp);

	const long long numComps = static_cast<long long>(numData) * static_cast<long long>(numModel);
	sensitivityMatrix = new double[numComps];
	fread(sensitivityMatrix, sizeof(double), numComps, fp);
	fclose(fp);
}

// Calculate constraining matrix of difference filter
void InversionGaussNewtonDataSpace_ABIC::calcConstrainingMatrixForDifferenceFilter(DoubleSparseMatrix &constrainingMatrix) const
{

	const ObservedData *const ptrObservedData = ObservedData::getInstance();
	const int numDistortionParamsNotFixed = ptrObservedData->getNumDistortionParamsNotFixed(); //default 0
	const int numResistivityBlockNotFixed = (ResistivityBlock::getInstance())->getNumResistivityBlockNotFixed();
	const int numModel = numResistivityBlockNotFixed + numDistortionParamsNotFixed;
	// std::cout << "numDistortionParamsNotFixed = " << numDistortionParamsNotFixed  << "   numResistivityBlockNotFixed = " << numResistivityBlockNotFixed << std::endl;

	std::vector<std::pair<int, int>> nonZeroCols;
	std::vector<double> matValues;
	std::vector<double> rhsValues;
	nonZeroCols.reserve(numModel);
	matValues.reserve(numModel);
	rhsValues.reserve(numModel);

	const AnalysisControl *const ptrAnalysisControl = AnalysisControl::getInstance();
	const double factor1 = ptrAnalysisControl->getTradeOffParameterForResistivityValue();
	(ResistivityBlock::getInstance())->calcRougheningMatrixDegeneratedForDifferenceFilter(factor1, nonZeroCols, matValues, rhsValues);

	const int nBlkNotFixed = (ResistivityBlock::getInstance())->getNumResistivityBlockNotFixed();
	int iMdl = nBlkNotFixed;
	if (ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
	{
		// For distortion matrix components
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForDistortionMatrixComplexity();
		for (int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed)
		{
			// constrainingMatrix->setStructureAndAddValueByTripletFormat( iMdl, iMdl, factor2 );
			nonZeroCols.push_back(std::make_pair(iMdl, -1));
			matValues.push_back(factor2);
			rhsValues.push_back(0.0);
			++iMdl;
		}
	}
	else if (ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
	{
		// For gains and rotations of distortion matrix
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForGainsOfDistortionMatrix();
		const double factor3 = ptrAnalysisControl->getTradeOffParameterForRotationsOfDistortionMatrix();
		for (int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed)
		{
			if (ptrObservedData->getTypesOfDistortionParamsNotFixed(iParamsNotFixed) == ObservedDataStationMT::EX_GAIN ||
				ptrObservedData->getTypesOfDistortionParamsNotFixed(iParamsNotFixed) == ObservedDataStationMT::EY_GAIN)
			{
				// Gains
				// constrainingMatrix->setStructureAndAddValueByTripletFormat( iMdl, iMdl, factor2 );
				nonZeroCols.push_back(std::make_pair(iMdl, -1));
				matValues.push_back(factor2);
				rhsValues.push_back(0.0);
			}
			else
			{
				// Rotations
				// constrainingMatrix->setStructureAndAddValueByTripletFormat( iMdl, iMdl, factor3 );
				nonZeroCols.push_back(std::make_pair(iMdl, -1));
				matValues.push_back(factor3);
				rhsValues.push_back(0.0);
			}
			++iMdl;
		}
	}
	else if (ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
	{
		// For gains of distortion matrix
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForGainsOfDistortionMatrix();
		for (int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed)
		{
			// constrainingMatrix->setStructureAndAddValueByTripletFormat( iMdl, iMdl, factor2 );
			nonZeroCols.push_back(std::make_pair(iMdl, -1));
			matValues.push_back(factor2);
			rhsValues.push_back(0.0);
			++iMdl;
		}
	}

	assert(nonZeroCols.size() == matValues.size());
	assert(nonZeroCols.size() == rhsValues.size());
	const int numRows = static_cast<int>(nonZeroCols.size());
	constrainingMatrix.setNumRowsAndColumns(numRows, numModel);

	for (int iRow = 0; iRow < numRows; ++iRow)
	{
		constrainingMatrix.setStructureAndAddValueByTripletFormat(iRow, nonZeroCols[iRow].first, matValues[iRow]);
		if (nonZeroCols[iRow].second >= 0)
		{
			constrainingMatrix.setStructureAndAddValueByTripletFormat(iRow, nonZeroCols[iRow].second, -matValues[iRow]);
		}
		constrainingMatrix.addRightHandSideVector(iRow, rhsValues[iRow]);
	}
#ifdef _DEBUG_WRITE
	dynamic_cast<RougheningMatrix *>(&constrainingMatrix)->outputRougheningMatrix("difference_filter.out");
#endif
	constrainingMatrix.convertToCRSFormat();
}

// Calculate forward computation
// No additional output compared to original Forward camputation
void InversionGaussNewtonDataSpace_ABIC::calcForwardComputation_ABIC()
{
	const AnalysisControl *const ptrAnalysisControl = AnalysisControl::getInstance();
	Forward3D *ptrForward3D = ptrAnalysisControl->getPointerOfForward3D();
	if (ptrForward3D == NULL)
	{
		OutputFiles::m_logFile << "Error : Pointer to the class Forward3D is NULL." << std::endl;
		exit(1);
	}

	ObservedData *const pObservedData = ObservedData::getInstance();
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
			OutputFiles::m_logFile << "# Start forward calculation. Frequency : " << frquencyValue << " [Hz], Polarization : " << polarizationName << std::endl;
			OutputFiles::m_logFile << "#================================================================================================" << std::endl;

			ptrForward3D->forwardCalculation(frquencyValue, iPol);

			pObservedData->calculateEMFieldOfAllStations(ptrForward3D, frquencyValue, iPol, ifreq);
		}

		OutputFiles::m_logFile << "#==============================================================================" << std::endl;
		OutputFiles::m_logFile << "# Calculate response functions. " << ptrAnalysisControl->outputElapsedTime() << std::endl;
		pObservedData->calculateResponseFunctionOfAllStations(ifreq);
		OutputFiles::m_logFile << "#==============================================================================" << std::endl;
	}

	OutputFiles::m_logFile << "# Release memory of coefficient matrix and sparse solver. " << ptrAnalysisControl->outputElapsedTime() << std::endl;
	// if (!m_holdMemoryForwardSolver) {// Release memory of sparse solver
	ptrForward3D->releaseMemoryOfMatrixAndSolver();
	//}
}