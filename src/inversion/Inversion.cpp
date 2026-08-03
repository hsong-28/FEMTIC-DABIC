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
#include "Inversion.h"
#include "ObservedData.h"
#include "AnalysisControl.h"
#include "OutputFiles.h"
#include "ResistivityBlock.h"
#include "RougheningMatrix.h"
#include "ComplexSparseMatrix.h"
#include "MeshDataBrickElement.h"
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <utility>
#include <vector>

#include "mkl_lapacke.h"
#include "mpi.h"

// Default constructer
Inversion::Inversion():
	m_numModel(0),
	m_numData(0),
	alphawithmaxc(0.0),
	determinant(0.0),
	m_sensitivityScalarValues(NULL)
{
	for( int i = 0; i < 2; ++i ){
		m_derivativesOfEMField[i] = NULL;
	}
}

// Constructor
Inversion::Inversion( const int nModel, const int nData ):
	m_numModel(nModel),
	m_numData(nData),
	m_sensitivityScalarValues(NULL)
{
	for( int i = 0; i < 2; ++i ){
		m_derivativesOfEMField[i] = NULL;
	}
}

// Destructor
Inversion::~Inversion(){

	for( int i = 0; i < 2; ++i ){
		if( m_derivativesOfEMField[i] != NULL ){
			delete [] m_derivativesOfEMField[i];
			m_derivativesOfEMField[i] = NULL;
		}
	}
	
	if( m_sensitivityScalarValues != NULL ){
		delete [] m_sensitivityScalarValues;
		m_sensitivityScalarValues = NULL;
	}

}

// Calculate derivatives of EM field
void Inversion::calculateDerivativesOfEMField( Forward3D* const ptrForward3D, const double freq, const int iPol ){

	ObservedData* const ptrObservedData = ObservedData::getInstance();
	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();
	const ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();

	const int numEquationFinallySolved = ptrForward3D->getNumOfEquationFinallySolved();;
	const int numInterpolatorVectors = ptrObservedData->getNumInterpolatorVectors();
	const int nBlkNotFixed = ptrResistivityBlock->getNumResistivityBlockNotFixed();

	//-------------------------------------------------------------------------
	// Calculate right-hand-side vectors consisting of interpolator vectors.
	//-------------------------------------------------------------------------
	OutputFiles::m_logFile << "# Calculate right-hand-side vectors consisting of interpolator vectors." << ptrAnalysisControl->outputElapsedTime() << std::endl;
	ptrObservedData->calculateInterpolatorVectors( ptrForward3D, freq );

	//--------------------------------------------------------------------------------
	// Solve phase for right-hand side vectors consisting of interpolator vectors ---
	//--------------------------------------------------------------------------------

	// Solution vectors of 3D forward computation for right-hand sides vectors consisting of interpolator vectors
	std::complex<double>* solutionForInterpolatorVectors = new std::complex<double>[static_cast<long long>(numEquationFinallySolved) * static_cast<long long>(numInterpolatorVectors)];

	OutputFiles::m_logFile << "# Solve phase of matrix solver for right-hand sides consisting of interpolator vectors." << ptrAnalysisControl->outputElapsedTime() << std::endl;
	ptrForward3D->solvePhaseForRhsConsistingInterpolatorVectors( numInterpolatorVectors, solutionForInterpolatorVectors );

	//--------------------------------------
	// Calculate derivatives of EM field ---
	//--------------------------------------
	if( m_derivativesOfEMField[iPol] != NULL ){
		delete [] m_derivativesOfEMField[iPol];
		m_derivativesOfEMField[iPol] = NULL;
	}
	if( numInterpolatorVectors <= 0 ){
		OutputFiles::m_logFile << "Error : Number of interpolator vectors is less than or equal to zero. numInterpolatorVectors = " << numInterpolatorVectors << std::endl;
		exit(1);
	}
	if( nBlkNotFixed <= 0 ){
		OutputFiles::m_logFile << "Error : Number of resistivity blocks whose values are not fixed is less than or equal to zero. nBlkNotFixed = " << nBlkNotFixed << std::endl;
		exit(1);
	}

	const long long numComps = static_cast<long long>(numInterpolatorVectors) * static_cast<long long>(nBlkNotFixed);
	m_derivativesOfEMField[iPol] = new std::complex<double>[numComps];
	for( long long i = 0; i < numComps; ++i ){
		m_derivativesOfEMField[iPol][i] = std::complex<double>(0.0, 0.0);// Initialize
	}

	OutputFiles::m_logFile << "# Calculate derivatives of EM field." << ptrAnalysisControl->outputElapsedTime() << std::endl;
	ptrForward3D->calculateDerivativesOfEMField( numInterpolatorVectors, solutionForInterpolatorVectors, m_derivativesOfEMField[iPol] );

	delete [] solutionForInterpolatorVectors;
	
}

// Calculate sensitivity matrix
void Inversion::calculateSensitivityMatrix( const int freqIDAmongThisPE, const double freq ){

	const ObservedData* const ptrObservedData = ObservedData::getInstance();
	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();
	const ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();

	const int numDataThisPE = ptrObservedData->getNumObservedDataThisPE(freqIDAmongThisPE);
	const int freqIDGlobal = ptrObservedData->getIDsOfFrequenciesCalculatedByThisPE(freqIDAmongThisPE);
	//const int numModel = ptrResistivityBlock->getNumResistivityBlockNotFixed();
	const int numModel = getNumberOfModel();
	const double sensitivityMatrixStartSec = ptrAnalysisControl->getElapsedTimeInSeconds();

	//-------------------------------------------
	// Allocate memory for sensitivity matrix ---
	//-------------------------------------------
	if( numDataThisPE <= 0 ){
		OutputFiles::m_logFile << "Error : Number of data of this PE is less than or equal to zero. numDataThisPE = " << numDataThisPE << std::endl;
		exit(1);
	}
	if( numModel <= 0 ){
		OutputFiles::m_logFile << "Error : Number of model is less than or equal to zero. numModel = " << numModel << std::endl;
		exit(1);
	}
	const long long numComps = static_cast<long long>(numDataThisPE) * static_cast<long long>(numModel);
	double* sensitivityMatrix = new double[numComps];
	for( long long i = 0; i < numComps; ++i ){
		sensitivityMatrix[i] = 0.0;// Initialize
	}

	OutputFiles::m_logFile << "# Calculate sensitivity matrix of frequency " << freq << "[Hz]." << ptrAnalysisControl->outputElapsedTime() << std::endl;
	ptrObservedData->calculateSensitivityMatrix( m_derivativesOfEMField[0], m_derivativesOfEMField[1], freq, numModel, sensitivityMatrix );
	
	// Release memory of derivatives of EM field
	for( int i = 0; i < 2; ++i ){
		if( m_derivativesOfEMField[i] != NULL ){
			delete[] m_derivativesOfEMField[i];
			m_derivativesOfEMField[i] = NULL;
		}
	}

#ifdef _DEBUG_WRITE
	std::cout << "sensitivity matrix. freq = " << freq << std::endl;
	for( int idat = 0; idat < numDataThisPE; ++idat ){
		for( int imdl = 0; imdl < numModel; ++imdl ){
			std::cout << "idat, imdl, val " << idat << " " <<  imdl << " " << sensitivityMatrix[imdl+idat*numModel] << std::endl;
		}
	}
#endif

	//-------------------------------------------------
	// Write sensitivity matrix to out-of-core file ---
	//-------------------------------------------------
	std::ostringstream fileName;
	fileName << "sensMatFreq" << freqIDGlobal;
	//std::ofstream outputFile(fileName.str().c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
	//if( outputFile.fail() ){
	//	OutputFiles::m_logFile << "File open error !! : " << fileName.str() << std::endl;
	//	exit(1);
	//}

	OutputFiles::m_logFile << "# Sensitivity matrix of frequency " << freq << "[Hz] is saved in the out-of-core file " << fileName.str() << "." << std::endl;
	FILE* fp = NULL;
	fp = fopen( fileName.str().c_str(), "wb" );
	if( fp == NULL ){
		OutputFiles::m_logFile << "File open error !! : " << fileName.str() << std::endl;
		exit(1);
	}

	fwrite( &numDataThisPE, sizeof(int), 1, fp );
	fwrite( &numModel, sizeof(int), 1, fp );
	fwrite( sensitivityMatrix, sizeof(double), static_cast<long long>(numDataThisPE) * static_cast<long long>(numModel), fp );
	fclose( fp );
	OutputFiles::m_logFile << "# Timing sensitivity-matrix write stage. Frequency : " << freq
						   << " [Hz], elapsed : "
						   << ptrAnalysisControl->getElapsedTimeInSeconds() - sensitivityMatrixStartSec
						   << " sec." << std::endl;

	//---------------------------------------------
	//--- Calculate scaled sensitivity values ---
	//---------------------------------------------
	if( ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY) ){// if output sensitivity

		const int nBlkTotal = ptrResistivityBlock->getNumResistivityBlockTotal();
		for( int iblk = 0; iblk < nBlkTotal; ++iblk ){
			if( ptrResistivityBlock->isFixedResistivityValue( iblk ) ){
				continue;
			}
			const int imdl = ptrResistivityBlock->getModelIDFromBlockID(iblk);
			for( int idat = 0; idat < numDataThisPE; ++idat ){
				const long long index = static_cast<long long>(imdl) + static_cast<long long>(idat) * static_cast<long long>(numModel);
				m_sensitivityScalarValues[imdl] += std::abs( sensitivityMatrix[index] ); //0525
			}
		}

		//MPI_Allreduce( sensitivityScalarValuesThisPE, m_sensitivityScalarValues, numModel, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

	}

	delete [] sensitivityMatrix;

#ifdef _DEBUG_WRITE
	std::cout << "read binary file and write agein." << std::endl;
	fp = fopen( fileName.str().c_str(), "rb" );
	if( fp == NULL ){
		OutputFiles::m_logFile << "File open error !! : " << fileName.str() << std::endl;
		exit(1);
	}

	int ndata(0);
	int nmodel(0);
	fread( &ndata, sizeof(int), 1, fp );
	fread( &nmodel, sizeof(int), 1, fp );

	double* temp = new double[ numDataThisPE * numModel ];
	fread( temp, sizeof(double), ndata * nmodel, fp );

	for( int idat = 0; idat < ndata; ++idat ){
		for( int imdl = 0; imdl < nmodel; ++imdl ){
			std::cout << "idat, imdl, val " << idat << " " <<  imdl << " " << temp[imdl+idat*numModel] << std::endl;
		}
	}

	fclose( fp );
#endif

}

// Allocate memory for  sensitivity values
void Inversion::allocateMemoryForSensitivityScalarValues(){

	//const int numModel = ( ResistivityBlock::getInstance() )->getNumResistivityBlockNotFixed();
	const int numModel = getNumberOfModel();

	if( m_sensitivityScalarValues != NULL ){
		delete [] m_sensitivityScalarValues;
		m_sensitivityScalarValues = NULL;
	}

	m_sensitivityScalarValues = new double[ numModel ];

	for( int i = 0; i < numModel; ++i ){
		m_sensitivityScalarValues[i] = 0.0;
	}
	
}

// Release memory of sensitivity values
void Inversion::releaseMemoryOfSensitivityScalarValues(){

	if( m_sensitivityScalarValues != NULL ){
		delete [] m_sensitivityScalarValues;
		m_sensitivityScalarValues = NULL;
	}

}

// Output scalar sensitivity values to vtk file
void Inversion::outputSensitivityScalarValuesToVtk(const int iterNum) const{
	
	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();

	if( !ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY ) &&
		!ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY_DENSITY ) ){
		return;
	}

	const double criteria = 1.0e-20;
	const double criteria1 = 1.0e-4;
	const double criteria2 = 1.0e+1;
	const int numModel = getNumberOfModel();
	double* sensitivityScalarValuesForOutput = new double[ numModel ];

#ifdef _DEBUG_WRITE
	// Get process ID and total process number
	//int myProcessID(0);
	//MPI_Comm_rank ( MPI_COMM_WORLD, &myProcessID );
	const int myProcessID = ptrAnalysisControl->getMyPE();
	for( int i = 0; i < numModel; ++i ){
		std::cout << "PE i m_sensitivityScalarValues : " << myProcessID << " " << i << " " << m_sensitivityScalarValues[i] << std::endl;
	}
#endif

	MPI_Allreduce( m_sensitivityScalarValues, sensitivityScalarValuesForOutput, numModel, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

#ifdef _DEBUG_WRITE
	for( int i = 0; i < numModel; ++i ){
		std::cout << "PE i sensitivityScalarValuesForOutput : " << myProcessID << " " << i << " " << sensitivityScalarValuesForOutput[i] << std::endl;
	}
#endif

	const MeshData* const ptrMeshData =ptrAnalysisControl->getPointerOfMeshData();
	const int nElem = ptrMeshData->getNumElemTotal();
	const ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();

	if( ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY ) ){// Output sensitivity
		OutputFiles::m_vtkFile << "SCALARS Sensitivity float" <<  std::endl;
		OutputFiles::m_vtkFile << "LOOKUP_TABLE default" <<  std::endl;

		double maxSensitivity(-1.0);
		for (int iElem = 0; iElem < nElem; ++iElem) {
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID(iElem);

			if (ptrResistivityBlock->isFixedResistivityValue(iblk)) {
			}
			else {
				const double a = (ptrResistivityBlock->calcVolumeOfBlock(iblk)) * 1.0e-9; //��mΪ��λ��ΪkmΪ��λ
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID(iblk);
				double sensitivity = fabs(sensitivityScalarValuesForOutput[imdl]);
				if (sensitivity > maxSensitivity) {
					maxSensitivity = sensitivity;
				}
			}
		}
		const double divMax = 1.0 / maxSensitivity;
		OutputFiles::m_logFile << "maxSensitivity : " << maxSensitivity << std::endl;



		std::ostringstream fileSensitivity;
		//fileName << "resistivity_model" << iterNum << ".dat";
		//fileName << "resistivity_model_iter" << iterNum << ".dat";
		fileSensitivity << "resistivity_block_iter" << 50+iterNum << ".dat";

		FILE* fp;
		if ((fp = fopen(fileSensitivity.str().c_str(), "w")) == NULL) {
			OutputFiles::m_logFile << "File open error !! : " << fileSensitivity.str() << std::endl;
			exit(1);
		}

		//const int numElemTotal = ptrMeshDataBrickElement->getNumElemTotal();
		const int numElemTotal = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getNumElemTotal();
		const int numBlkTotal = ptrResistivityBlock->getNumResistivityBlockTotal();
		fprintf(fp, "%10d%10d\n", numElemTotal, numBlkTotal);

		for (int iElem = 0; iElem < numElemTotal; ++iElem) {
			const int iBlk = ptrResistivityBlock->getBlockIDFromElemID(iElem);
			fprintf(fp, "%10d%10d\n", iElem, iBlk);
		}

		for (int iBlk = 0; iBlk < numBlkTotal; ++iBlk) {
			

			if (ptrResistivityBlock->isFixedResistivityValue(iBlk)) {
				OutputFiles::m_vtkFile << criteria << std::endl;
				fprintf(fp, "%10d%5s%15e%15e%15e%15e%10d\n", iBlk, "     ",
					criteria, 1.0e-20, 1.0e+20, 1.0, 1);
			}
			else {

				const double volume = (ptrResistivityBlock->calcVolumeOfBlock(iBlk)) * 1.0e-9; //��mΪ��λ��ΪkmΪ��λ
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID(iBlk);
				//OutputFiles::m_logFile << "ModelID:  " << imdl << ";  " << "BlockID:  " << iblk << std::endl;
				//double sensitivity = fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) );
				double sensitivity = fabs(sensitivityScalarValuesForOutput[imdl])/ volume;
				const int node0ID = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getNodesOfElements(iBlk + 5886, 0);
				float x0 = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getXCoordinatesOfNodes(node0ID);
				float y0 = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getYCoordinatesOfNodes(node0ID);
				float z0 = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getZCoordinatesOfNodes(node0ID);
				const int node6ID = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getNodesOfElements(iBlk + 5886, 6);
				float x1 = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getXCoordinatesOfNodes(node6ID);
				float y1 = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getYCoordinatesOfNodes(node6ID);
				float z1 = ((AnalysisControl::getInstance())->getPointerOfMeshData())->getZCoordinatesOfNodes(node6ID);
				if (fabs(x0) > 60.0 * 1000.0 || fabs(y0) > 60.0 * 1000.0 || fabs(z0) > 60.0 * 1000.0 || fabs(x1) > 60.0 * 1000.0 || fabs(y1) > 60.0 * 1000.0 || fabs(z1) > 60.0 * 1000.0) {
					sensitivity = 1.0;
				}

				/*if (sensitivity < criteria1) {
					sensitivity = criteria1;
				}
				if (sensitivity > criteria2) {
					sensitivity = criteria2;
				}*/
				//sensitivity = sensitivity * 100.0;
				fprintf(fp, "%10d%5s%15e%15e%15e%15e%10d\n", iBlk, "     ",
					sensitivity, 1.0e-20, 1.0e+20, 1.0, 0);
			}
		}

		fclose(fp);

		//double maxSensitivity(-1.0);
		for( int iElem = 0 ; iElem < nElem; ++iElem ){
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID( iElem );

			if( ptrResistivityBlock->isFixedResistivityValue( iblk ) ){
				OutputFiles::m_vtkFile << criteria << std::endl;
			}else{
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID( iblk );
				//OutputFiles::m_logFile << "ModelID:  " << imdl << ";  " << "BlockID:  " << iblk << std::endl;
				//double sensitivity = fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) );
				double sensitivity = fabs( sensitivityScalarValuesForOutput[imdl] );
				if( sensitivity < criteria ){
					sensitivity = criteria;
				}
				OutputFiles::m_vtkFile << sensitivity << std::endl;
				//if( sensitivity > maxSensitivity ){
				//	maxSensitivity = sensitivity;
				//}
			}
		}

		//divMax = 1.0 / maxSensitivity;

		OutputFiles::m_vtkFile << "SCALARS NormalizedSensitivity float" <<  std::endl;
		OutputFiles::m_vtkFile << "LOOKUP_TABLE default" <<  std::endl;

		for( int iElem = 0 ; iElem < nElem; ++iElem ){
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID( iElem );

			if( ptrResistivityBlock->isFixedResistivityValue( iblk ) ){
				OutputFiles::m_vtkFile << criteria << std::endl;
			}else{
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID( iblk );
				//double sensitivityNormalized = divMax * fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) );
				double sensitivityNormalized = divMax * fabs( sensitivityScalarValuesForOutput[imdl] );
				if( sensitivityNormalized < criteria ){
					sensitivityNormalized = criteria;
				}
				OutputFiles::m_vtkFile << sensitivityNormalized << std::endl;
			}
		}
	}

	if( ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY_DENSITY ) ){// Output sensitivity density
		OutputFiles::m_vtkFile << "SCALARS SensitivityDensity float" <<  std::endl;
		OutputFiles::m_vtkFile << "LOOKUP_TABLE default" <<  std::endl;

		double maxSensitivityDensity(-1.0);
		for( int iElem = 0 ; iElem < nElem; ++iElem ){
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID( iElem );

			if( ptrResistivityBlock->isFixedResistivityValue( iblk ) ){
				OutputFiles::m_vtkFile << criteria << std::endl;
			}else{
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID( iblk );
				const double volume = ptrResistivityBlock->calcVolumeOfBlock( iblk );
				//double sensitivityDensity = fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) ) / volume;
				double sensitivityDensity = fabs( sensitivityScalarValuesForOutput[imdl] ) / volume;
				if( sensitivityDensity < criteria ){
					sensitivityDensity = criteria;
				}
				OutputFiles::m_vtkFile << sensitivityDensity << std::endl;
				if( sensitivityDensity > maxSensitivityDensity ){
					maxSensitivityDensity = sensitivityDensity;
				}
			}
		}

		const double divMax = 1.0 / maxSensitivityDensity;

		OutputFiles::m_vtkFile << "SCALARS NormalizedSensitivityDensity float" <<  std::endl;
		OutputFiles::m_vtkFile << "LOOKUP_TABLE default" <<  std::endl;

		for( int iElem = 0 ; iElem < nElem; ++iElem ){
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID( iElem );

			if( ptrResistivityBlock->isFixedResistivityValue( iblk ) ){
				OutputFiles::m_vtkFile << criteria << std::endl;
			}else{
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID( iblk );
				const double volume = ptrResistivityBlock->calcVolumeOfBlock( iblk );
				//double sensitivityDensityNormalized = divMax * fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) ) / volume;
				double sensitivityDensityNormalized = divMax * fabs( sensitivityScalarValuesForOutput[imdl] ) / volume;
				if( sensitivityDensityNormalized < criteria ){
					sensitivityDensityNormalized = criteria;
				}
				OutputFiles::m_vtkFile << sensitivityDensityNormalized << std::endl;
			}
		}
	}

	delete [] sensitivityScalarValuesForOutput;
}

// Output scalar sensitivity values to binary file
void Inversion::outputSensitivityScalarValuesToBinary( const int interNum ) const{

	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();

	if( !ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY ) &&
		!ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY_DENSITY ) ){
		return;
	}

	const double criteria = 1.0e-20;
	const int numModel = getNumberOfModel();
	double* sensitivityScalarValuesForOutput = new double[ numModel ];

	MPI_Allreduce( m_sensitivityScalarValues, sensitivityScalarValuesForOutput, numModel, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

	if( ptrAnalysisControl->getMyPE() != 0 ){// Only PE 0 output sensitivity
		delete [] sensitivityScalarValuesForOutput;
		return;
	}

	const MeshData* const ptrMeshData =ptrAnalysisControl->getPointerOfMeshData();
	const ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();
	const int nElem = ptrMeshData->getNumElemTotal();

	if( ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY ) ){// Output sensitivity
		std::ostringstream oss;
		oss << "Sensitivity.iter" << interNum;
		std::ofstream fout;

		// Output sensitivity
		fout.open( oss.str().c_str(), std::ios::out | std::ios::binary | std::ios::trunc );

		char line[80];
		std::ostringstream ossTitle;
		ossTitle << "Sensitivity";
		strcpy( line, ossTitle.str().c_str() );
		fout.write( line, 80 );

		strcpy( line, "part" );
		fout.write( line, 80 );

		int ibuf(1);
		fout.write( (char*) &ibuf, sizeof( int ) );

		if( ptrAnalysisControl->getTypeOfMesh() ==  MeshData::TETRA ){
			strcpy( line, "tetra4" );
		}
		else{
			strcpy( line, "hexa8" );
		}
		fout.write( line, 80 );

		double maxSensitivity(-1.0);
		for( int iElem = 0 ; iElem < nElem; ++iElem ){
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID( iElem );

			float dbuf(criteria);
			if( !ptrResistivityBlock->isFixedResistivityValue( iblk ) ){// Modifiable resistivity
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID( iblk );
				//dbuf = static_cast<float>( fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) ) );
				dbuf = static_cast<float>( fabs( sensitivityScalarValuesForOutput[imdl] ) );
				if( dbuf < criteria ){
					dbuf = criteria;
				}
				if( dbuf > maxSensitivity ){
					maxSensitivity = dbuf;
				}
			}
			fout.write( (char*) &dbuf, sizeof( float ) );
		}
		fout.close();

		const double divMax = 1.0 / maxSensitivity;

		// Output normalized sensitivity
		const std::string outputFileName = "Normalized"+oss.str();
		fout.open( outputFileName.c_str(), std::ios::out | std::ios::binary | std::ios::trunc );

		ossTitle.clear();
		ossTitle << "NormalizedSensitivity";
		strcpy( line, ossTitle.str().c_str() );
		fout.write( line, 80 );

		strcpy( line, "part" );
		fout.write( line, 80 );

		fout.write( (char*) &ibuf, sizeof( int ) );

		if( ptrAnalysisControl->getTypeOfMesh() ==  MeshData::TETRA ){
			strcpy( line, "tetra4" );
		}
		else{
			strcpy( line, "hexa8" );
		}
		fout.write( line, 80 );

		for( int iElem = 0 ; iElem < nElem; ++iElem ){
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID( iElem );

			float dbuf(criteria);
			if( !ptrResistivityBlock->isFixedResistivityValue( iblk ) ){// Modifiable resistivity
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID( iblk );
				//dbuf = static_cast<float>( divMax * fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) ) );
				dbuf = static_cast<float>( divMax * fabs( sensitivityScalarValuesForOutput[imdl] ) );
				if( dbuf < criteria ){
					dbuf = criteria;
				}
			}
			fout.write( (char*) &dbuf, sizeof( float ) );
		}

		fout.close();
	}

	if( ptrAnalysisControl->doesOutputToVTK( AnalysisControl::OUTPUT_SENSITIVITY_DENSITY ) ){// Output sensitivity density
		std::ostringstream oss;
		oss << "SensitivityDensity.iter" << interNum;
		std::ofstream fout;

		// Output sensitivity
		fout.open( oss.str().c_str(), std::ios::out | std::ios::binary | std::ios::trunc );

		char line[80];
		std::ostringstream ossTitle;
		ossTitle << "SensitivityDensity";
		strcpy( line, ossTitle.str().c_str() );
		fout.write( line, 80 );

		strcpy( line, "part" );
		fout.write( line, 80 );

		int ibuf(1);
		fout.write( (char*) &ibuf, sizeof( int ) );

		if( ptrAnalysisControl->getTypeOfMesh() ==  MeshData::TETRA ){
			strcpy( line, "tetra4" );
		}
		else{
			strcpy( line, "hexa8" );
		}
		fout.write( line, 80 );

		double maxSensitivityDensity(-1.0);
		for( int iElem = 0 ; iElem < nElem; ++iElem ){
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID( iElem );

			float dbuf(criteria);
			if( !ptrResistivityBlock->isFixedResistivityValue( iblk ) ){// Modifiable resistivity
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID( iblk );
				const double volume = ptrResistivityBlock->calcVolumeOfBlock( iblk );
				//dbuf = static_cast<float>( fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) ) / volume );
				dbuf = static_cast<float>( fabs( sensitivityScalarValuesForOutput[imdl] ) / volume );
				if( dbuf < criteria ){
					dbuf = criteria;
				}
				if( dbuf > maxSensitivityDensity ){
					maxSensitivityDensity = dbuf;
				}
			}
			fout.write( (char*) &dbuf, sizeof( float ) );
		}

		fout.close();

		const double divMax = 1.0 / maxSensitivityDensity;

		// Output normalized sensitivity
		const std::string outputFileName = "Normalized"+oss.str();
		fout.open( outputFileName.c_str(), std::ios::out | std::ios::binary | std::ios::trunc );

		ossTitle.clear();
		ossTitle << "NormalizedSensitivityDensity";
		strcpy( line, ossTitle.str().c_str() );
		fout.write( line, 80 );

		strcpy( line, "part" );
		fout.write( line, 80 );

		fout.write( (char*) &ibuf, sizeof( int ) );

		if( ptrAnalysisControl->getTypeOfMesh() ==  MeshData::TETRA ){
			strcpy( line, "tetra4" );
		}
		else{
			strcpy( line, "hexa8" );
		}
		fout.write( line, 80 );

		for( int iElem = 0 ; iElem < nElem; ++iElem ){
			const int iblk = ptrResistivityBlock->getBlockIDFromElemID( iElem );

			float dbuf(criteria);
			if( !ptrResistivityBlock->isFixedResistivityValue( iblk ) ){// Modifiable resistivity
				const int imdl = ptrResistivityBlock->getModelIDFromBlockID( iblk );
				const double volume = ptrResistivityBlock->calcVolumeOfBlock( iblk );
				//dbuf = static_cast<float>( divMax * fabs( sensitivityScalarValuesForOutput[imdl] * log10( ptrResistivityBlock->getResistivityValuesFromBlockID(iblk) ) ) / volume );
				dbuf = static_cast<float>( divMax * fabs( sensitivityScalarValuesForOutput[imdl] ) / volume );
				if( dbuf < criteria ){
					dbuf = criteria;
				}
			}
			fout.write( (char*) &dbuf, sizeof( float ) );
		}

		fout.close();
	}

	delete [] sensitivityScalarValuesForOutput;
}



void Inversion::setdeterminant(double value) {
	determinant = value;
}

double Inversion::getdeterminant() const {
	return determinant;
}

void Inversion::setdeterminantRTR(double value) {
	determinantRTR = value;
}

double Inversion::getdeterminantRTR() const {
	return determinantRTR;
}

void Inversion::setAlphawithmaxc(double value) {
	alphawithmaxc = value;
}

double Inversion::getAlphawithmaxc() const {
	return alphawithmaxc;
}

void Inversion::setrms(double value) {
	rms = value;
}

double Inversion::getrms() const {
	return rms;
}

double Inversion::getminABIC() const {
	return ABICmin;
}

void Inversion::setabic(const std::vector<double>& values) {
	abicVec = values;
}

const std::vector<double>& Inversion::getabic() const {
	return abicVec;
}

void Inversion::setmratio(double value) {
	mratio = value;
}

double Inversion::getmratio() const {
	return mratio;
}



// Calculate constraining matrix
void Inversion::calcConstrainingMatrix( DoubleSparseMatrix& constrainingMatrix ) const{

	const ObservedData* const ptrObservedData = ObservedData::getInstance();
	const int numDistortionParamsNotFixed = ptrObservedData->getNumDistortionParamsNotFixed();
	const int numResistivityBlockNotFixed = ( ResistivityBlock::getInstance() )->getNumResistivityBlockNotFixed();
	const int numModel = numResistivityBlockNotFixed + numDistortionParamsNotFixed;
	constrainingMatrix.setNumRowsAndColumns( numModel, numModel );

	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();
	const double factor1 = ptrAnalysisControl->getTradeOffParameterForResistivityValue();
	( ResistivityBlock::getInstance() )->calcRougheningMatrixDegeneratedForLaplacianFilter( constrainingMatrix, factor1 );

	const int nBlkNotFixed = ( ResistivityBlock::getInstance() )->getNumResistivityBlockNotFixed();
	int iMdl = nBlkNotFixed;

	if( ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE ){
		// For distortion matrix components
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForDistortionMatrixComplexity();
		for( int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed ){
			constrainingMatrix.setStructureAndAddValueByTripletFormat( iMdl, iMdl, factor2 ); 
			++iMdl;
		}
	}
	else if( ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS ){
		// For gains and rotations of distortion matrix
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForGainsOfDistortionMatrix();
		const double factor3 = ptrAnalysisControl->getTradeOffParameterForRotationsOfDistortionMatrix();
		for( int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed ){
			if( ptrObservedData->getTypesOfDistortionParamsNotFixed(iParamsNotFixed) == ObservedDataStationMT::EX_GAIN ||
				ptrObservedData->getTypesOfDistortionParamsNotFixed(iParamsNotFixed) == ObservedDataStationMT::EY_GAIN ){
				// Gains
				constrainingMatrix.setStructureAndAddValueByTripletFormat( iMdl, iMdl, factor2 );
			}else{
				// Rotations
				constrainingMatrix.setStructureAndAddValueByTripletFormat( iMdl, iMdl, factor3 );
			}
			++iMdl;
		}
	}
	else if( ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY ){
		// For gains of distortion matrix
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForGainsOfDistortionMatrix();
		for( int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed ){
			constrainingMatrix.setStructureAndAddValueByTripletFormat( iMdl, iMdl, factor2 );
			++iMdl;
		}
	}

	constrainingMatrix.convertToCRSFormat();

}

void Inversion::calcRoughnessMatrix(DoubleSparseMatrix& constrainingMatrix) const
{
	const int numResistivityBlockNotFixed = (ResistivityBlock::getInstance())->getNumResistivityBlockNotFixed();
	constrainingMatrix.setNumRowsAndColumns(numResistivityBlockNotFixed, numResistivityBlockNotFixed);

	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();
	const double factor = ptrAnalysisControl->getTradeOffParameterForResistivityValue();
	(ResistivityBlock::getInstance())->calcRougheningMatrixDegeneratedForLaplacianFilter(constrainingMatrix, factor);
	constrainingMatrix.convertToCRSFormat();
}

void Inversion::calcRoughnessMatrix(DoubleSparseMatrix& constrainingMatrix, const int ito) const
{
	const int numResistivityBlockNotFixed = (ResistivityBlock::getInstance())->getNumResistivityBlockNotFixed();
	constrainingMatrix.setNumRowsAndColumns(numResistivityBlockNotFixed, numResistivityBlockNotFixed);

	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();
	const double factor = ptrAnalysisControl->get_ithTradeOffParameterForResistivityValue(ito);
	(ResistivityBlock::getInstance())->calcRougheningMatrixDegeneratedForLaplacianFilter(constrainingMatrix, factor);
	constrainingMatrix.convertToCRSFormat();
}

void Inversion::calcRoughnessMatrix_OCCAM(DoubleSparseMatrix& constrainingMatrix) const
{
	calcConstrainingMatrix(constrainingMatrix);
}

void Inversion::calcConstrainingMatrixForDifferenceFilterShared(DoubleSparseMatrix& constrainingMatrix) const
{
	const ObservedData* const ptrObservedData = ObservedData::getInstance();
	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();
	const int numDistortionParamsNotFixed = ptrObservedData->getNumDistortionParamsNotFixed();
	const int numResistivityBlockNotFixed = (ResistivityBlock::getInstance())->getNumResistivityBlockNotFixed();
	const bool runCrossGradient = ptrAnalysisControl->runCG();
	const int numModel = numResistivityBlockNotFixed + numDistortionParamsNotFixed;

	std::vector<std::pair<int, int> > nonZeroCols;
	std::vector<double> matValues;
	std::vector<double> rhsValues;
	nonZeroCols.reserve(numModel);
	matValues.reserve(numModel);
	rhsValues.reserve(numModel);

	const double factor1 = ptrAnalysisControl->getTradeOffParameterForResistivityValue();
	(ResistivityBlock::getInstance())->calcRougheningMatrixDegeneratedForDifferenceFilter(
		factor1,
		nonZeroCols,
		matValues,
		rhsValues);

	const int nBlkNotFixed = (ResistivityBlock::getInstance())->getNumResistivityBlockNotFixed();
	int iMdl = nBlkNotFixed;
	if (ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE) {
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForDistortionMatrixComplexity();
		for (int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed) {
			nonZeroCols.push_back(std::make_pair(iMdl, -1));
			matValues.push_back(factor2);
			rhsValues.push_back(0.0);
			++iMdl;
		}
	} else if (ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS) {
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForGainsOfDistortionMatrix();
		const double factor3 = ptrAnalysisControl->getTradeOffParameterForRotationsOfDistortionMatrix();
		for (int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed) {
			if (ptrObservedData->getTypesOfDistortionParamsNotFixed(iParamsNotFixed) == ObservedDataStationMT::EX_GAIN ||
				ptrObservedData->getTypesOfDistortionParamsNotFixed(iParamsNotFixed) == ObservedDataStationMT::EY_GAIN) {
				nonZeroCols.push_back(std::make_pair(iMdl, -1));
				matValues.push_back(factor2);
			} else {
				nonZeroCols.push_back(std::make_pair(iMdl, -1));
				matValues.push_back(factor3);
			}
			rhsValues.push_back(0.0);
			++iMdl;
		}
	} else if (ptrAnalysisControl->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY) {
		const double factor2 = ptrAnalysisControl->getTradeOffParameterForGainsOfDistortionMatrix();
		for (int iParamsNotFixed = 0; iParamsNotFixed < numDistortionParamsNotFixed; ++iParamsNotFixed) {
			nonZeroCols.push_back(std::make_pair(iMdl, -1));
			matValues.push_back(factor2);
			rhsValues.push_back(0.0);
			++iMdl;
		}
	}

	assert(nonZeroCols.size() == matValues.size());
	assert(nonZeroCols.size() == rhsValues.size());
	const int numRows = static_cast<int>(nonZeroCols.size());
	if (runCrossGradient) {
		constrainingMatrix.setNumRowsAndColumns(numRows + nBlkNotFixed, numModel);
	} else {
		constrainingMatrix.setNumRowsAndColumns(numRows, numModel);
	}

	for (int iRow = 0; iRow < numRows; ++iRow) {
		constrainingMatrix.setStructureAndAddValueByTripletFormat(iRow, nonZeroCols[iRow].first, matValues[iRow]);
		if (nonZeroCols[iRow].second >= 0) {
			constrainingMatrix.setStructureAndAddValueByTripletFormat(iRow, nonZeroCols[iRow].second, -matValues[iRow]);
		}
		constrainingMatrix.addRightHandSideVector(iRow, rhsValues[iRow]);
	}

	if (runCrossGradient) {
		const double factor3 = ptrAnalysisControl->getTradeOffParameterForCrossGradient();
		(ResistivityBlock::getInstance())->calcCrossGradientMatrixDegenerated(constrainingMatrix, factor3, iMdl);
	}
	constrainingMatrix.convertToCRSFormat();
}

void Inversion::buildProductionAppraisalRougheningState(
	RougheningMatrix& constrainingMatrix,
	DoubleSparseSquareSymmetricMatrix& rtrMatrix) const
{
	const AnalysisControl* const ptrAnalysisControl = AnalysisControl::getInstance();
	if (ptrAnalysisControl->useDifferenceFilter()) {
		calcConstrainingMatrixForDifferenceFilterShared(constrainingMatrix);
	} else {
		calcConstrainingMatrix(constrainingMatrix);
	}

	const ResistivityBlock* const ptrResistivityBlock = ResistivityBlock::getInstance();
	if (ptrResistivityBlock->getFlagAddSmallValueToDiagonals()) {
		constrainingMatrix.makeRTRMatrix(rtrMatrix, ptrResistivityBlock->getSmallValueAddedToDiagonals());
	} else {
		constrainingMatrix.makeRTRMatrix(rtrMatrix);
	}
}


// Copy model transforming jacobian matrix
void Inversion::copyModelTransformingJacobian( const int numBlockNotFixed, const int numModel, double* jacobian ) const{

	( ResistivityBlock::getInstance() )->copyDerivativeLog10ResistivityWithRespectToX( jacobian );
	for( int iMdl = numBlockNotFixed; iMdl < numModel; ++iMdl ){
		jacobian[iMdl] = 1.0;
	}

}

// Multiply model transforming jacobian matrix
void Inversion::multiplyModelTransformingJacobian( const int numData, const int numModel, const double* jacobian, double* matrix ) const{

	int iDat(0);
	int offset(0);
	int iMdl(0);
#ifdef _USE_OMP
	#pragma omp parallel for default(shared) private( iDat, offset, iMdl )
#endif
	for( iDat = 0; iDat < numData; ++iDat ){
		offset = iDat * numModel;
		for( iMdl = 0; iMdl < numModel; ++iMdl ){
			matrix[ iMdl + offset ] *= jacobian[iMdl];
		}
	}

}

// Delete out-of-core file all
void Inversion::deleteOutOfCoreFileAll(){

	const ObservedData* const ptrObservedData = ObservedData::getInstance();
	const int nFreq = ptrObservedData->getNumOfFrequenciesCalculatedByThisPE();
	for( int iFreq = 0; iFreq < nFreq; ++iFreq ){
		const int freqID = ptrObservedData->getIDsOfFrequenciesCalculatedByThisPE(iFreq);
		std::ostringstream fileName;
		fileName << "sensMatFreq" << freqID;
		FILE* fp = fopen( fileName.str().c_str(), "rb" );
		if( fp != NULL ){// File exists
			fclose( fp );
			if( remove( fileName.str().c_str() ) != 0 ){
				OutputFiles::m_logFile << "Error : Fail to delete " << fileName.str() << std::endl;
				exit(1);
			}
		}
	}

}

// Get number of model
int Inversion::getNumberOfModel() const{

	int numModel = ( ResistivityBlock::getInstance() )->getNumResistivityBlockNotFixed();

	numModel+= ( ObservedData::getInstance() )->getNumDistortionParamsNotFixed();

	return numModel;

}

// Get damping factor for resistivity value
double Inversion::alphawithmaxcurvature() const {
	std::cout << "Inside alphawithmaxcurvature, alphawithmaxc is: " << alphawithmaxc << std::endl;
	std::cout << "Address of alphawithmaxc: " << &alphawithmaxc << std::endl;

	return alphawithmaxc;
}

// Output number of model to log file
void Inversion::outputNumberOfModel() const{

	const int numBlockNotFixed = ( ResistivityBlock::getInstance() )->getNumResistivityBlockNotFixed();
	const int numDistortionParams = ( ObservedData::getInstance() )->getNumDistortionParamsNotFixed();

	OutputFiles::m_logFile << "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
	OutputFiles::m_logFile << "# Total number of model parameter : " << getNumberOfModel() << "." << std::endl;
	if( numBlockNotFixed > 0 ){
		OutputFiles::m_logFile << "#  - Number of modifiable resistivity values : " << numBlockNotFixed << "." << std::endl;
	}
	if( ( AnalysisControl::getInstance() )->estimateDistortionMatrix() ){
		OutputFiles::m_logFile << "#  - Number of modifiable distortion parameters : " << numDistortionParams << "." << std::endl;
	}
	OutputFiles::m_logFile << "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;


}
