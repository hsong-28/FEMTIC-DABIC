//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
// Modified by Dieno Diba (c) 2023
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
#include "ResistivityBlock.h"
#include "ConstrainingModel.h"
#include "MeshDataBrickElement.h"
#include "MeshDataNonConformingHexaElement.h"
#include "OutputFiles.h"
#include "ObservedData.h"
#ifdef _ANISOTOROPY
#include "Util.h"
#endif
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <iomanip>
#include <vector>
#include <algorithm>

// Return the the instance of the class
ConstrainingModel* ConstrainingModel::getInstance(){
   	static ConstrainingModel instance;// The only instance
  	return &instance;
}

// Constructor
ConstrainingModel::ConstrainingModel():
	m_elementID2blockID(NULL),
	m_blockID2modelID(NULL),
	m_modelID2blockID(NULL),
	m_numConstrainingBlockTotal(0),
	m_numResistivityBlockNotFixed(0),
	m_constrainingValues(NULL),
	m_resistivityValuesPre(NULL),
	m_resistivityValuesUpdatedFull(NULL),
	m_resistivityValuesMin(NULL),
	m_resistivityValuesMax(NULL),
	m_weightingConstants(NULL),
	m_fixResistivityValues(NULL),
	m_isolated(NULL),
	m_rougheningMatrix(),
	m_includeBottomResistivity(false),
	m_blockID2Elements(NULL),
	m_bottomResistivity(1.0),
	m_roughningFactorAtBottom(1.0),
	m_addSmallValueToDiagonals(false),
	m_smallValueAddedToDiagonals(0.0),
	m_minDistanceToBounds(0.01),
	m_inverseDistanceWeightingFactor(1.0),
	m_typeBoundConstraints(ResistivityBlock::SIMPLE_BOUND_CONSTRAINING)
{}

// Destructor
ConstrainingModel::~ConstrainingModel(){

	if( m_elementID2blockID != NULL){
		delete[] m_elementID2blockID;
		m_elementID2blockID = NULL;
	}

	if( m_blockID2modelID != NULL){
		delete[] m_blockID2modelID;
		m_blockID2modelID = NULL;
	}

	if( m_modelID2blockID != NULL){
		delete[] m_modelID2blockID;
		m_modelID2blockID = NULL;
	}

	if( m_constrainingValues != NULL){
		delete[] m_constrainingValues;
		m_constrainingValues = NULL;
	}

	if( m_resistivityValuesPre != NULL){
		delete[] m_resistivityValuesPre;
		m_resistivityValuesPre = NULL;
	}

	if( m_resistivityValuesUpdatedFull != NULL){
		delete[] m_resistivityValuesUpdatedFull;
		m_resistivityValuesUpdatedFull = NULL;
	}

	if( m_resistivityValuesMin != NULL){
		delete[] m_resistivityValuesMin;
		m_resistivityValuesMin = NULL;
	}

	if( m_resistivityValuesMax != NULL){
		delete[] m_resistivityValuesMax;
		m_resistivityValuesMax = NULL;
	}

	if( m_weightingConstants != NULL){
		delete[] m_weightingConstants;
		m_weightingConstants = NULL;
	}

	if( m_fixResistivityValues != NULL){
		delete[] m_fixResistivityValues;
		m_fixResistivityValues = NULL;
	}

	if( m_isolated != NULL){
		delete[] m_isolated;
		m_isolated = NULL;
	}

	if( m_blockID2Elements != NULL){
		delete[] m_blockID2Elements;
		m_blockID2Elements = NULL;
	}

	m_rougheningMatrix.releaseMemory();

}

// Read data of constraining model from input file
void ConstrainingModel::inputConstrainingModel(){

	std::ostringstream inputFile;
	inputFile << "constrainingmodel.dat";
	std::ifstream inFile( inputFile.str().c_str(), std::ios::in );

	if( inFile.fail() )
	{
        OutputFiles::m_logFile << "File open error : " << inputFile.str().c_str() << " !!" << std::endl;
		exit(1); 
	}

	int nElem(0);
	inFile >> nElem;
	if( m_elementID2blockID != NULL ){
		delete[] m_elementID2blockID;
		m_elementID2blockID = NULL;
	}
	m_elementID2blockID = new int[ nElem ];

	int nBlk(0);
	inFile >> nBlk;	
	m_numConstrainingBlockTotal = nBlk;

    //std::cout << "nConstrainBlock = " << m_numConstrainingBlockTotal << std::endl;

	if( m_blockID2modelID != NULL){
		delete[] m_blockID2modelID;
		m_blockID2modelID = NULL;
	}
	m_blockID2modelID = new int[ m_numConstrainingBlockTotal ];

	if( m_constrainingValues != NULL ){
		delete[] m_constrainingValues;
		m_constrainingValues = NULL;
	}
	m_constrainingValues = new double[ m_numConstrainingBlockTotal ];

	if( m_resistivityValuesPre != NULL){
		delete[] m_resistivityValuesPre;
		m_resistivityValuesPre = NULL;
	}
	m_resistivityValuesPre = new double[ m_numConstrainingBlockTotal ];

	if( m_resistivityValuesUpdatedFull != NULL){
		delete[] m_resistivityValuesUpdatedFull;
		m_resistivityValuesUpdatedFull = NULL;
	}
	m_resistivityValuesUpdatedFull = new double[ m_numConstrainingBlockTotal ];

	if( m_resistivityValuesMin != NULL){
		delete[] m_resistivityValuesMin;
		m_resistivityValuesMin = NULL;
	}
	m_resistivityValuesMin = new double[ m_numConstrainingBlockTotal ];

	if( m_resistivityValuesMax != NULL){
		delete[] m_resistivityValuesMax;
		m_resistivityValuesMax = NULL;
	}
	m_resistivityValuesMax = new double[ m_numConstrainingBlockTotal ];

	if( m_weightingConstants != NULL){
		delete[] m_weightingConstants;
		m_weightingConstants = NULL;
	}
	m_weightingConstants = new double[ m_numConstrainingBlockTotal ];

	if( m_fixResistivityValues != NULL ){
		delete[] m_fixResistivityValues;
		m_fixResistivityValues = NULL;
	}
	m_fixResistivityValues = new bool[ m_numConstrainingBlockTotal ];

	if( m_isolated != NULL){
		delete[] m_isolated;
		m_isolated = NULL;
	}
	m_isolated = new bool[ m_numConstrainingBlockTotal ];

	for( int i = 0; i < m_numConstrainingBlockTotal; ++i ){
		m_constrainingValues[i] = 0.0;
		m_resistivityValuesPre[i] = 0.0;
		m_resistivityValuesUpdatedFull[i] = 0.0;
		m_resistivityValuesMin[i] = 0.0;
		m_resistivityValuesMax[i] = 0.0;
		m_weightingConstants[i] = 1.0;
		m_fixResistivityValues[i] = false;
		m_isolated[i] = false;
	}

#ifdef _DEBUG_WRITE
	std::cout << nElem << " " << m_numResistivityBlockTotal << std::endl; // For debug
#endif
	
	for( int iElem = 0; iElem < nElem; ++iElem ){
		int idum(0);
		int iblk(0);// Resistivity block ID
		inFile >> idum >> iblk;
		m_elementID2blockID[ iElem ] = iblk;

		if( iblk >= m_numConstrainingBlockTotal || iblk < 0 )	{
			OutputFiles::m_logFile << "Error : Resistivity block ID " << iblk << " of element " << iElem << " is improper !!" << std::endl;
			exit(1);
		}		

#ifdef _DEBUG_WRITE
		std::cout << iElem << " " << iblk << std::endl; // For debug
#endif
	}
	
	const bool dataSpaceInversionMethod = ( ( AnalysisControl::getInstance() )->getInversionMethod() == Inversion::GAUSS_NEWTON_DATA_SPECE );
	m_numResistivityBlockNotFixed = 0;
#ifdef _ANISOTOROPY
	int counterAnisotropicBlock(0);
#endif
	for( int iBlk = 0; iBlk < m_numConstrainingBlockTotal; ++iBlk ){
		int idum(0);
		int itype(0);

#ifdef _ANISOTOROPY
		inFile >> idum;
		if( idum != iBlk ){
			OutputFiles::m_logFile << "Error : Block ID is wrong !!" << std::endl;
			exit(1);
		}
		bool anisotropy(false);
		if( ( AnalysisControl::getInstance() )->isAnisotropyConsidered() ){
			inFile >> idum;
			if( idum != 0 ){
				anisotropy = true;
			}
		}
		if( anisotropy ){
			m_mapBlockIDWithAnisotropyToIndex.insert( std::make_pair(iBlk, counterAnisotropicBlock) );
			++counterAnisotropicBlock;
			m_resistivityValues[iBlk] = -9999.999;
			CommonParameters::Vector3D resistivity;
			inFile >> resistivity.X >> resistivity.Y >> resistivity.Z;
			if( resistivity.X <= 0.0 ){
				OutputFiles::m_logFile << "Error : Resistivity component XX of block " << iBlk << " is less than or equal to zero !! : " << resistivity.X << std::endl;
				exit(1);
			}
			if( resistivity.Y <= 0.0 ){
				OutputFiles::m_logFile << "Error : Resistivity component YY of block " << iBlk << " is less than or equal to zero !! : " << resistivity.Y << std::endl;
				exit(1);
			}
			if( resistivity.Z <= 0.0 ){
				OutputFiles::m_logFile << "Error : Resistivity component ZZ of block " << iBlk << " is less than or equal to zero !! : " << resistivity.Z << std::endl;
				exit(1);
			}
			m_resistivityValuesAxialAnisotropy.push_back(resistivity);
			m_resistivityValuesAxialAnisotropyPre.push_back(resistivity);
			m_resistivityValuesAxialAnisotropyFull.push_back(resistivity);
			double strike(0.0);
			double dip(0.0);
			double slant(0.0);
			inFile >> strike >> dip >> slant;
			strike *= CommonParameters::deg2rad;
			dip *= CommonParameters::deg2rad;
			slant *= CommonParameters::deg2rad;
			m_axialAnisotropyStrileAngle.push_back(strike);
			m_axialAnisotropyDipAngle.push_back(dip);
			m_axialAnisotropySlantAngle.push_back(slant);
		}else{
			inFile >> m_resistivityValues[iBlk];
			if( m_resistivityValues[iBlk] <= 0.0 ){
				OutputFiles::m_logFile << "Error : Resistivity value of block " << iBlk << " is less than or equal to zero !! : " << m_resistivityValues[iBlk] << std::endl;
				exit(1);
			}
		}
		inFile >> m_resistivityValuesMin[iBlk] >> m_resistivityValuesMax[iBlk] >> m_weightingConstants[iBlk] >> itype;
#else
		inFile >> idum >> m_constrainingValues[iBlk] >> m_resistivityValuesMin[iBlk] >> m_resistivityValuesMax[iBlk] >> m_weightingConstants[iBlk] >> itype;
		if( idum != iBlk ){
			OutputFiles::m_logFile << "Error : Block ID is wrong !!" << std::endl;
			exit(1);
		}
		// Dieno2023
		//if( m_constrainingValues[iBlk] <= 0.0 ){
		//	OutputFiles::m_logFile << "Error : Constraining value of block " << iBlk << " is less than or equal to zero !! : " << m_constrainingValues[iBlk] << std::endl;
		//	exit(1);
		//}
#endif
		//if( m_resistivityValuesMin[iBlk] <= 0.0 ){
		//	OutputFiles::m_logFile << "Error : Minimum resistivity value of block " << iBlk << " is less than or equal to zero !! : " << m_resistivityValuesMin[iBlk] << std::endl;
		//	exit(1);
		//}
		//if( m_resistivityValuesMax[iBlk] <= 0.0 ){
		//	OutputFiles::m_logFile << "Error : Maximum resistivity value of block " << iBlk << " is less than or equal to zero !! : " << m_resistivityValuesMax[iBlk] << std::endl;
		//	exit(1);
		//}
#ifdef _ANISOTOROPY
		if( anisotropy ){
			const CommonParameters::Vector3D resistivity = m_resistivityValuesAxialAnisotropy.back();
			if( m_resistivityValuesMax[iBlk] < resistivity.X ){
				OutputFiles::m_logFile << "Error : Maximum resistivity value ( " << m_resistivityValuesMax[iBlk] << " ) is less than initial resistivity ( " << resistivity.X << " )." << std::endl;
				exit(1);
			}
			if( m_resistivityValuesMax[iBlk] < resistivity.Y ){
				OutputFiles::m_logFile << "Error : Maximum resistivity value ( " << m_resistivityValuesMax[iBlk] << " ) is less than initial resistivity ( " << resistivity.Y << " )." << std::endl;
				exit(1);
			}
			if( m_resistivityValuesMax[iBlk] < resistivity.Z ){
				OutputFiles::m_logFile << "Error : Maximum resistivity value ( " << m_resistivityValuesMax[iBlk] << " ) is less than initial resistivity ( " << resistivity.Z << " )." << std::endl;
				exit(1);
			}
			if( m_resistivityValuesMin[iBlk] > resistivity.X ){
				OutputFiles::m_logFile << "Error : Minimum resistivity value ( " << m_resistivityValuesMax[iBlk] << " ) is greater than initial resistivity ( " << resistivity.X << " )." << std::endl;
				exit(1);
			}
			if( m_resistivityValuesMin[iBlk] > resistivity.Y ){
				OutputFiles::m_logFile << "Error : Minimum resistivity value ( " << m_resistivityValuesMax[iBlk] << " ) is greater than initial resistivity ( " << resistivity.Y << " )." << std::endl;
				exit(1);
			}
			if( m_resistivityValuesMin[iBlk] > resistivity.Z ){
				OutputFiles::m_logFile << "Error : Minimum resistivity value ( " << m_resistivityValuesMax[iBlk] << " ) is greater than initial resistivity ( " << resistivity.Z << " )." << std::endl;
				exit(1);
			}
		}else{
			if( m_resistivityValuesMax[iBlk] < m_resistivityValues[iBlk] ){
				OutputFiles::m_logFile << "Error : Maximum resistivity value ( " << m_resistivityValuesMax << " ) is less than initial resistivity ( " << m_resistivityValues[iBlk] << " )." << std::endl;
				exit(1);
			}
			if( m_resistivityValuesMin[iBlk] > m_resistivityValues[iBlk] ){
				OutputFiles::m_logFile << "Error : Minimum resistivity value ( " << m_resistivityValuesMax << " ) is greater than initial resistivity ( " << m_resistivityValues[iBlk] << " )." << std::endl;
				exit(1);
			}
		}
#else
		if( m_resistivityValuesMax[iBlk] < m_constrainingValues[iBlk] ){
			OutputFiles::m_logFile << "Error : Maximum resistivity value ( " << m_resistivityValuesMax << " ) is less than initial resistivity ( " << m_constrainingValues[iBlk] << " )." << std::endl;
			exit(1);
		}
		if( m_resistivityValuesMin[iBlk] > m_constrainingValues[iBlk] ){
			OutputFiles::m_logFile << "Error : Minimum resistivity value ( " << m_resistivityValuesMax << " ) is greater than initial resistivity ( " << m_constrainingValues[iBlk] << " )." << std::endl;
			exit(1);
		}
#endif
		if( m_weightingConstants[iBlk] <= 0.0 ){
			OutputFiles::m_logFile << "Error : Weighting constant of block " << iBlk << " is less than or equal to zero !! : " << m_weightingConstants[iBlk] << std::endl;
			exit(1);
		}
		if( dataSpaceInversionMethod && itype == FREE_AND_ISOLATED){
			OutputFiles::m_logFile << "Error : Resistivity of isolated block must be fixed when data space inverson method is selected !!" << std::endl;
			exit(1);
		}

		switch(itype){
			case ResistivityBlock::FREE_AND_CONSTRAINED:// Go through
			case ResistivityBlock::FREE_AND_ISOLATED:
				m_blockID2modelID[iBlk] = m_numResistivityBlockNotFixed++;
				break;
			case ResistivityBlock::FIXED_AND_ISOLATED:// Go through
			case ResistivityBlock::FIXED_AND_CONSTRAINED:
				m_fixResistivityValues[iBlk] = true;
				m_blockID2modelID[iBlk] = -1;
				break;
			default:
				OutputFiles::m_logFile << "Error : Type of resistivity block is unknown !! : " << itype << std::endl;
				exit(1);
				break;
		}

		switch(itype){
			case ResistivityBlock::FREE_AND_CONSTRAINED:// Go through
			case ResistivityBlock::FIXED_AND_CONSTRAINED:
				m_isolated[iBlk] = false;
				break;
			case ResistivityBlock::FIXED_AND_ISOLATED:// Go through
			case ResistivityBlock::FREE_AND_ISOLATED:
				m_isolated[iBlk] = true;
				break;
			default:
				OutputFiles::m_logFile << "Error : Type of resistivity block is unknown !! : " << itype << std::endl;
				exit(1);
				break;
		}

#ifdef _DEBUG_WRITE
		std::cout << std::setw(5) << iBlk << std::setw(15) << m_resistivityValues[iBlk] << std::setw(15)	<< m_resistivityValuesMin[iBlk]	<< std::setw(15) << m_resistivityValuesMax[iBlk]
			<< std::setw(15) << m_weightingConstants[iBlk] <<  std::setw(5) << m_fixResistivityValues[iBlk]  <<  std::setw(5) << m_blockID2modelID[iBlk] << std::endl; // For debug
#endif

	}

	if( !m_fixResistivityValues[0] ){
		OutputFiles::m_logFile << "Error : Resistivity block 0 must be the air. And, its resistivity must be fixed." << std::endl;
		exit(1);
	}
	
	inFile.close();

	memcpy( m_resistivityValuesPre, m_constrainingValues, sizeof(double)*(m_numConstrainingBlockTotal) );

	if( m_modelID2blockID != NULL){
		delete[] m_modelID2blockID;
		m_modelID2blockID = NULL;
	}
	m_modelID2blockID = new int[ m_numResistivityBlockNotFixed ];

	int icount(0);
	for( int iBlk = 0; iBlk < m_numConstrainingBlockTotal; ++iBlk ){

		if( !m_fixResistivityValues[iBlk] ){
			m_modelID2blockID[icount] = iBlk;
			++icount;
		}

	}

	if( icount != m_numResistivityBlockNotFixed ){
		OutputFiles::m_logFile << "Error : icount is not equal to m_numResistivityBlockNotFixed. icount = " << icount << " m_numResistivityBlockNotFixed = " << m_numResistivityBlockNotFixed << std::endl;
		exit(1);
	}

#ifdef _DEBUG_WRITE
	for( int iMdl = 0; iMdl < m_numResistivityBlockNotFixed; ++iMdl ){
		std::cout << " iMdl m_modelID2blockID[iMdl] : " << iMdl << " " << m_modelID2blockID[iMdl] << std::endl;
	}
#endif

	if( m_blockID2Elements != NULL){
		delete[] m_blockID2Elements;
		m_blockID2Elements = NULL;
	}
	m_blockID2Elements= new std::vector< std::pair<int,double> >[m_numConstrainingBlockTotal];

	for( int iElem = 0; iElem < nElem; ++iElem ){
		m_blockID2Elements[ m_elementID2blockID[ iElem ] ].push_back( std::make_pair(iElem,1.0) );
	}
#ifndef _LINUX
	for( int iBlk = 0; iBlk < m_numConstrainingBlockTotal; ++iBlk ){
		m_blockID2Elements[ iBlk ].shrink_to_fit();
	}
#endif

	if( m_numResistivityBlockNotFixed <= 0 ){
		OutputFiles::m_logFile << "Error : Total number of modifiable resistivity values is zero or negative !! : " << m_numResistivityBlockNotFixed << std::endl;
		exit(1);
	}

#ifdef _DEBUG_WRITE
	for( int iBlk = 0; iBlk < m_numResistivityBlockTotal; ++iBlk ){
		const int num = static_cast<int>( m_blockID2Elements[ iBlk ].size() ); 
		for( int i = 0; i < num; ++i ){
			std::cout << " m_blockID2Elements[ " << iBlk << " ][ " << i << "] : " << m_blockID2Elements[iBlk][i].first << std::endl;
		}
	}
#endif

}

// Get constraining values from resistivity block ID
double ConstrainingModel::getConstrainingValuesFromBlockID( const int iblk ) const{

	//if( iblk < 0 || iblk >= m_numResistivityBlockTotal ){
	//	OutputFiles::m_logFile << "Error : Specified block ID is out of range. iblk = " << iblk << std::endl;
	//	exit(1);
	//}
	assert( iblk >= 0 );
	assert( iblk < m_numConstrainingBlockTotal );

	if( m_constrainingValues[ iblk ] < 0 ){
		return 1.0e+20;
	}

	return m_constrainingValues[ iblk ];
}

// Get constraining values from element ID
double ConstrainingModel::getConstrainingValuesFromElemID( const int ielem ) const{

	//const int iblk = m_elementID2blockID[ ielem ];
	const int iblk = getBlockIDFromElemID(ielem);
	return getConstrainingValuesFromBlockID( iblk );
}

// Get model ID from block ID
int ConstrainingModel::getModelIDFromBlockID( const int iblk ) const{

	//if( iblk < 0 || iblk >= m_numResistivityBlockTotal ){
	//	OutputFiles::m_logFile << "Error : Specified block ID is out of range. iblk = " << iblk << std::endl;
	//	exit(1);
	//}
	assert( iblk >= 0 );
	assert( iblk < m_numConstrainingBlockTotal );

	return m_blockID2modelID[ iblk ];
}

// Get block ID from model ID
int ConstrainingModel::getBlockIDFromModelID( const int imdl ) const{

	//if( imdl < 0 || imdl >= m_numResistivityBlockNotFixed ){
	//	OutputFiles::m_logFile << "Error : Specified model ID is out of range. imdl = " << imdl << std::endl;
	//	exit(1);
	//}
	assert( imdl >= 0 );
	assert( imdl < m_numResistivityBlockNotFixed );

	return m_modelID2blockID[ imdl ];

}

// Get total number of constraining blocks
int ConstrainingModel::getNumConstrainingBlockTotal() const{
	return m_numConstrainingBlockTotal;
}
