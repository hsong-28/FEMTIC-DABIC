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
#include <stddef.h> // For null pointer
#include <stdlib.h> // For exit
#include <string.h> // For memcpy
#include <iostream>
#include <assert.h>
#include "RougheningMatrix.h"
#include "OutputFiles.h"
#include "AnalysisControl.h"
#include "ResistivityBlock.h"

#include <iomanip>
#include <algorithm>

// Default Constructer
RougheningMatrix::RougheningMatrix() : DoubleSparseMatrix()
{
}

// Constructer
RougheningMatrix::RougheningMatrix(const int nrows, const int ncols, const int nRhs) : DoubleSparseMatrix(nrows, ncols, nRhs)
{
}

// Destructer
RougheningMatrix::~RougheningMatrix()
{
}

// Set matrix structure ( locations of non-zero components ) by triplet format
// Note : This function must be called BEFORE the matrix is converted into CRS format
void RougheningMatrix::setStructureByTripletFormat(const int row, const int col)
{
	DoubleSparseMatrix::setStructureByTripletFormat(row, col);
}

// Set matrix structure ( locations of non-zero components ) and add values by triplet format
void RougheningMatrix::setStructureAndAddValueByTripletFormat(const int row, const int col, const double val)
{
	DoubleSparseMatrix::setStructureAndAddValueByTripletFormat(row, col, val);
}

// Add non-zero value to matrix
// Note : This function must be called AFTER the matrix is converted into CRS format
void RougheningMatrix::addNonZeroValues(const int row, const int col, const double val)
{
	DoubleSparseMatrix::addNonZeroValues(row, col, val);
}

// Make [R]T[R] matrix, where [R] is a constraining matrix
void RougheningMatrix::makeRTRMatrix(DoubleSparseSquareSymmetricMatrix &RTRMatrix, const double smallValueOnDiagonals) const
{

	assert(!m_matrixTripletFormat);

	const AnalysisControl *const ptrAnalysisControl = AnalysisControl::getInstance();
	const bool minnorm = ptrAnalysisControl->MinNormInv();
	const bool LM = ptrAnalysisControl->LMdamping();
	const double dampingofLM = ptrAnalysisControl->getDampingofLM();
	const int numResistivityBlockNotFixed = (ResistivityBlock::getInstance())->getNumResistivityBlockNotFixed();

	RTRMatrix.setDegreeOfEquation(m_numColumns);
	const int pValue = ptrAnalysisControl->getDegreeOfLpMinimumNorm();
	const double diffLog10RhoMin = ptrAnalysisControl->getLowerLimitOfDifflog10RhoForLpMinimumNorm();
	const double diffLog10RhoMax = ptrAnalysisControl->getUpperLimitOfDifflog10RhoForLpMinimumNorm();
	const double smallvalueofMinimumSupport = ptrAnalysisControl->getSmallValueOfMinimumSupport();

	for (int iRow = 0; iRow < m_numRows; ++iRow)
	{
		for (int iColRight = m_rowIndex[iRow]; iColRight < m_rowIndex[iRow + 1]; ++iColRight)
		{
			const int col = m_columns[iColRight];
			for (int iColLeft = m_rowIndex[iRow]; iColLeft <= iColRight; ++iColLeft)
			{
				const int row = m_columns[iColLeft];
				const double value = m_values[iColRight] * m_values[iColLeft];
				RTRMatrix.setStructureAndAddValueByTripletFormat(row, col, value);
			}
		}
	}
	// std::cout << "m_numColumns = " << m_numColumns  << std::endl;

	if (LM)
	{
		for (int iCol = 0; iCol < m_numColumns; ++iCol)
		{
			RTRMatrix.setStructureAndAddValueByTripletFormat(iCol, iCol, (smallValueOnDiagonals + dampingofLM));
		}
	}
	else
	{
		for (int iCol = 0; iCol < m_numColumns; ++iCol)
		{
			RTRMatrix.setStructureAndAddValueByTripletFormat(iCol, iCol, smallValueOnDiagonals);
		}
	}

	if (minnorm)
	{
		const double factor = ptrAnalysisControl->getTradeOffParameterForResistivityValue();
		const double factor1 = ptrAnalysisControl->getTradeOffParameterForMinNorm();
		for (int iCol = 0; iCol < numResistivityBlockNotFixed; ++iCol)
		{
			int modelID2blockID = (ResistivityBlock::getInstance())->getBlockIDFromModelID(iCol);
			double resistivityvalue(1.0);
			double referencevalue(1.0);
			resistivityvalue = (ResistivityBlock::getInstance())->getResistivityValuesFromBlockID(modelID2blockID);
			referencevalue = (ResistivityBlock::getInstance())->getReferencemodelFromBlockID(modelID2blockID);
			double diffLog10rho = fabs((log10(resistivityvalue) - log10(referencevalue)));
			if (pValue >= 1)
			{
			if (diffLog10rho < diffLog10RhoMin)
			{
				diffLog10rho = diffLog10RhoMin;
			}
			if (diffLog10rho > diffLog10RhoMax)
			{
				diffLog10rho = diffLog10RhoMax;
			}
				const double weight = 0.5 * pValue * pow(diffLog10rho, pValue - 2); // if pValue = 2, weight == 1
				if (ptrAnalysisControl->ABICinversion())
				{
					// α2 * β2
					RTRMatrix.setStructureAndAddValueByTripletFormat(iCol, iCol, weight * factor * factor * factor1 * factor1);
				}
				else
				{
					//  β2
					RTRMatrix.setStructureAndAddValueByTripletFormat(iCol, iCol, weight * factor1 * factor1);
				}
			}
			else if (pValue == 0)
			{
				const double weight = pow((pow(diffLog10rho, 2) + smallvalueofMinimumSupport),-1); // if pValue = 2, weight == 1
				if (ptrAnalysisControl->ABICinversion())
				{
					// α2 * β2
					RTRMatrix.setStructureAndAddValueByTripletFormat(iCol, iCol, weight * factor * factor * factor1 * factor1);
				}
				else
				{
					//  β2
					RTRMatrix.setStructureAndAddValueByTripletFormat(iCol, iCol, weight * factor1 * factor1);
				}
			}
		}
	}
	RTRMatrix.convertToCRSFormat();
}

// Make [R]T[R] matrix, where [R] is a constraining matrix
void RougheningMatrix::makeRTRMatrix1(DoubleSparseSquareSymmetricMatrix &RTRMatrix1, const double smallValueOnDiagonals) const
{

	assert(!m_matrixTripletFormat);
	RTRMatrix1.setDegreeOfEquation(m_numColumns);
	for (int iRow = 0; iRow < m_numRows; ++iRow)
	{
		for (int iColRight = m_rowIndex[iRow]; iColRight < m_rowIndex[iRow + 1]; ++iColRight)
		{
			const int col = m_columns[iColRight];
			for (int iColLeft = m_rowIndex[iRow]; iColLeft <= iColRight; ++iColLeft)
			{
				const int row = m_columns[iColLeft];
				const double value = m_values[iColRight] * m_values[iColLeft];
				RTRMatrix1.setStructureAndAddValueByTripletFormat(row, col, value);
			}
		}
	}
	for (int iCol = 0; iCol < m_numColumns; ++iCol)
	{
		RTRMatrix1.setStructureAndAddValueByTripletFormat(iCol, iCol, smallValueOnDiagonals);
	}
	RTRMatrix1.convertToCRSFormat();
}

// Calculate model roughness
double RougheningMatrix::calcModelRoughness(const double *modelVec) const
{

	assert(!m_matrixTripletFormat);

	double modelRoughness(0.0);
	for (int i = 0; i < m_numRows; ++i)
	{
		double work(0.0);
		for (int j = m_rowIndex[i]; j < m_rowIndex[i + 1]; ++j)
		{
			work += m_values[j] * modelVec[m_columns[j]];
#ifdef _DEBUG_WRITE
			std::cout << "i m_rowIndex m_values modelVec: " << i << " " << m_rowIndex[i] << " " << m_values[j] << " " << modelVec[m_columns[j]] << std::endl;
#endif
		}
		modelRoughness += pow(m_rightHandSideVector[i] - work, 2);

#ifdef _DEBUG_WRITE
		std::cout << "modelRoughness m_rightHandSideVector work: " << modelRoughness << " " << m_rightHandSideVector[i] << " " << work << std::endl;
#endif
	}
	return modelRoughness;
}

// Calculate vector of model roughness
void RougheningMatrix::calcVectorOfModelRoughness(const double *modelVec, double *roughnessVec) const
{

	assert(!m_matrixTripletFormat);

	calcMatrixVectorProduct(modelVec, roughnessVec);

	for (int i = 0; i < m_numRows; ++i)
	{
		roughnessVec[i] = m_rightHandSideVector[i] - roughnessVec[i];
	}
}

// Postmultiply diagonal matrix
void RougheningMatrix::postmultiplyDiagonalMatrix(const double *diagMatrix)
{

	assert(!m_matrixTripletFormat);

	for (int i = 0; i < m_numRows; ++i)
	{
		for (int j = m_rowIndex[i]; j < m_rowIndex[i + 1]; ++j)
		{
			m_values[j] *= diagMatrix[m_columns[j]];
		}
	}
}

// Copy constructer
RougheningMatrix::RougheningMatrix(const RougheningMatrix &matrix)
{
	std::cerr << "Error : Copy constructer of the class RougheningMatrix is not implemented." << std::endl;
	exit(1);
}

// Assignment operator
RougheningMatrix &RougheningMatrix::operator=(const RougheningMatrix &rhs)
{
	std::cerr << "Error : Assignment operator of the class RougheningMatrix is not implemented." << std::endl;
	exit(1);
}

// Output roughening matrix
void RougheningMatrix::outputRougheningMatrix(const std::string &fileName) const
{

	assert(!m_hasConvertedToCRSFormat);

	std::ofstream ofs(fileName.c_str());

	if (ofs.fail())
	{
		OutputFiles::m_logFile << "File open error : " << fileName.c_str() << " !!" << std::endl;
		exit(1);
	}

	ofs << std::setw(10) << m_numRows << std::endl;

	for (int irow = 0; irow < m_numRows; ++irow)
	{
		const std::map<int, double>::iterator itEnd = m_matrixTripletFormat[irow].end();
		const int numNonzeros = static_cast<int>(m_matrixTripletFormat[irow].size());
		ofs << std::setw(10) << irow << std::endl;
		ofs << std::setw(10) << numNonzeros;
		for (std::map<int, double>::iterator it = m_matrixTripletFormat[irow].begin(); it != itEnd; ++it)
		{
			ofs << std::setw(15) << it->first;
		}
		ofs << std::endl;
		if (numNonzeros > 0)
		{
			ofs << std::setw(10) << "";
			for (std::map<int, double>::iterator it = m_matrixTripletFormat[irow].begin(); it != itEnd; ++it)
			{
				ofs << std::setw(15) << std::scientific << std::setprecision(6) << it->second;
			}
			ofs << std::endl;
		}
	}

	ofs.close();
}