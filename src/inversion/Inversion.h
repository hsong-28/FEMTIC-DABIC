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
#ifndef DBLDEF_INVERSION
#define DBLDEF_INVERSION

#include <iostream>
#include <complex>
#include <vector>
#include "Forward3D.h"
#include "RougheningSquareMatrix.h"
#include "DoubleSparseSquareSymmetricMatrix.h"

class RougheningMatrix;

// Class of inversion
class Inversion{

public:
	enum InversionMethod{
		GAUSS_NEWTON_MODEL_SPECE = 0,
		GAUSS_NEWTON_DATA_SPECE = 1,
		ABIC_DATA_SPECE = 2,
		OCCAM_DATA_SPECE = 3,
		LINEAR_LCURVE_DATA_SPECE = 4,
		NONLINEAR_LCURVE_DATA_SPECE = 5,
		DATA_FIT_COOLING_DATA_SPECE = 6,
		LCURVE_DATA_SPECE = LINEAR_LCURVE_DATA_SPECE,
	};

	// Constructor
	explicit Inversion();

	// Constructor
	explicit Inversion( const int nModel, const int nData );

	// Destructor
	virtual ~Inversion();

	// Calculate derivatives of EM field
	void calculateDerivativesOfEMField( Forward3D* const ptrForward3D, const double freq, const int iPol );
	
	// Calculate sensitivity matrix
	void calculateSensitivityMatrix( const int freqIDAmongThisPE, const double freq );
	
	// Allocate memory for sensitivity values
	void allocateMemoryForSensitivityScalarValues();
	
	// Release memory of sensitivity values
	void releaseMemoryOfSensitivityScalarValues();
	
	// Output scalar sensitivity values to vtk file
	void outputSensitivityScalarValuesToVtk(const int interNum) const;

	// Output scalar sensitivity values to binary file
	void outputSensitivityScalarValuesToBinary( const int interNum ) const;

	// Perform inversion
	virtual void inversionCalculation() = 0;

	// Delete out-of-core file all
	void deleteOutOfCoreFileAll();

	// Get number of model
	int getNumberOfModel() const;

	// Build the production roughening state for future appraisal diagnostics.
	void buildProductionAppraisalRougheningState(
		RougheningMatrix& constrainingMatrix,
		DoubleSparseSquareSymmetricMatrix& rtrMatrix) const;

	// Output number of model to log file
	void outputNumberOfModel() const;

	// Get trade off parameter with maximum curvature
	double alphawithmaxcurvature() const;

	void setAlphawithmaxc(double value);

	double getAlphawithmaxc() const;

	void setdeterminant(double value);

	double getdeterminant() const;

	void setdeterminantRTR(double value);

	double getdeterminantRTR() const;

	void setrms(double value);

	double getrms() const;

	double getminABIC() const;

	void setabic(const std::vector<double>& values);

	const std::vector<double>& getabic() const;

	void setmratio(double value);

	double getmratio() const;

protected:
	// Calculate constraining matrix
	void calcConstrainingMatrix( DoubleSparseMatrix& constrainingMatrix ) const;

	// Calculate constraining matrix only, excluding the distortion, CG, FCM
	void calcRoughnessMatrix(DoubleSparseMatrix& constrainingMatrix) const;
	void calcRoughnessMatrix(DoubleSparseMatrix& constrainingMatrix, const int ito) const;
	void calcRoughnessMatrix_OCCAM(DoubleSparseMatrix& constrainingMatrix) const;

	// Calculate the shared Difference-filter constraining matrix used by the
	// data-space ABIC/generic roughening path.
	void calcConstrainingMatrixForDifferenceFilterShared(DoubleSparseMatrix& constrainingMatrix) const;

	// Copy model transforming jacobian matrix
	void copyModelTransformingJacobian( const int numBlockNotFixed, const int numModel, double* jacobian ) const;

	// Multiply model transforming jacobian matrix
	void multiplyModelTransformingJacobian( const int numData, const int numModel, const double* jacobian, double* matrix ) const;

private:
	// Copy constructor
	Inversion( const Inversion& rhs ){
		std::cerr << "Error : Copy constructor of the class Inversion is not implemented." << std::endl;
		exit(1);
	}

	// Copy assignment operator
	Inversion& operator=( const Inversion& rhs ){
		std::cerr << "Error : Assignment operator of the class Inversion is not implemented." << std::endl;
		exit(1);
	}

	// Number of model
	int m_numModel;

	// Number of data
	int m_numData;

	// Derivatives of EM field
	std::complex<double>* m_derivativesOfEMField[2];

	// Sensitivity values
	double* m_sensitivityScalarValues;

	double alphawithmaxc;

	double rms;
	
	double determinant;
	double determinantRTR;
	std::vector<double> abicVec;
	double mratio;
	double ABICmin;

};

#endif
