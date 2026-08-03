//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 Yoshiya Usui
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
#ifndef InversionGaussNewtonDataSpaceLCurveFWD
#define InversionGaussNewtonDataSpaceLCurveFWD

#include "Inversion.h"
#include "RougheningMatrix.h"

// Class for cubic-spline L-curve trial inversion in data space.
class InversionGaussNewtonDataSpaceLCurve : public Inversion {

public:
	// Constructor
	explicit InversionGaussNewtonDataSpaceLCurve();

	// Constructor
	explicit InversionGaussNewtonDataSpaceLCurve( const int nModel, const int nData );

	// Destructor
	virtual ~InversionGaussNewtonDataSpaceLCurve();

	// Perform inversion
	virtual void inversionCalculation();

	// Perform inversion by the new method
	void inversionCalculationByNewMethod() const;

	// Perform inversion by the new method using inverse of [R]T[R] matrix
	void inversionCalculationByNewMethodUsingInvRTRMatrix();

	// Read sensitivity matrix
	void readSensitivityMatrix( const std::string& fileName, int& numData, int& numModel, double*& sensitivityMatrix ) const;

	// Return the the instance of the class
	//static InversionGaussNewtonDataSpaceLCurve* getInstance();



private:
	// Copy constructor
	InversionGaussNewtonDataSpaceLCurve( const InversionGaussNewtonDataSpaceLCurve& rhs ){
		std::cerr << "Error : Copy constructor of the class InversionGaussNewtonDataSpaceLCurve is not implemented." << std::endl;
		exit(1);
	}

	// Copy assignment operator
	InversionGaussNewtonDataSpaceLCurve& operator=( const InversionGaussNewtonDataSpaceLCurve& rhs ){
		std::cerr << "Error : Assignment operator of the class InversionGaussNewtonDataSpaceLCurve is not implemented." << std::endl;
		exit(1);
	}

	// Calculate constraining matrix for difference filter
	void calcConstrainingMatrixForDifferenceFilter( DoubleSparseMatrix& constrainingMatrix ) const;

	// Calculate constraining matrix for a Difference-filter L-curve trial.
	void calcConstrainingMatrixForDifferenceFilter_L (DoubleSparseMatrix& constrainingMatrix, int ito) const;



};



#endif
