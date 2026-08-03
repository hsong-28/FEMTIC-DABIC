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
#ifndef Inversion_ABIC
#define Inversion_ABIC

#include "Inversion.h"
#include "RougheningMatrix.h"
#include "Forward3D.h"
#include "Forward3DBrickElement0thOrder.h"
#include "Forward3DTetraElement0thOrder.h"
#include "Forward3DNonConformingHexaElement0thOrder.h"
#include "MeshData.h"

// Class of inversion using Gauss-Newton method (data space)
class InversionGaussNewtonDataSpace_ABIC: public Inversion {

public:
	// Constructer
	explicit InversionGaussNewtonDataSpace_ABIC();

	// Constructer
	explicit InversionGaussNewtonDataSpace_ABIC( const int nModel, const int nData );

	// Destructer
	virtual ~InversionGaussNewtonDataSpace_ABIC();

	// Perform inversion
	virtual void inversionCalculation();

	// Read sensitivity matrix
	void readSensitivityMatrix( const std::string& fileName, int& numData, int& numModel, double*& sensitivityMatrix ) const;

private:
	// Copy constructer
	InversionGaussNewtonDataSpace_ABIC( const InversionGaussNewtonDataSpace_ABIC& rhs ){
		std::cerr << "Error : Copy constructer of the class InversionGaussNewtonDataSpace is not implemented." << std::endl;
		exit(1);
	}

	// Copy assignment operator
	InversionGaussNewtonDataSpace_ABIC& operator=( const InversionGaussNewtonDataSpace_ABIC& rhs ){
		std::cerr << "Error : Assignment operator of the class InversionGaussNewtonDataSpace is not implemented." << std::endl;
		exit(1);
	}

	// Calculate constraining matrix for difference filter
	void calcConstrainingMatrixForDifferenceFilter( DoubleSparseMatrix& constrainingMatrix ) const;

	// Compatibility wrapper for the ABIC trial forward-response evaluator.
	void calcForwardComputation_ABIC();

	// Evaluate forward responses for the current ABIC trial without calculating
	// derivatives or sensitivity matrices.
	void evaluateForwardResponsesForCurrentABICTrial();

	// Delete per-rank modified sensitivity-matrix files before the response-only
	// ABIC trial solve.
	void deleteModifiedSensitivityMatrixFilesForCurrentRank() const;

};



#endif
