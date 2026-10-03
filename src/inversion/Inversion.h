//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Original FEMTIC source:
// Copyright (c) 2021 Yoshiya Usui
//
// FEMTIC-DABIC modifications and extensions:
// Copyright (c) 2025-2026 Han Song
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

	// Perform MPI_Allreduce and return globally-summed sensitivity values.
	// Caller is responsible for deleting the returned array.
	// Ported from femtic_v4_src, 2026-08-21. Made unconditional (no longer
	// requires _HDF5_OUT) 2026-09-14, since
	// ResistivityBlock::outputSensitivityBlock() also uses it
	// unconditionally -- see Inversion.cpp and AnalysisControl.cpp.
	double* getSensitivityScalarValuesReduced() const;

#ifdef _HDF5_JAC
	// Assemble the full dense Jacobian for iterNum from the out-of-core
	// per-frequency sensitivity-matrix files (sensMatFreq<N>, written by
	// calculateSensitivityMatrix() whenever doesCalculateSensitivity(iter)
	// is true) together with the per-datum error/SD vector, and write
	// jacobian.h5. This is the single, shared implementation used by:
	//   - InversionGaussNewtonModelSpace/DataSpace::inversionCalculation(),
	//     when called with writeJacobianHDF5=true for the deterministic
	//     last iteration at which the Jacobian is computed without early
	//     convergence (iter == m_iterationNumMax - 1);
	//   - AnalysisControl::run(), directly, when the inversion converges
	//     *before* that scheduled iteration -- inversionCalculation() is
	//     never called for the iteration at which convergence is detected
	//     (a converged model needs no further update), so the Jacobian
	//     output that lives inside it never ran on early convergence.
	//     Fixed 2026-09-11 (ported from femtic_v4_src); previously this
	//     case only printed a log note advising a rerun with a smaller
	//     ITERATION_NUM_MAX.
	// COLLECTIVE: must be called by every PE (it performs MPI_Allgather
	// and MPI_Gatherv internally); only PE 0 actually reads the
	// out-of-core files and writes jacobian.h5.
	void assembleAndWriteJacobianToHDF5( const int iterNum ) const;
#endif // _HDF5_JAC

	// Perform inversion
	// writeJacobianHDF5: when true and _HDF5_JAC is enabled, dump the full
	// dense Jacobian for this iteration to jacobian.h5 (fixed filename,
	// overwritten each time). Callers should only pass true for the last
	// iteration at which the Jacobian is computed, since Femtic Jacobians
	// can be very large; see AnalysisControl::run() for how this is
	// determined. Default false so existing callers/overrides are unaffected.
	// Note: InversionGaussNewtonDataSpace/ModelSpace (TO_Fixed) and the
	// OCCAM/L-curve/ABIC trade-off-parameter-search variants (via
	// AnalysisControlOCCAMLineSearch.cpp / AnalysisControl.cpp) all act on
	// this flag; each of the latter's many trial inversionCalculation()
	// calls per outer iteration shares the same flag, so during the final
	// outer iteration jacobian.h5 is (harmlessly) overwritten by each trial
	// in turn and ends up holding the last one evaluated.
	virtual void inversionCalculation( const bool writeJacobianHDF5 = false ) = 0;

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
