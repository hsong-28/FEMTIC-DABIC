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
#ifndef DBLDEF_OBSERVED_DATA_APPARENT_RESISTIVITY_AND_PHASE
#define DBLDEF_OBSERVED_DATA_APPARENT_RESISTIVITY_AND_PHASE

#include <vector>
#include <complex>

#include "Forward3D.h"
#include "ObservedDataStationMT.h"
#include "CommonParameters.h"
#include "MeshDataTetraElement.h"
#include "FemticHDF5CalcTypes.h"

// Observed data of apparent resistivity and phase station
class ObservedDataStationApparentResistivityAndPhase: public ObservedDataStationMT{
	public:

		// Constructer
		explicit ObservedDataStationApparentResistivityAndPhase();

		// Destructer
		~ObservedDataStationApparentResistivityAndPhase();
			
		// Read data from input file
		void inputObservedData( std::ifstream& inFile );

		// Calulate Impedance tensor
		void calculateApparentResistivityAndPhase( const double freq, const ObservedDataStationPoint* const ptrStationOfMagneticField, int& icount );

		// Initialize apparent resistivitu, phase and their errors
		void initializeApparentResistivityPhaseAndErrors();

		// Allocate memory for the apparent resistivity, phase and their errors
		void allocateMemoryForCalculatedValues();

		// Output calculated apparent resistivity and phase
		void outputCalculatedValues() const;
		
		// Calulate sensitivity matrix of apparent resistivity and phase
		void calculateSensitivityMatrix( const double freq, const int nModel,
			const ObservedDataStationPoint* const ptrStationOfMagneticField,
			const std::complex<double>* const derivativesOfEMFieldExPol,
			const std::complex<double>* const derivativesOfEMFieldEyPol,
			double* const sensitivityMatrix ) const;
	
		// Calculate data vector of this PE
		void calculateResidualVectorOfDataThisPE( const double freq, const int offset, double* vector ) const;

		// Calulate sum of square of misfit
		double calculateErrorSumOfSquaresThisPE() const;

		// Apparent resistivity is ignored for all frequencies
		bool isApparentResistivityIgnoredForAllFrequencies( const int iComp ) const;

#ifdef _HDF5_JAC
		// Collect data-error (SD) vector in same slot order as residual vector
		// (ported from femtic_v4_src, 2026-08-21).
		void collectErrorVectorThisPE( const double freq, const int offset, double* vector ) const;

#endif // _HDF5_JAC

#ifdef _HDF5_OUT
		// --- HDF5 output accessors (ported from femtic_v4_src, 2026-08-21) ---
		double getApparentResistivityXXObserved(const int i) const { return m_apparentResistivityXXObserved[i]; }
		double getApparentResistivityXYObserved(const int i) const { return m_apparentResistivityXYObserved[i]; }
		double getApparentResistivityYXObserved(const int i) const { return m_apparentResistivityYXObserved[i]; }
		double getApparentResistivityYYObserved(const int i) const { return m_apparentResistivityYYObserved[i]; }
		double getPhaseXXObserved(const int i) const { return m_PhaseXXObserved[i]; }
		double getPhaseXYObserved(const int i) const { return m_PhaseXYObserved[i]; }
		double getPhaseYXObserved(const int i) const { return m_PhaseYXObserved[i]; }
		double getPhaseYYObserved(const int i) const { return m_PhaseYYObserved[i]; }
		double getApparentResistivityXXSD(const int i) const { return m_apparentResistivityXXSD[i]; }
		double getApparentResistivityXYSD(const int i) const { return m_apparentResistivityXYSD[i]; }
		double getApparentResistivityYXSD(const int i) const { return m_apparentResistivityYXSD[i]; }
		double getApparentResistivityYYSD(const int i) const { return m_apparentResistivityYYSD[i]; }
		double getPhaseXXSD(const int i) const { return m_PhaseXXSD[i]; }
		double getPhaseXYSD(const int i) const { return m_PhaseXYSD[i]; }
		double getPhaseYXSD(const int i) const { return m_PhaseYXSD[i]; }
		double getPhaseYYSD(const int i) const { return m_PhaseYYSD[i]; }

		// Collect this PE's calculated apparent-resistivity/phase values for
		// results_iterN.h5. Component order 0-3=rhoXX,XY,YX,YY, 4-7=phsXX,
		// XY,YX,YY, matching OutputHDF5.cpp's /data row layout.
		// Added by Volker Rath (DIAS) with the help of Claude Sonnet 5
		// (Anthropic), 2026-09-13.
		void collectCalculatedValuesForHDF5( std::vector<FemticHDF5CalcRow>& rows ) const;

#endif // _HDF5_OUT

	private:

		double* m_apparentResistivityXXObserved;
		double* m_apparentResistivityXYObserved;
		double* m_apparentResistivityYXObserved;
		double* m_apparentResistivityYYObserved;
		double* m_PhaseXXObserved;
		double* m_PhaseXYObserved;
		double* m_PhaseYXObserved;
		double* m_PhaseYYObserved;

		double* m_apparentResistivityXXSD;
		double* m_apparentResistivityXYSD;
		double* m_apparentResistivityYXSD;
		double* m_apparentResistivityYYSD;
		double* m_PhaseXXSD;
		double* m_PhaseXYSD;
		double* m_PhaseYXSD;
		double* m_PhaseYYSD;

		double* m_apparentResistivityXXCalculated;
		double* m_apparentResistivityXYCalculated;
		double* m_apparentResistivityYXCalculated;
		double* m_apparentResistivityYYCalculated;
		double* m_PhaseXXCalculated;
		double* m_PhaseXYCalculated;
		double* m_PhaseYXCalculated;
		double* m_PhaseYYCalculated;

		double* m_apparentResistivityXXResidual;
		double* m_apparentResistivityXYResidual;
		double* m_apparentResistivityYXResidual;
		double* m_apparentResistivityYYResidual;
		double* m_PhaseXXResidual;
		double* m_PhaseXYResidual;
		double* m_PhaseYXResidual;
		double* m_PhaseYYResidual;

		int* m_dataIDOfApparentResistivityXX;
		int* m_dataIDOfApparentResistivityXY;
		int* m_dataIDOfApparentResistivityYX;
		int* m_dataIDOfApparentResistivityYY;
		int* m_dataIDOfPhaseXX;
		int* m_dataIDOfPhaseXY;
		int* m_dataIDOfPhaseYX;
		int* m_dataIDOfPhaseYY;

		// Calculate log 10 error of apparent resistivity
		double calcLog10ErrorOfApparentResistivity( const int freqIDGlobalInSta, const int iComp ) const;

		// Determine whether to use impedance tensor instead of phase by checking sign differece of real part of impedance tensor
		bool isUsedImpedanceTensor( const double phaseObs, const double phaseError, const double phaseCalc ) const;

		// Determine whether to use impedance tensor instead of phase by checking sign differece of real part of impedance tensor from frequeny IDs
		bool isUsedImpedanceTensorFromFreqIDs( const int freqIDThisPEInSta, const int iComp ) const;

		// Copy constructer
		ObservedDataStationApparentResistivityAndPhase(const ObservedDataStationApparentResistivityAndPhase& rhs);

		// Copy assignment operator
		ObservedDataStationApparentResistivityAndPhase& operator=(const ObservedDataStationApparentResistivityAndPhase& rhs);
};
#endif
