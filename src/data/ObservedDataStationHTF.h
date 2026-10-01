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
#ifndef DBLDEF_OBSERVED_DATA_HTF
#define DBLDEF_OBSERVED_DATA_HTF

#include <vector>
#include <complex>

#include "ObservedDataStationPoint.h"
#include "Forward3D.h"
#include "CommonParameters.h"
#include "MeshDataTetraElement.h"
#include "FemticHDF5CalcTypes.h"

// Observed data of HTF station
class ObservedDataStationHTF: public ObservedDataStationPoint{
	public:
		// Constructer
		explicit ObservedDataStationHTF();

		// Destructer
		~ObservedDataStationHTF();
			
		// Read data from input file
		void inputObservedData( std::ifstream& inFile );

		// Calulate horizontal magnetic field transfer function
		void calculateHTF( const double freq, const ObservedDataStationPoint* const ptrStationOfMagneticField, int& icount );

		// Initialize horizontal magnetic field transfer functions and errors
		void initializeHTFsAndErrors();

		// Allocate memory for the calculated values of horizontal magnetic field transfer functions and errors
		void allocateMemoryForCalculatedValues();

		// Output calculated values of horizontal magnetic field transfer functions
		void outputCalculatedValues() const;

		// Calulate sensitivity matrix of HTF
		void calculateSensitivityMatrix( const double freq, const int nModel,
			const ObservedDataStationPoint* const ptrStationOfMagneticField,
			const std::complex<double>* const derivativesOfEMFieldExPol,
			const std::complex<double>* const derivativesOfEMFieldEyPol,
			double* const sensitivityMatrix ) const;
	
		// Calculate data vector of this PE
		void calculateResidualVectorOfDataThisPE( const double freq, const int offset, double* vector ) const;

		// Calulate sum of square of misfit
		double calculateErrorSumOfSquaresThisPE() const;

#ifdef _HDF5_JAC
		// Collect data-error (SD) vector in same slot order as residual vector
		// (ported from femtic_v4_src, 2026-08-21).
		void collectErrorVectorThisPE( const double freq, const int offset, double* vector ) const;

#endif // _HDF5_JAC

#ifdef _HDF5_OUT
		// --- HDF5 output accessors (ported from femtic_v4_src, 2026-08-21) ---
		std::complex<double> getTxxObserved(const int i) const { return m_TxxObserved[i]; }
		std::complex<double> getTxyObserved(const int i) const { return m_TxyObserved[i]; }
		std::complex<double> getTyxObserved(const int i) const { return m_TyxObserved[i]; }
		std::complex<double> getTyyObserved(const int i) const { return m_TyyObserved[i]; }
		double getTxxSDRe(const int i) const { return m_TxxSD[i].realPart; }
		double getTxxSDIm(const int i) const { return m_TxxSD[i].imagPart; }
		double getTxySDRe(const int i) const { return m_TxySD[i].realPart; }
		double getTxySDIm(const int i) const { return m_TxySD[i].imagPart; }
		double getTyxSDRe(const int i) const { return m_TyxSD[i].realPart; }
		double getTyxSDIm(const int i) const { return m_TyxSD[i].imagPart; }
		double getTyySDRe(const int i) const { return m_TyySD[i].realPart; }
		double getTyySDIm(const int i) const { return m_TyySD[i].imagPart; }

		// Collect this PE's calculated HTF values for results_iterN.h5.
		// Component order 0=Txx 1=Txy 2=Tyx 3=Tyy, matching
		// OutputHDF5.cpp's /data row layout.
		// Added by Volker Rath (DIAS) with the help of Claude Sonnet 5
		// (Anthropic), 2026-09-13.
		void collectCalculatedValuesForHDF5( std::vector<FemticHDF5CalcRow>& rows ) const;

#endif // _HDF5_OUT

	private:
		std::complex<double>* m_TxxObserved;
		std::complex<double>* m_TxyObserved;
		std::complex<double>* m_TyxObserved;
		std::complex<double>* m_TyyObserved;

		CommonParameters::DoubleComplexValues* m_TxxSD;
		CommonParameters::DoubleComplexValues* m_TxySD;
		CommonParameters::DoubleComplexValues* m_TyxSD;
		CommonParameters::DoubleComplexValues* m_TyySD;

		std::complex<double>* m_TxxCalculated;
		std::complex<double>* m_TxyCalculated;
		std::complex<double>* m_TyxCalculated;
		std::complex<double>* m_TyyCalculated;

		CommonParameters::DoubleComplexValues* m_TxxResidual;
		CommonParameters::DoubleComplexValues* m_TxyResidual;
		CommonParameters::DoubleComplexValues* m_TyxResidual;
		CommonParameters::DoubleComplexValues* m_TyyResidual;

		CommonParameters::InitComplexValues* m_dataIDOfTxx;
		CommonParameters::InitComplexValues* m_dataIDOfTxy;
		CommonParameters::InitComplexValues* m_dataIDOfTyx;
		CommonParameters::InitComplexValues* m_dataIDOfTyy;

		// Copy constructer
		ObservedDataStationHTF(const ObservedDataStationHTF& rhs);

		// Copy assignment operator
		ObservedDataStationHTF& operator=(const ObservedDataStationHTF& rhs);

};

#endif
