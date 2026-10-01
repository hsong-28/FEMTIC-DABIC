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
#ifndef DBLDEF_OBSERVED_DATA_VTF
#define DBLDEF_OBSERVED_DATA_VTF

#include <vector>
#include <complex>

#include "ObservedDataStationPoint.h"
#include "Forward3D.h"
#include "CommonParameters.h"
#include "MeshDataTetraElement.h"
#include "FemticHDF5CalcTypes.h"

// Observed data of VTF station
class ObservedDataStationVTF: public ObservedDataStationPoint{
	public:
		// Constructer
		explicit ObservedDataStationVTF();

		// Destructer
		~ObservedDataStationVTF();
			
		// Read data from input file
		void inputObservedData( std::ifstream& inFile );
			
		// Calulate vertical magnetic field
		void calculateVerticalMagneticField( const Forward3D* const ptrForward3D, const int rhsVectorIDOfHz );

		// Calulate vertical magnetic field transfer function
		void calculateVTF( const double freq, const ObservedDataStationPoint* const ptrStationOfMagneticField, int& icount );

		// Initialize vertical magnetic field
		void initializeVerticalMagneticField( const int iPol );

		// Initialize vertical magnetic field transfer functions and errors
		void initializeVTFsAndErrors();

		// Allocate memory for the calculated values of vertical magnetic field transfer functions and errors
		void allocateMemoryForCalculatedValues();

		// Output calculated values of vertical magnetic field transfer functions
		void outputCalculatedValues() const;

		// Calulate interpolator vector of vertical magnetic field
		void calcInterpolatorVectorOfVerticalMagneticField( Forward3D* const ptrForward3D );

		// Calulate sensitivity matrix of VTF
		void calculateSensitivityMatrix( const double freq, const int nModel,
			const ObservedDataStationPoint* const ptrStationOfMagneticField,
			const std::complex<double>* const derivativesOfEMFieldExPol,
			const std::complex<double>* const derivativesOfEMFieldEyPol,
			double* const sensitivityMatrix ) const;
	
		// Calculate data vector of this PE
		void calculateResidualVectorOfDataThisPE( const double freq, const int offset, double* vector ) const;

		// Calulate sum of square of misfit
		double calculateErrorSumOfSquaresThisPE() const;

		// Cache and restore calculated response state for the selected ABIC trial.
		void cacheSelectedTrialForwardResponse();
		void restoreSelectedTrialForwardResponse();
		void clearSelectedTrialForwardResponseCache();
		bool hasSelectedTrialForwardResponseCache() const;

		// Get VTK
		bool getVTF( const double freq, std::complex<double>& Tzx, std::complex<double>& Tzy ) const;

#ifdef _HDF5_JAC
		// Collect data-error (SD) vector in same slot order as residual vector
		// (ported from femtic_v4_src, 2026-08-21).
		void collectErrorVectorThisPE( const double freq, const int offset, double* vector ) const;

#endif // _HDF5_JAC

#ifdef _HDF5_OUT
		// --- HDF5 output accessors (ported from femtic_v4_src, 2026-08-21) ---
		std::complex<double> getTzxObserved(const int i) const { return m_TzxObserved[i]; }
		std::complex<double> getTzyObserved(const int i) const { return m_TzyObserved[i]; }
		double getTzxSDRe(const int i) const { return m_TzxSD[i].realPart; }
		double getTzxSDIm(const int i) const { return m_TzxSD[i].imagPart; }
		double getTzySDRe(const int i) const { return m_TzySD[i].realPart; }
		double getTzySDIm(const int i) const { return m_TzySD[i].imagPart; }

		// Collect this PE's calculated VTF values for results_iterN.h5.
		// Component order 0=Tzx 1=Tzy, matching OutputHDF5.cpp's /data
		// row layout.
		// Added by Volker Rath (DIAS) with the help of Claude Sonnet 5
		// (Anthropic), 2026-09-13.
		void collectCalculatedValuesForHDF5( std::vector<FemticHDF5CalcRow>& rows ) const;

#endif // _HDF5_OUT

	private:
		std::complex<double>* m_TzxObserved;
		std::complex<double>* m_TzyObserved;

		CommonParameters::DoubleComplexValues* m_TzxSD;
		CommonParameters::DoubleComplexValues* m_TzySD;

		std::complex<double>* m_TzxCalculated;
		std::complex<double>* m_TzyCalculated;

		CommonParameters::DoubleComplexValues* m_TzxResidual;
		CommonParameters::DoubleComplexValues* m_TzyResidual;

		struct SelectedTrialForwardResponseSnapshot{
			std::vector<std::complex<double> > TzxCalculated;
			std::vector<std::complex<double> > TzyCalculated;
			std::vector<CommonParameters::DoubleComplexValues> TzxResidual;
			std::vector<CommonParameters::DoubleComplexValues> TzyResidual;
		};

		bool m_hasSelectedTrialForwardResponseCache;
		SelectedTrialForwardResponseSnapshot m_selectedTrialForwardResponseSnapshot;

		std::complex<double> m_HzCalculated[2];

		//int m_columnNumberOfHzInRhsMatrix;
		int m_rhsVectorIDOfHz;

		//int* m_dataIDOfTzx;
		//int* m_dataIDOfTzy;
		CommonParameters::InitComplexValues* m_dataIDOfTzx;
		CommonParameters::InitComplexValues* m_dataIDOfTzy;

		// Copy constructer
		ObservedDataStationVTF(const ObservedDataStationVTF& rhs);

		// Copy assignment operator
		ObservedDataStationVTF& operator=(const ObservedDataStationVTF& rhs);

};

#endif
