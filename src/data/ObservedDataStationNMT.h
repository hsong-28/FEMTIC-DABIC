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
#ifndef DBLDEF_OBSERVED_DATA_NMT
#define DBLDEF_OBSERVED_DATA_NMT

#include <vector>
#include <complex>

#include "ObservedDataStation.h"
#include "ObservedDataStationPoint.h"
#include "Forward3D.h"
#include "CommonParameters.h"
#include "MeshDataTetraElement.h"
#include "FemticHDF5CalcTypes.h"

// Observed data of NMT station ( line )
class ObservedDataStationNMT: public ObservedDataStation{
	public:
		// Constructer
		explicit ObservedDataStationNMT();

		// Destructer
		~ObservedDataStationNMT();
			
		// Read data from input file
		void inputObservedData( std::ifstream& inFile );

		// Find elements including dipole
		void findElementsIncludingDipole();

		// Calulate difference of voltages
		void calculateVoltageDifferences( const Forward3D* const ptrForward3D, const int rhsVectorIDOfVoltageDifference );

		// Calulate Network-MT response
		void calculateNetworkMTResponse( const double freq, const ObservedDataStationPoint* const ptrStationOfMagneticField, int& icount );

		// Initialize difference of voltages
		void initializeVoltageDifferences( const int iPol );

		// Initialize Network-MT responses and errors
		void initializeNetworkMTResponsesAndErrors();

		// Allocate memory for the calculated values of Network-MT responses and errors
		void allocateMemoryForCalculatedValues();

		// Output calculated values of Network-MT responses
		void outputCalculatedValues() const;
			
		// Calulate interpolator vector of voltage difference
		void calcInterpolatorVectorOfVoltageDifference( Forward3D* const ptrForward3D );

		// Calulate sensitivity matrix of Network-MT response
		void calculateSensitivityMatrix( const double freq, const int nModel,
			const ObservedDataStationPoint* const ptrStationOfMagneticField,
			const std::complex<double>* const derivativesOfEMFieldExPol,
			const std::complex<double>* const derivativesOfEMFieldEyPol,
			double* const sensitivityMatrix ) const;
	
		// Calculate data vector of this PE
		void calculateResidualVectorOfDataThisPE( const double freq, const int offset, double* vector ) const;

		// Calulate sum of square of misfit
		double calculateErrorSumOfSquaresThisPE() const;

		// Get location of the station
		const CommonParameters::locationDipole& getLocationOfStation() const;

		// Get Z coordinate of the point
		double getZCoordOfPoint( const int num ) const;

#ifdef _HDF5_JAC
		// Collect data-error (SD) vector in same slot order as residual vector
		// (ported from femtic_v4_src, 2026-08-21).
		void collectErrorVectorThisPE( const double freq, const int offset, double* vector ) const;

#endif // _HDF5_JAC

#ifdef _HDF5_OUT
		// --- HDF5 output accessors (ported from femtic_v4_src, 2026-08-21) ---
		// Note: getLocationOfStation() already declared above (non-inline).
		std::complex<double> getYxObserved(const int i) const { return m_YxObserved[i]; }
		std::complex<double> getYyObserved(const int i) const { return m_YyObserved[i]; }
		double getYxSDRe(const int i) const { return m_YxSD[i].realPart; }
		double getYxSDIm(const int i) const { return m_YxSD[i].imagPart; }
		double getYySDRe(const int i) const { return m_YySD[i].realPart; }
		double getYySDIm(const int i) const { return m_YySD[i].imagPart; }

		// Collect this PE's calculated NMT response values for
		// results_iterN.h5. Component order 0=Yx 1=Yy, matching
		// OutputHDF5.cpp's /data row layout.
		// Added by Volker Rath (DIAS) with the help of Claude Sonnet 5
		// (Anthropic), 2026-09-13.
		void collectCalculatedValuesForHDF5( std::vector<FemticHDF5CalcRow>& rows ) const;

#endif // _HDF5_OUT

	private:
		// Location of the station
		CommonParameters::locationDipole m_location;

		std::complex<double>* m_YxObserved;
		std::complex<double>* m_YyObserved;

		CommonParameters::DoubleComplexValues* m_YxSD;
		CommonParameters::DoubleComplexValues* m_YySD;

		std::complex<double>* m_YxCalculated;
		std::complex<double>* m_YyCalculated;

		CommonParameters::DoubleComplexValues* m_YxResidual;
		CommonParameters::DoubleComplexValues* m_YyResidual;

		std::complex<double> m_voltageCalculated[2];

		//int m_columnNumberOfVoltageDifferenceInRhsMatrix;
		int m_rhsVectorIDOfVoltageDifference;

		//int* m_dataIDOfYx;
		//int* m_dataIDOfYy;
		CommonParameters::InitComplexValues* m_dataIDOfYx;
		CommonParameters::InitComplexValues* m_dataIDOfYy;

		// Total number of elemenst including dipole
		int m_numElementsIncludingDipole;

		// Elements including dipole
		int* m_elementsIncludingDipole;

		// Faces including dipole
		int* m_facesIncludingDipole;

		// local coordinate values of the location of start point of line segment for hexahedral mesh
		CommonParameters::locationXY* m_localCoordinateValuesStartPoint;

		// local coordinate values of the location of end point of line segment for hexahedral mesh
		CommonParameters::locationXY* m_localCoordinateValuesEndPoint;
			
		//// Volume coordinate values of the location of start point of line segment for tetrahedral mesh
		//CommonParameters::VolumeCoords* m_volumeCoordinateValuesStartPoint;

		//// Volume coordinate values of the location of start point of line segment for tetrahedral mesh
		//CommonParameters::VolumeCoords* m_volumeCoordinateValuesEndPoint;

		// Area coordinate values of the location of start point of line segment for tetrahedral mesh
		CommonParameters::AreaCoords* m_areaCoordinateValuesStartPoint;

		// Area coordinate values of the location of start point of line segment for tetrahedral mesh
		CommonParameters::AreaCoords* m_areaCoordinateValuesEndPoint;

		// Copy constructer
		ObservedDataStationNMT(const ObservedDataStationNMT& rhs);

		// Copy assignment operator
		ObservedDataStationNMT& operator=(const ObservedDataStationNMT& rhs);

};
#endif
