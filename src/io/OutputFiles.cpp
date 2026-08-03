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
#include <iostream>
#include <sstream>
#include <iomanip>
#include "mpi.h"

#include "AnalysisControl.h"
#include "OutputFiles.h"
#include "CommonParameters.h"
#include "FemticDabicFileNames.h"
#include "MeshDataBrickElement.h"
#include "ObservedData.h"

std::ofstream OutputFiles::m_logFile;
std::ofstream OutputFiles::m_vtkFile;
std::ofstream OutputFiles::m_vtkFileForObservedStation;
// std::ofstream OutputFiles::m_csvFile;
// std::ofstream OutputFiles::m_csvFileFor2DFwd;
FILE *OutputFiles::m_csvFile;
FILE *OutputFiles::m_csvFileFor2DFwd;
std::ofstream OutputFiles::m_cnvFile;
std::ofstream OutputFiles::m_cnvFileitr;

// Return the the instance of the class
OutputFiles *OutputFiles::getInstance()
{
	static OutputFiles instance; // The only instance
	return &instance;
}

// Close csv file in which the results of 3D forward calculation is written
void OutputFiles::openVTKFile(const int iterNum)
{

	if (m_vtkFile.is_open())
	{
		m_vtkFile.close();
	}

	int myPE(0);

	// Get process ID and total process number
	MPI_Comm_rank(MPI_COMM_WORLD, &myPE);

	// Open VTK file
	const std::string vtkFileName = FemticDabicFileNames::forwardVtk(myPE, iterNum);
	m_vtkFile.open(vtkFileName.c_str(), std::ios::out);
	if (m_vtkFile.fail())
	{
		m_logFile << "File open error !! : " << vtkFileName << std::endl;
		exit(1);
	}
	m_vtkFile << "# vtk DataFile Version 2.0" << std::endl;
	m_vtkFile << "Iter" << iterNum << "_PE" << myPE << std::endl;
	m_vtkFile << "ASCII" << std::endl;

	// Output mesh data to VTK file
	//(  MeshDataBrickElement::getInstance() )->outputMeshDataToVTK();
	((AnalysisControl::getInstance())->getPointerOfMeshData())->outputMeshDataToVTK();
}

// Open VTK file for ploting observed station
void OutputFiles::openVTKFileForObservedStation()
{

	if (m_vtkFileForObservedStation.is_open())
	{
		m_vtkFileForObservedStation.close();
	}

	int myPE(0);

	// Get process ID and total process number
	MPI_Comm_rank(MPI_COMM_WORLD, &myPE);

	if (myPE == 0)
	{ // If this PE number is zero

		// Open VTK file for ploting observed station
		const std::string vtkFileName = FemticDabicFileNames::observedStationVtk();
		m_vtkFileForObservedStation.open(vtkFileName.c_str(), std::ios::out);
		if (m_vtkFileForObservedStation.fail())
		{
			m_logFile << "File open error !! : " << vtkFileName << std::endl;
			exit(1);
		}
		m_vtkFileForObservedStation << "# vtk DataFile Version 2.0" << std::endl;
		m_vtkFileForObservedStation << "Locations of observed stations" << std::endl;
		m_vtkFileForObservedStation << "ASCII" << std::endl;
	}
}

// Open csv file in which the results of 2D forward calculation is written
void OutputFiles::openCsvFileFor2DFwd(const int iterNum)
{

	// if( m_csvFileFor2DFwd.is_open() ){
	//	m_csvFileFor2DFwd.close();
	// }

	int myPE(0);

	// Get process ID and total process number
	MPI_Comm_rank(MPI_COMM_WORLD, &myPE);

	// Open csv file in which the results of 2D forward calculation is written
	const std::string fileName = FemticDabicFileNames::forward2DCsv(myPE, iterNum);
	// m_csvFileFor2DFwd.open( fileName.str().c_str(), std::ios::out );
	// if( m_csvFileFor2DFwd.fail() ){
	//	m_logFile << "File open error !! : " << fileName.str() << std::endl;
	//	exit(1);
	// }

	if ((m_csvFileFor2DFwd = fopen(fileName.c_str(), "w")) == NULL)
	{
		m_logFile << "File open error !! : " << fileName << std::endl;
		exit(1);
	}
}

// Open csv file in which the results of 3D forward calculation is written
void OutputFiles::openCsvFileFor3DFwd(const int iterNum)
{

	if (m_csvFile != NULL)
	{
		fclose(m_csvFile);
	}

	int myPE(0);

	// Get process ID and total process number
	MPI_Comm_rank(MPI_COMM_WORLD, &myPE);

	// Open csv file in which the results of 3D forward calculation is written
	const std::string filename = FemticDabicFileNames::forward3DCsv(myPE, iterNum);
	// m_csvFile.open( filename.str().c_str(), std::ios::out );
	// if( m_csvFile.fail() )
	//{
	//	OutputFiles::m_logFile << "File open error !!" << filename.str() <<std::endl;
	//	exit(1);
	// }

	if ((m_csvFile = fopen(filename.c_str(), "w")) == NULL)
	{
		m_logFile << "File open error !! : " << filename << std::endl;
		exit(1);
	}
}

// Open cnv file
void OutputFiles::openCnvFile(const int iterNum)
{

	if (m_cnvFile.is_open())
	{
		m_cnvFile.close();
	}

	const std::string filename = FemticDabicFileNames::convergence();
	m_cnvFile.open(filename.c_str(), std::ios::app);

	// if(joint)
	const std::string filenameitr = FemticDabicFileNames::iterationConvergence(iterNum);
	m_cnvFileitr.open(filenameitr.c_str(), std::ios::out);
	// end

	if (m_cnvFile.fail())
	{
		OutputFiles::m_logFile << "File open error !!" << filename << std::endl;
		exit(1);
	}

	bool runCrossGradient = (AnalysisControl::getInstance())->runCG();

	if (iterNum == 0)
	{
		if ((AnalysisControl::getInstance())->ABICinversion())
		{
			if (runCrossGradient)
			{
				OutputFiles::m_cnvFileitr << std::setw(15) << "Retrial#"
										  << std::setw(15) << "Alpha"
										  << std::setw(15) << "LmdCG"
										  << std::setw(15) << "obj_func"
										  << std::setw(15) << "dataMisfit"
										  << std::setw(15) << "modelroughness"
										  << std::setw(15) << "modelnorm"
										  << std::setw(15) << "CrossGra"
										  << std::setw(15) << "rms"
										  << std::setw(15) << "rms_obj"
										  << std::setw(15) << "abic"
										  << std::endl;

				if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta"
										   << std::setw(15) << "LmdCG"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Distortion"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "CrossGra"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ABIC"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
				{

					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta1"
										   << std::setw(15) << "Beta2"
										   << std::setw(15) << "LmdCG"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Gain"
										   << std::setw(15) << "Rotation"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "CrossGra"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ABIC"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta"
										   << std::setw(15) << "LmdCG"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Gain"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "CrossGra"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ABIC"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "LmdCG"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "CrossGra"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ABIC"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
			}
			else
			{
				OutputFiles::m_cnvFileitr << std::setw(15) << "Retrial#"
										  << std::setw(15) << "Alpha"
										  << std::setw(15) << "obj_func"
										  << std::setw(15) << "dataMisfit"
										  << std::setw(15) << "modelroughness"
										  << std::setw(15) << "modelnorm"
										  << std::setw(15) << "rms"
										  << std::setw(15) << "rms_obj"
										  << std::setw(15) << "m_updated"
										  << std::setw(15) << "abic"
										  << std::endl;

				if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Distortion"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ABIC"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
				{

					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta1"
										   << std::setw(15) << "Beta2"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Gain"
										   << std::setw(15) << "Rotation"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ABIC"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Gain"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ABIC"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ABIC"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
			}
		}
		else
		{
			if (runCrossGradient)
			{
				if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "LmdCG"
										   << std::setw(15) << "Beta"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Distortion"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "CrossGra"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
				{

					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta1"
										   << std::setw(15) << "Beta2"
										   << std::setw(15) << "LmdCG"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Gain"
										   << std::setw(15) << "Rotation"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "CrossGra"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta"
										   << std::setw(15) << "LmdCG"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Gain"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "CrossGra"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "LmdCG"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "CrossGra"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
			}
			else
			{
				if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_DISTORTION_MATRIX_DIFFERENCE)
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Distortion"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_AND_ROTATIONS)
				{

					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta1"
										   << std::setw(15) << "Beta2"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Gain"
										   << std::setw(15) << "Rotation"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else if ((AnalysisControl::getInstance())->getTypeOfDistortion() == AnalysisControl::ESTIMATE_GAINS_ONLY)
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Beta"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Gain"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
				else
				{
					OutputFiles::m_cnvFile << std::setw(10) << "Iter#" << std::setw(10) << "Retrial#"
										   << std::setw(15) << "Alpha"
										   << std::setw(15) << "Damp"
										   << std::setw(15) << "Roughness"
										   << std::setw(15) << "Misfit"
										   << std::setw(15) << "RMS"
										   << std::setw(15) << "MupdateMean"
										   << std::setw(15) << "ObjFunc"
										   << std::endl;
				}
			}
		}
	}
}

// Open cnv file
void OutputFiles::openCnvFile_iter(const int iterNum)
{

	// if(joint)
	if (m_cnvFileitr.is_open())
	{
		m_cnvFileitr.close();
	}

	m_cnvFileitr << std::left;
	const std::string filenameitr = FemticDabicFileNames::iterationConvergence(iterNum);
	m_cnvFileitr.open(filenameitr.c_str(), std::ios::out);

	OutputFiles::m_cnvFileitr << std::setw(15) << "Retrial#"
							  << std::setw(15) << "Alpha"
							  << std::setw(15) << "obj_func"
							  << std::setw(15) << "dataMisfit"
							  << std::setw(15) << "roughness"
							  << std::setw(15) << "modelnorm"
							  << std::setw(15) << "rms"
							  << std::setw(15) << "rms_obj"
							  << std::setw(15) << "m_updated"
							  << std::setw(15) << "abic"
							  << std::endl;
}

// Output case file
void OutputFiles::outputCaseFile() const
{

	std::ofstream ofs(FemticDabicFileNames::caseFile().c_str());

	ofs << "FORMAT" << std::endl;
	ofs << "type:	ensight gold" << std::endl;
	ofs << std::endl;

	ofs << "GEOMETRY" << std::endl;
	ofs << "model:	Mesh.geo" << std::endl;
	ofs << std::endl;

	const AnalysisControl *const ptrAnalysisControl = AnalysisControl::getInstance();
	const ObservedData *const ptrObservedData = ObservedData::getInstance();

	ofs << "VARIABLE" << std::endl;
	ofs << "scalar per element:	BlockIDs	BlockIDs" << std::endl;

	if (ptrAnalysisControl->doesOutputToVTK(AnalysisControl::OUTPUT_RESISTIVITY_VALUES_TO_VTK))
	{
		ofs << "scalar per element:	1	Resistivity[Ohm-m]	Resistivity.iter*" << std::endl;
	}

	if (ptrAnalysisControl->doesOutputToVTK(AnalysisControl::OUTPUT_ELECTRIC_FIELD_VECTORS_TO_VTK))
	{
		const int nFreq = ptrObservedData->getTotalNumberOfDifferenetFrequencies();
		for (int ifreq = 0; ifreq < nFreq; ++ifreq)
		{
			ofs << "vector per element:	1	ReE@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](ExPol)	ReE_Freq" << ifreq << "_ExPol.iter*" << std::endl;
			ofs << "vector per element:	1	ImE@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](ExPol)	ImE_Freq" << ifreq << "_ExPol.iter*" << std::endl;
			ofs << "vector per element:	1	ReE@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](EyPol)	ReE_Freq" << ifreq << "_EyPol.iter*" << std::endl;
			ofs << "vector per element:	1	ImE@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](EyPol)	ImE_Freq" << ifreq << "_EyPol.iter*" << std::endl;
		}
	}

	if (ptrAnalysisControl->doesOutputToVTK(AnalysisControl::OUTPUT_MAGNETIC_FIELD_VECTORS_TO_VTK))
	{
		const int nFreq = ptrObservedData->getTotalNumberOfDifferenetFrequencies();
		for (int ifreq = 0; ifreq < nFreq; ++ifreq)
		{
			ofs << "vector per element:	1	ReH@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](ExPol)	ReH_Freq" << ifreq << "_ExPol.iter*" << std::endl;
			ofs << "vector per element:	1	ImH@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](ExPol)	ImH_Freq" << ifreq << "_ExPol.iter*" << std::endl;
			ofs << "vector per element:	1	ReH@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](EyPol)	ReH_Freq" << ifreq << "_EyPol.iter*" << std::endl;
			ofs << "vector per element:	1	ImH@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](EyPol)	ImH_Freq" << ifreq << "_EyPol.iter*" << std::endl;
		}
	}

	if (ptrAnalysisControl->doesOutputToVTK(AnalysisControl::OUTPUT_CURRENT_DENSITY))
	{
		const int nFreq = ptrObservedData->getTotalNumberOfDifferenetFrequencies();
		for (int ifreq = 0; ifreq < nFreq; ++ifreq)
		{
			ofs << "vector per element:	1	Rej@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](ExPol)	Rej_Freq" << ifreq << "_ExPol.iter*" << std::endl;
			ofs << "vector per element:	1	Imj@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](ExPol)	Imj_Freq" << ifreq << "_ExPol.iter*" << std::endl;
			ofs << "vector per element:	1	Rej@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](EyPol)	Rej_Freq" << ifreq << "_EyPol.iter*" << std::endl;
			ofs << "vector per element:	1	Imj@Freq" << ptrObservedData->getFrequencyValue(ifreq) << "[Hz](EyPol)	Imj_Freq" << ifreq << "_EyPol.iter*" << std::endl;
		}
	}

	if (ptrAnalysisControl->doesOutputToVTK(AnalysisControl::OUTPUT_SENSITIVITY))
	{
		ofs << "scalar per element:	1	Sensitiivty	Sensitivity.iter*" << std::endl;
		ofs << "scalar per element:	1	NormalizedSensitiivty	NormalizedSensitivity.iter*" << std::endl;
	}

	if (ptrAnalysisControl->doesOutputToVTK(AnalysisControl::OUTPUT_SENSITIVITY_DENSITY))
	{
		ofs << "scalar per element:	1	SensitiivtyDensity	SensitivityDensity.iter*" << std::endl;
		ofs << "scalar per element:	1	NormalizedSensitiivtyDensity	NormalizedSensitivityDensity.iter*" << std::endl;
	}

	ofs << std::endl;

	ofs << "TIME" << std::endl;
	ofs << "time set:	1" << std::endl;

	const int iterStart = ptrAnalysisControl->getIterationNumInit();
	const int numIter = ptrAnalysisControl->getIterationNumMax() - iterStart + 1;
	ofs << "number of steps:	" << numIter << std::endl;
	ofs << "filename start number:	" << iterStart << std::endl;
	ofs << "filename increment:	" << 1 << std::endl;
	ofs << "time values:" << std::endl;
	for (int iter = 0; iter < numIter; ++iter)
	{
		ofs << iter + iterStart << std::endl;
	}

	ofs.close();
}
// Constructer
OutputFiles::OutputFiles()
{

	int myPE(0);

	// Get process ID and total process number
	MPI_Comm_rank(MPI_COMM_WORLD, &myPE);

	// Open log file
	const std::string logFileName = FemticDabicFileNames::logFile(myPE);
	m_logFile.open(logFileName.c_str(), std::ios::out);
	if (m_logFile.fail())
	{
		std::cerr << "File open error !! : " << logFileName << std::endl;
		exit(1);
	}

	//// Open csv file
	// std::ostringstream csvFileName;
	// csvFileName << CommonParameters::programName << "." << myPE << ".csv";
	// m_csvFile.open( csvFileName.str().c_str(), std::ios::out );
	// if( m_csvFile.fail() )
	//{
	//	OutputFiles::m_logFile << "File open error !!" << csvFileName.str() <<std::endl;
	//	exit(1);
	// }

	// #ifdef _OUTPUT_2D_RESULT
	//	// Open csv file for result of 2D Forward analysis
	//	std::ostringstream csvFileFor2DFwdName;
	//	csvFileFor2DFwdName << "VFEMTINV3D_2DFwd." << myPE << ".csv";
	//	m_csvFileFor2DFwd.open( csvFileFor2DFwdName.str().c_str(), std::ios::out );
	//	if( m_csvFileFor2DFwd.fail() )
	//	{
	//		std::cerr << "File open error !!" << csvFileFor2DFwdName.str() <<std::endl;
	//		exit(1);
	//	}
	// #endif
}

// Destructer
OutputFiles::~OutputFiles()
{

	if (m_logFile.is_open())
	{
		m_logFile.close();
	}

	if (m_vtkFile.is_open())
	{
		m_vtkFile.close();
	}

	// if( m_csvFile.is_open() ){
	//	m_csvFile.close();
	// }

	// if( m_csvFileFor2DFwd.is_open() ){
	//	m_csvFileFor2DFwd.close();
	// }
}

// Copy constructer
OutputFiles::OutputFiles(const OutputFiles &rhs)
{
	std::cerr << "Error : Copy constructer of the class LogFile is not implemented." << std::endl;
	exit(1);
}

// Copy assignment operator
OutputFiles &OutputFiles::operator=(const OutputFiles &rhs)
{
	std::cerr << "Error : Copy assignment operator of the class LogFile is not implemented." << std::endl;
	exit(1);
}
