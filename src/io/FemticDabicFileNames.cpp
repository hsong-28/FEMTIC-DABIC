//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// SPDX-License-Identifier: MIT
//
// File-name helpers for FEMTIC-DABIC 3-D input/output contracts.
//-------------------------------------------------------------------------------------------------------
#include "FemticDabicFileNames.h"

#include <sstream>

#include "CommonParameters.h"

namespace {

std::string resultFileName(
	const char* const prefix,
	const int processID,
	const int iterNum,
	const char* const suffix){

	std::ostringstream fileName;
	fileName << prefix << processID << "_iter" << iterNum << suffix;
	return fileName.str();
}

}

namespace FemticDabicFileNames {

std::string observedData(){
	return "observe.dat";
}

std::string resistivityBlock(const int iterNum){
	std::ostringstream fileName;
	fileName << "resistivity_block_iter" << iterNum << ".dat";
	return fileName.str();
}

std::string observedStationVtk(){
	return "obs_loc.vtk";
}

std::string forwardVtk(const int processID, const int iterNum){
	return resultFileName("result_", processID, iterNum, ".vtk");
}

std::string forward2DCsv(const int processID, const int iterNum){
	return resultFileName("result_2DFwd_", processID, iterNum, ".csv");
}

std::string forward3DCsv(const int processID, const int iterNum){
	return resultFileName("result_", processID, iterNum, ".csv");
}

std::string convergence(){
	return std::string(CommonParameters::programName) + ".cnv";
}

std::string iterationConvergence(const int iterNum){
	std::ostringstream fileName;
	fileName << CommonParameters::programName << "_iter" << iterNum << ".cnv";
	return fileName.str();
}

std::string caseFile(){
	return "result.case";
}

std::string logFile(const int processID){
	std::ostringstream fileName;
	fileName << CommonParameters::programName << "_" << processID << ".log";
	return fileName.str();
}

}
