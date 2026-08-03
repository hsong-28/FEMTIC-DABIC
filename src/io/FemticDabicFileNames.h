//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// SPDX-License-Identifier: MIT
//
// File-name helpers for FEMTIC-DABIC 3-D input/output contracts.
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_FEMTIC_DABIC_FILE_NAMES
#define DBLDEF_FEMTIC_DABIC_FILE_NAMES

#include <string>

namespace FemticDabicFileNames {

std::string observedData();
std::string resistivityBlock(const int iterNum);
std::string observedStationVtk();
std::string forwardVtk(const int processID, const int iterNum);
std::string forward2DCsv(const int processID, const int iterNum);
std::string forward3DCsv(const int processID, const int iterNum);
std::string convergence();
std::string iterationConvergence(const int iterNum);
std::string caseFile();
std::string logFile(const int processID);

}

#endif
