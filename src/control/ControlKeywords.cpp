//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// SPDX-License-Identifier: MIT
//
// Helpers for control.dat keyword parsing.
//-------------------------------------------------------------------------------------------------------
#include "ControlKeywords.h"

#include <cstdlib>
#include <ostream>
#include <string>

#include "OutputFiles.h"

namespace ControlKeywords {

void resetReadFlags(bool* hasAlreadyRead, const int numParams){

	for( int iParam = 0; iParam < numParams; ++iParam ){
		hasAlreadyRead[iParam] = false;
	}
}

void ensureNotAlreadyRead(const bool* hasAlreadyRead, const int paramID, const char* keyword){

	if( hasAlreadyRead[paramID] ){
		OutputFiles::m_logFile << "Error : Already read the data from control.dat !! : "
			<< keyword << std::endl;
		std::exit(1);
	}
}

bool isEndKeywordLine(const std::string& line){

	return line.substr(0, 3).compare("END") == 0;
}

}
