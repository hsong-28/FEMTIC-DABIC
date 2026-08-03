//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// SPDX-License-Identifier: MIT
//
// Helpers for control.dat keyword parsing.
//-------------------------------------------------------------------------------------------------------
#ifndef CONTROL_KEYWORDS_H
#define CONTROL_KEYWORDS_H

#include <string>

namespace ControlKeywords {

void resetReadFlags(bool* hasAlreadyRead, const int numParams);
void ensureNotAlreadyRead(const bool* hasAlreadyRead, const int paramID, const char* keyword);
bool isEndKeywordLine(const std::string& line);

}

#endif
