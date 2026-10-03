//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Volker Rath (DIAS)
// SPDX-License-Identifier: MIT
//
// Optional runtime remapping of FEMTIC-DABIC's hard-coded input file names.
//
// If a file named "inputfiles.dat" is present in the working directory, it may contain
// "key = value" assignments (one per line) that redirect the hard-coded input file names
// FEMTIC reads at startup, e.g.:
//
//     control = my_control_file.dat
//     mesh    = mesh_v2.dat
//     observe = observe_2024.dat
//     initial = resistivity_block_iter000_restart.dat
//
// Any key that is not listed, or the absence of "inputfiles.dat" altogether, falls back to
// the previous hard-coded default file name unchanged, preserving full backward compatibility.
// Lines whose first non-whitespace character is '#' are treated as comments and ignored, as
// is any text following a '#' later in the line. Keys are matched case-insensitively.
//
// resolve() uses findOnDisk() for case-insensitive matching of the resolved name.
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_INPUT_FILE_MAP
#define DBLDEF_INPUT_FILE_MAP

#include <string>

namespace InputFileMap {

// Returns the file name to use for the given logical key: the value from "inputfiles.dat"
// if that file exists and defines the key, otherwise defaultName unchanged -- in both cases
// passed through findOnDisk() so the on-disk case actually present is used.
std::string resolve(const std::string& key, const std::string& defaultName);

// Returns an existing file name in the current working directory that matches "requestedName"
// case-insensitively. If a file named exactly requestedName can be opened, it is returned
// unchanged (fast path -- the common case, and the only case on non-Linux builds). Otherwise
// the working directory is scanned once for a case-insensitive match; if found, that file's
// actual on-disk spelling is returned. If no match (exact or case-insensitive) exists at all,
// requestedName is returned unchanged, so the caller's normal "file not found" handling still
// reports the originally intended name.
std::string findOnDisk(const std::string& requestedName);

}

#endif
