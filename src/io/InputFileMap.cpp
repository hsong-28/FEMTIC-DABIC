//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Volker Rath (DIAS)
// SPDX-License-Identifier: MIT
//
// New file added by Volker Rath (DIAS) with the help of Claude Sonnet 5, 2026-08-05.
//
// Optional runtime remapping of FEMTIC-DABIC's hard-coded input file names.
// See InputFileMap.h for the file format and semantics.
//
// Modified (added findOnDisk(): case-insensitive on-disk file name matching, wired into
// resolve()) by Volker Rath (DIAS) with the help of Claude Sonnet 5, 2026-09-27.
//-------------------------------------------------------------------------------------------------------
#include "InputFileMap.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <string>

#ifdef _LINUX
#include <dirent.h>
#endif

namespace {

const char* const kInputFileMapFileName = "inputfiles.dat";

std::string trim(const std::string& s) {
	const std::string::size_type first = s.find_first_not_of(" \t\r\n");
	if (first == std::string::npos) {
		return std::string();
	}
	const std::string::size_type last = s.find_last_not_of(" \t\r\n");
	return s.substr(first, last - first + 1);
}

std::string toLower(const std::string& s) {
	std::string out(s);
	std::transform(out.begin(), out.end(), out.begin(),
		[](unsigned char c){ return static_cast<char>(std::tolower(c)); });
	return out;
}

// Lazily loads and caches "inputfiles.dat" (if present) on first use.
class Map {
public:
	static Map& getInstance() {
		static Map instance;
		return instance;
	}

	bool lookup(const std::string& key, std::string& value) {
		load();
		std::map<std::string, std::string>::const_iterator it = m_entries.find(toLower(key));
		if (it == m_entries.end()) {
			return false;
		}
		value = it->second;
		return true;
	}

private:
	Map() : m_loaded(false) {}

	void load() {
		if (m_loaded) {
			return;
		}
		m_loaded = true;

		std::ifstream inFile(kInputFileMapFileName, std::ios::in);
		if (!inFile) {
			return;
		}

		std::string line;
		while (std::getline(inFile, line)) {
			// Drop everything from the first '#' onward (comment-to-end-of-line),
			// and skip lines that are entirely comments/blank.
			const std::string::size_type hashPos = line.find('#');
			if (hashPos != std::string::npos) {
				line = line.substr(0, hashPos);
			}
			line = trim(line);
			if (line.empty()) {
				continue;
			}

			const std::string::size_type eqPos = line.find('=');
			if (eqPos == std::string::npos) {
				continue;
			}

			const std::string key = toLower(trim(line.substr(0, eqPos)));
			const std::string value = trim(line.substr(eqPos + 1));
			if (key.empty() || value.empty()) {
				continue;
			}

			m_entries[key] = value;
		}
	}

	bool m_loaded;
	std::map<std::string, std::string> m_entries;
};

std::string findOnDiskImpl(const std::string& requestedName) {

	// Fast path: the requested name already exists exactly as given, which
	// covers every normally-cased working directory and is the only outcome
	// possible on case-insensitive file systems.
	{
		std::ifstream exact(requestedName.c_str(), std::ios::in);
		if (exact.good()) {
			return requestedName;
		}
	}

#ifdef _LINUX
	// Split off a leading directory component, if any; requestedName is
	// almost always a bare file name in the working directory, but this
	// keeps the scan correct either way.
	std::string dirPart = ".";
	std::string basePart = requestedName;
	const std::string::size_type slashPos = requestedName.find_last_of('/');
	if (slashPos != std::string::npos) {
		dirPart = requestedName.substr(0, slashPos);
		if (dirPart.empty()) {
			dirPart = "/";
		}
		basePart = requestedName.substr(slashPos + 1);
	}

	const std::string basePartLower = toLower(basePart);

	DIR* const dir = opendir(dirPart.c_str());
	if (dir != NULL) {
		std::string found;
		const struct dirent* entry;
		while ((entry = readdir(dir)) != NULL) {
			const std::string entryName(entry->d_name);
			if (entryName == basePart) {
				// Exact match turned up in the listing even though the
				// direct open above failed (e.g. a permissions issue) --
				// don't mask that with a different, misleading name.
				found.clear();
				break;
			}
			if (found.empty() && toLower(entryName) == basePartLower) {
				found = entryName;
			}
		}
		closedir(dir);
		if (!found.empty()) {
			return (slashPos != std::string::npos) ? (dirPart + "/" + found) : found;
		}
	}
#endif // _LINUX

	return requestedName;
}

}

namespace InputFileMap {

std::string resolve(const std::string& key, const std::string& defaultName) {
	std::string value;
	if (Map::getInstance().lookup(key, value)) {
		return findOnDiskImpl(value);
	}
	return findOnDiskImpl(defaultName);
}

std::string findOnDisk(const std::string& requestedName) {
	return findOnDiskImpl(requestedName);
}

}
