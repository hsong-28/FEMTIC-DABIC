//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// SPDX-License-Identifier: MIT
//
// Run-summary helpers for FEMTIC-DABIC screen and log output.
//-------------------------------------------------------------------------------------------------------
#ifndef DBLDEF_FEMTIC_DABIC_RUN_SUMMARY
#define DBLDEF_FEMTIC_DABIC_RUN_SUMMARY

#include <ctime>
#include <iosfwd>
#include <string>

namespace FemticDabicRunSummary {

struct StartupSummary {
	std::string elapsedTimeText;
	int germanYear;
	int germanMonth;
	int germanDay;
	int germanHour;
	int germanMin;
	int germanSec;
};

struct RunConfigurationSummary {
	int iterationNumInit;
	int iterationNumMax;
	std::string inversionMethodLabel;
	std::string tradeOffParameterLabel;
	std::string abicLineSearchLabel;
	std::string regularizationFilterLabel;
	std::string inversionUpdateLabel;
};

struct FinishSummary {
	std::time_t startTime;
	std::time_t endTime;
	int totalPE;
	int numThreads;
};

void outputStartupLog(std::ostream& out, const StartupSummary& summary);
void outputStartupConsole(std::ostream& out, const StartupSummary& summary);
void outputRunConfigurationLog(std::ostream& out, const RunConfigurationSummary& summary);
void outputRunConfigurationConsole(std::ostream& out, const RunConfigurationSummary& summary);
void outputFinishConsole(std::ostream& out, const FinishSummary& summary);

}

#endif
