//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// SPDX-License-Identifier: MIT
//
// Run-summary helpers for FEMTIC-DABIC screen and log output.
//-------------------------------------------------------------------------------------------------------
#include "FemticDabicRunSummary.h"

#include <cmath>
#include <ctime>
#include <iomanip>
#include <ostream>

#include "CommonParameters.h"
#include "FemticDabicFileNames.h"

namespace {

void outputGermanyLocalTime(
	std::ostream& out,
	const int year,
	const int month,
	const int day,
	const int hour,
	const int minute,
	const int second){

	out << year << "-"
		<< std::setw(2) << std::setfill('0') << month << "-"
		<< std::setw(2) << std::setfill('0') << day << " "
		<< std::setw(2) << std::setfill('0') << hour << ":"
		<< std::setw(2) << std::setfill('0') << minute << ":"
		<< std::setw(2) << std::setfill('0') << second;
}

int germanyHourFromUtcTm(const std::tm* const utcTime){

	int hour = utcTime->tm_hour + 1;
	if( utcTime->tm_mon >= 2 && utcTime->tm_mon <= 8 ){
		hour += 1;
	}
	return hour;
}

void outputDuration(std::ostream& out, const double seconds){

	const int hours = static_cast<int>(seconds) / 3600;
	const int minutes = (static_cast<int>(seconds) % 3600) / 60;
	const int remainingSeconds = static_cast<int>(seconds) % 60;
	out << std::setw(2) << std::setfill('0') << hours << ":"
		<< std::setw(2) << std::setfill('0') << minutes << ":"
		<< std::setw(2) << std::setfill('0') << remainingSeconds;
}

void outputMainInputFiles(std::ostream& out, const int iterationNumInit){

	out << "control.dat, mesh.dat, "
		<< FemticDabicFileNames::observedData() << ", "
		<< FemticDabicFileNames::resistivityBlock(iterationNumInit);
}

void outputMainOutputFiles(std::ostream& out){

	out << FemticDabicFileNames::convergence()
		<< ", " << CommonParameters::programName << "_<rank>.log"
		<< ", " << CommonParameters::programName << "_iter<iter>.cnv"
		<< ", result_<rank>_iter<iter>.csv"
		<< ", " << FemticDabicFileNames::caseFile();
}

}

namespace FemticDabicRunSummary {

void outputStartupLog(std::ostream& out, const StartupSummary& summary){

#ifdef _LINUX
	out << " " << CommonParameters::programName
		<< " (v" << CommonParameters::versionID << ")" << std::endl;
#else
	out << "# Start " << CommonParameters::programName << " Windows Version "
		<< CommonParameters::versionID << " " << summary.elapsedTimeText << std::endl;
#endif
	out << " FEMTIC-DABIC is a 3-D MT inversion code derived from FEMTIC (v4.2)." << std::endl;
	out << " It supports maintained fixed-alpha, ABIC, cubic-spline L-curve, and staged OCCAM" << std::endl;
	out << " trade-off parameter modes for controlling resistivity-model smoothness." << std::endl;
	out << " --------------------------------------------------- " << std::endl;
	out << " License: MIT" << std::endl;
	out << " Copyright (c) 2026 Han Song" << std::endl;
	out << " Modified from Copyright (c) 2021 Yoshiya Usui" << std::endl;
	out << " Developer: Han Song" << std::endl;
	out << " Contributors: Yoshiya Usui, Dieno Diba, Makoto Uyeshima, Peng Yu" << std::endl;
	out << " Contact: han.song@tu-berlin.de | 1831736@tongji.edu.cn" << std::endl;
	out << " --------------------------------------------------- " << std::endl;
	out << " Initializing FEMTIC-DABIC ..." << std::endl;
	out << " Start Time (Local, Germany): ";
	outputGermanyLocalTime(
		out,
		summary.germanYear,
		summary.germanMonth,
		summary.germanDay,
		summary.germanHour,
		summary.germanMin,
		summary.germanSec);
	out << std::endl;
	out << " --------------------------------------------------- " << std::endl;
	out << "  " << std::endl;
}

void outputStartupConsole(std::ostream& out, const StartupSummary& summary){

	out << " " << CommonParameters::programName
		<< " (v" << CommonParameters::versionID << ")" << std::endl;

	out << " 3-D MT inversion with maintained regularization-parameter selection, derived from FEMTIC (v4.2)." << std::endl;
	out << " --------------------------------------------------- " << std::endl;
	out << " License: MIT" << std::endl;
	out << " Copyright (c) 2026 Han Song" << std::endl;
	out << " Modified from Copyright (c) 2021 Yoshiya Usui" << std::endl;
	out << " Developer: Han Song" << std::endl;
	out << " Contributors: Yoshiya Usui, Dieno Diba, Makoto Uyeshima, Peng Yu" << std::endl;
	out << " Contact: han.song@tu-berlin.de | 1831736@tongji.edu.cn" << std::endl;
	out << " --------------------------------------------------- " << std::endl;
	out << " Initializing FEMTIC-DABIC ..." << std::endl;
	out << " Start Time (Local, Germany): ";
	outputGermanyLocalTime(
		out,
		summary.germanYear,
		summary.germanMonth,
		summary.germanDay,
		summary.germanHour,
		summary.germanMin,
		summary.germanSec);
	out << std::endl;
}

void outputRunConfigurationLog(std::ostream& out, const RunConfigurationSummary& summary){

	out << "# FEMTIC-DABIC run configuration summary." << std::endl;
	out << "# Run mode : 3-D MT inversion branch." << std::endl;
	out << "# Iteration range : " << summary.iterationNumInit << " - "
		<< summary.iterationNumMax << "." << std::endl;
	out << "# Inversion method : " << summary.inversionMethodLabel << "." << std::endl;
	out << "# Trade-off parameter mode : " << summary.tradeOffParameterLabel << "." << std::endl;
	out << "# ABIC search mode : " << summary.abicLineSearchLabel << "." << std::endl;
	out << "# Regularization filter : " << summary.regularizationFilterLabel << "." << std::endl;
	out << "# Optimization method : " << summary.inversionUpdateLabel << "." << std::endl;
	out << "# Main input files : ";
	outputMainInputFiles(out, summary.iterationNumInit);
	out << "." << std::endl;
	out << "# Conditional/runtime files : Referencemodel.dat, distortion_iter<iter>.dat, and sensMatFreq* files when their corresponding options are active." << std::endl;
	out << "# Main output files : ";
	outputMainOutputFiles(out);
	out << "." << std::endl;
	out << "# External postprocessing boundary : mesh.dat, Resistivity.iter<iter>, resistivity_block_iter<iter>.dat, result_<rank>_iter<iter>.csv, and VTK/CNV outputs are the files used by Hyssi4D-style mesh, GMT, and plotting workflows." << std::endl;
}

void outputRunConfigurationConsole(std::ostream& out, const RunConfigurationSummary& summary){

	out << " Mode: 3-D MT inversion" << std::endl;
	out << " Iteration range: " << summary.iterationNumInit << " - "
		<< summary.iterationNumMax << std::endl;
	out << " Inversion method: " << summary.inversionMethodLabel << std::endl;
	out << " Trade-off parameter mode: " << summary.tradeOffParameterLabel << std::endl;
	out << " ABIC search mode: " << summary.abicLineSearchLabel << std::endl;
	out << " Regularization filter: " << summary.regularizationFilterLabel << std::endl;
	out << " Optimization method: " << summary.inversionUpdateLabel << std::endl;
	out << " Main input files: ";
	outputMainInputFiles(out, summary.iterationNumInit);
	out << std::endl;
	out << " Main output files: ";
	outputMainOutputFiles(out);
	out << std::endl;
}

void outputFinishConsole(std::ostream& out, const FinishSummary& summary){

	std::tm* berlinEnd = std::gmtime(&summary.endTime);
	const int endHour = germanyHourFromUtcTm(berlinEnd);

	out << std::endl;
	out << " FEMTIC-DABIC run finished. " << std::endl;
	out << " Stop Time (Germany) : ";
	outputGermanyLocalTime(
		out,
		berlinEnd->tm_year + 1900,
		berlinEnd->tm_mon + 1,
		berlinEnd->tm_mday,
		endHour,
		berlinEnd->tm_min,
		berlinEnd->tm_sec);
	out << std::endl;

	const double totalRunTime = std::difftime(summary.endTime, summary.startTime);
	out << " Total Run Time : ";
	outputDuration(out, totalRunTime);
	out << std::endl;

	const double totalCPUTime = totalRunTime * summary.totalPE * summary.numThreads;
	out << " Total CPU Time : ";
	outputDuration(out, totalCPUTime);
	out << std::endl;

	out << " End FEMTIC-DABIC." << std::endl;
	out << " ===================================================" << std::endl;
}

}
