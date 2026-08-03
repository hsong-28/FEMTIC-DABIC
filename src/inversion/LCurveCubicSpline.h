//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// SPDX-License-Identifier: MIT
//
// Cubic-spline L-curve helper for restored regularization-parameter selection.
//-------------------------------------------------------------------------------------------------------
#ifndef LCURVE_CUBIC_SPLINE_H
#define LCURVE_CUBIC_SPLINE_H

#include <string>
#include <vector>

class LCurveCubicSpline {

public:
	struct Point {
		Point();
		Point(const double alphaIn, const double dataMisfitIn, const double modelRoughnessIn);
		double alpha;
		double dataMisfit;
		double modelRoughness;
	};

	struct Options {
		Options();
		bool useLogLog;
		bool useRootNorm;
		int interpolationSamples;
		std::string diagnosticPrefix;
		std::string roughnessOperator;
	};

	struct Result {
		Result();
		double selectedAlpha;
		double selectedPredictedDataMisfit;
		double selectedModelRoughness;
		double maxCurvature;
		double curvatureContrast;
		double lowerAlpha;
		double upperAlpha;
		int inputPointCount;
		int filteredPointCount;
		bool nonMonotonicPredictedMisfit;
		bool endpointDominatedSelection;
		bool weakCurvatureContrast;
		bool selected;
		std::string failureIndicators;
	};

	static std::string modeName(const bool useLogLog, const bool useRootNorm);

	Result selectAlpha(
		const std::vector<Point>& points,
		const double previousDataMisfit,
		const int iteration,
		const Options& options) const;
};

#endif
