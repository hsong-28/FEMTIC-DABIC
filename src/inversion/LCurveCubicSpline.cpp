//-------------------------------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2026 Han Song
// SPDX-License-Identifier: MIT
//
// Cubic-spline L-curve helper for restored regularization-parameter selection.
//-------------------------------------------------------------------------------------------------------
#include "LCurveCubicSpline.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace {

bool isPositiveFinite(const double value){

	return std::isfinite(value) && value > 0.0;
}

LCurveCubicSpline::Point log10Point(const LCurveCubicSpline::Point& point);

class NaturalCubicSpline1D {

public:
	bool build(const std::vector<double>& xValues, const std::vector<double>& yValues){

		if( xValues.size() != yValues.size() || xValues.size() < 3 ){
			return false;
		}
		for( size_t iValue = 1; iValue < xValues.size(); ++iValue ){
			if( xValues[iValue] <= xValues[iValue - 1] ){
				return false;
			}
		}

		m_xValues = xValues;
		m_yValues = yValues;
		m_secondDerivatives.assign(xValues.size(), 0.0);
		std::vector<double> work(xValues.size(), 0.0);

		for( size_t iValue = 1; iValue + 1 < xValues.size(); ++iValue ){
			const double sig = (xValues[iValue] - xValues[iValue - 1]) /
				(xValues[iValue + 1] - xValues[iValue - 1]);
			const double p = sig * m_secondDerivatives[iValue - 1] + 2.0;
			m_secondDerivatives[iValue] = (sig - 1.0) / p;
			const double slopeForward =
				(yValues[iValue + 1] - yValues[iValue]) /
				(xValues[iValue + 1] - xValues[iValue]);
			const double slopeBackward =
				(yValues[iValue] - yValues[iValue - 1]) /
				(xValues[iValue] - xValues[iValue - 1]);
			work[iValue] =
				(6.0 * (slopeForward - slopeBackward) /
				(xValues[iValue + 1] - xValues[iValue - 1]) -
				sig * work[iValue - 1]) / p;
		}

		for( int iValue = static_cast<int>(xValues.size()) - 2; iValue >= 0; --iValue ){
			m_secondDerivatives[iValue] =
				m_secondDerivatives[iValue] * m_secondDerivatives[iValue + 1] + work[iValue];
		}
		return true;
	}

	double evaluate(const double xValue) const{

		std::vector<double>::const_iterator upper =
			std::upper_bound(m_xValues.begin(), m_xValues.end(), xValue);
		if( upper == m_xValues.begin() ){
			upper = m_xValues.begin() + 1;
		}
		if( upper == m_xValues.end() ){
			upper = m_xValues.end() - 1;
		}
		const size_t upperIndex = static_cast<size_t>(upper - m_xValues.begin());
		const size_t lowerIndex = upperIndex - 1;
		const double h = m_xValues[upperIndex] - m_xValues[lowerIndex];
		const double a = (m_xValues[upperIndex] - xValue) / h;
		const double b = (xValue - m_xValues[lowerIndex]) / h;
		return a * m_yValues[lowerIndex] + b * m_yValues[upperIndex] +
			((a * a * a - a) * m_secondDerivatives[lowerIndex] +
			(b * b * b - b) * m_secondDerivatives[upperIndex]) * h * h / 6.0;
	}

private:
	std::vector<double> m_xValues;
	std::vector<double> m_yValues;
	std::vector<double> m_secondDerivatives;
};

LCurveCubicSpline::Point transformForCurveNorm(
	const LCurveCubicSpline::Point& point,
	const bool useRootNorm){

	if( useRootNorm ){
		return LCurveCubicSpline::Point(
			point.alpha,
			std::sqrt(point.dataMisfit),
			std::sqrt(point.modelRoughness));
	}

	return LCurveCubicSpline::Point(
		point.alpha * point.alpha,
		point.dataMisfit,
		point.modelRoughness);
}

LCurveCubicSpline::Point pointForCurvatureCoordinates(
	const LCurveCubicSpline::Point& point,
	const bool useLogLog,
	const bool useRootNorm){

	const LCurveCubicSpline::Point transformedPoint =
		transformForCurveNorm(point, useRootNorm);
	return useLogLog ? log10Point(transformedPoint) : transformedPoint;
}

LCurveCubicSpline::Point originalPointFromCurvatureCoordinates(
	const LCurveCubicSpline::Point& point,
	const bool useLogLog,
	const bool useRootNorm){

	if( useLogLog ){
		if( useRootNorm ){
			return LCurveCubicSpline::Point(
				std::pow(10.0, point.alpha),
				std::pow(10.0, 2.0 * point.dataMisfit),
				std::pow(10.0, 2.0 * point.modelRoughness));
		}
		return LCurveCubicSpline::Point(
			std::pow(10.0, 0.5 * point.alpha),
			std::pow(10.0, point.dataMisfit),
			std::pow(10.0, point.modelRoughness));
	}

	if( useRootNorm ){
		return LCurveCubicSpline::Point(
			point.alpha,
			point.dataMisfit * point.dataMisfit,
			point.modelRoughness * point.modelRoughness);
	}
	return LCurveCubicSpline::Point(
		std::sqrt(point.alpha),
		point.dataMisfit,
		point.modelRoughness);
}

LCurveCubicSpline::Point log10Point(const LCurveCubicSpline::Point& point){

	return LCurveCubicSpline::Point(
		std::log10(point.alpha),
		std::log10(point.dataMisfit),
		std::log10(point.modelRoughness));
}

LCurveCubicSpline::Point pow10Point(const LCurveCubicSpline::Point& point){

	return LCurveCubicSpline::Point(
		std::pow(10.0, point.alpha),
		std::pow(10.0, point.dataMisfit),
		std::pow(10.0, point.modelRoughness));
}

bool buildChordLengthParameter(
	const std::vector<LCurveCubicSpline::Point>& points,
	std::vector<double>& parameters){

	parameters.assign(points.size(), 0.0);
	double cumulativeLength = 0.0;
	for( size_t iPoint = 1; iPoint < points.size(); ++iPoint ){
		const double dAlpha = points[iPoint].alpha - points[iPoint - 1].alpha;
		const double dMisfit = points[iPoint].dataMisfit - points[iPoint - 1].dataMisfit;
		const double dRoughness = points[iPoint].modelRoughness - points[iPoint - 1].modelRoughness;
		double segmentLength = std::sqrt(
			dAlpha * dAlpha +
			dMisfit * dMisfit +
			dRoughness * dRoughness);
		if( segmentLength <= std::numeric_limits<double>::epsilon() ){
			segmentLength = std::numeric_limits<double>::epsilon();
		}
		cumulativeLength += segmentLength;
		parameters[iPoint] = cumulativeLength;
	}
	if( cumulativeLength <= std::numeric_limits<double>::epsilon() ){
		return false;
	}
	for( size_t iPoint = 1; iPoint < parameters.size(); ++iPoint ){
		parameters[iPoint] /= cumulativeLength;
	}
	return true;
}

void splitSplineCoordinates(
	const std::vector<LCurveCubicSpline::Point>& points,
	std::vector<double>& alphaValues,
	std::vector<double>& misfitValues,
	std::vector<double>& roughnessValues){

	alphaValues.resize(points.size());
	misfitValues.resize(points.size());
	roughnessValues.resize(points.size());
	for( size_t iPoint = 0; iPoint < points.size(); ++iPoint ){
		alphaValues[iPoint] = points[iPoint].alpha;
		misfitValues[iPoint] = points[iPoint].dataMisfit;
		roughnessValues[iPoint] = points[iPoint].modelRoughness;
	}
}

std::string iterationFileName(
	const std::string& prefix,
	const char* const suffix,
	const int iteration){

	std::ostringstream fileName;
	fileName << prefix << suffix << "_iter" << iteration << ".csv";
	return fileName.str();
}

std::string csvFileName(
	const std::string& prefix,
	const char* const suffix){

	std::ostringstream fileName;
	fileName << prefix << suffix << ".csv";
	return fileName.str();
}

bool fileHasContent(const std::string& fileName){

	std::ifstream input(fileName.c_str());
	return input.good() && input.peek() != std::ifstream::traits_type::eof();
}

void writeOriginalData(
	const std::string& prefix,
	const int iteration,
	const std::vector<LCurveCubicSpline::Point>& sortedPoints){

	if( prefix.empty() ){
		return;
	}

	std::ofstream output(iterationFileName(prefix, "_original_data", iteration).c_str());
	output << "alpha,data_misfit,model_roughness\n";
	for( std::vector<LCurveCubicSpline::Point>::const_iterator itr = sortedPoints.begin();
		itr != sortedPoints.end(); ++itr ){
		output << itr->alpha << "," << itr->dataMisfit << "," << itr->modelRoughness << "\n";
	}
}

void writeTrialSummary(
	const LCurveCubicSpline::Options& options,
	const int iteration,
	const std::vector<LCurveCubicSpline::Point>& sortedPoints,
	const std::vector<bool>& usedForSpline,
	const std::vector<std::string>& filterReasons){

	if( options.diagnosticPrefix.empty() ){
		return;
	}

	const std::string fileName = csvFileName(options.diagnosticPrefix, "_trial_summary");
	const bool writeHeader = !fileHasContent(fileName);
	std::ofstream output(fileName.c_str(), std::ios::app);
	if( writeHeader ){
		output
			<< "iteration,trial_index,mode,roughness_operator,alpha,"
			<< "predicted_data_misfit,model_roughness,"
			<< "curve_alpha,curve_data_misfit,curve_model_roughness,"
			<< "used_for_spline,filter_reason\n";
	}
	for( size_t iPoint = 0; iPoint < sortedPoints.size(); ++iPoint ){
		LCurveCubicSpline::Point curvePoint(
			std::numeric_limits<double>::quiet_NaN(),
			std::numeric_limits<double>::quiet_NaN(),
			std::numeric_limits<double>::quiet_NaN());
		if( isPositiveFinite(sortedPoints[iPoint].alpha) &&
			isPositiveFinite(sortedPoints[iPoint].dataMisfit) &&
			isPositiveFinite(sortedPoints[iPoint].modelRoughness) ){
			curvePoint = pointForCurvatureCoordinates(
				sortedPoints[iPoint],
				options.useLogLog,
				options.useRootNorm);
		}
		output
			<< iteration << ","
			<< iPoint << ","
			<< LCurveCubicSpline::modeName(options.useLogLog, options.useRootNorm) << ","
			<< options.roughnessOperator << ","
			<< sortedPoints[iPoint].alpha << ","
			<< sortedPoints[iPoint].dataMisfit << ","
			<< sortedPoints[iPoint].modelRoughness << ","
			<< curvePoint.alpha << ","
			<< curvePoint.dataMisfit << ","
			<< curvePoint.modelRoughness << ","
			<< (usedForSpline[iPoint] ? "yes" : "no") << ","
			<< filterReasons[iPoint] << "\n";
	}
}

void appendFailureIndicator(
	std::string& indicators,
	const std::string& value){

	if( indicators.empty() ){
		indicators = value;
	}else{
		indicators += "|";
		indicators += value;
	}
}

bool isEndpointDominatedSelection(
	const double selectedAlpha,
	const double lowerAlpha,
	const double upperAlpha){

	if( !isPositiveFinite(selectedAlpha) ||
		!isPositiveFinite(lowerAlpha) ||
		!isPositiveFinite(upperAlpha) ||
		upperAlpha <= lowerAlpha ){
		return false;
	}
	const double logLower = std::log10(lowerAlpha);
	const double logUpper = std::log10(upperAlpha);
	const double logSelected = std::log10(selectedAlpha);
	const double width = logUpper - logLower;
	if( width <= std::numeric_limits<double>::epsilon() ){
		return false;
	}
	const double normalizedDistanceToLower = std::fabs(logSelected - logLower) / width;
	const double normalizedDistanceToUpper = std::fabs(logUpper - logSelected) / width;
	return normalizedDistanceToLower < 0.05 || normalizedDistanceToUpper < 0.05;
}

void writeSelectionSummary(
	const LCurveCubicSpline::Options& options,
	const int iteration,
	const double previousDataMisfit,
	const LCurveCubicSpline::Result& result){

	if( options.diagnosticPrefix.empty() ){
		return;
	}

	const std::string fileName = csvFileName(options.diagnosticPrefix, "_selection_summary");
	const bool writeHeader = !fileHasContent(fileName);
	std::ofstream output(fileName.c_str(), std::ios::app);
	if( writeHeader ){
		output
			<< "iteration,mode,roughness_operator,selected_alpha,max_curvature,"
			<< "curvature_contrast,input_points,filtered_points,lower_alpha,upper_alpha,"
			<< "previous_data_misfit,selected_predicted_data_misfit,"
			<< "selected_model_roughness,nonmonotonic_predicted_misfit,"
			<< "endpoint_dominated_selection,weak_curvature_contrast,"
			<< "failure_indicators\n";
	}
	output
		<< iteration << ","
		<< LCurveCubicSpline::modeName(options.useLogLog, options.useRootNorm) << ","
		<< options.roughnessOperator << ","
		<< result.selectedAlpha << ","
		<< result.maxCurvature << ","
		<< result.curvatureContrast << ","
		<< result.inputPointCount << ","
		<< result.filteredPointCount << ","
		<< result.lowerAlpha << ","
		<< result.upperAlpha << ","
		<< previousDataMisfit << ","
		<< result.selectedPredictedDataMisfit << ","
		<< result.selectedModelRoughness << ","
		<< (result.nonMonotonicPredictedMisfit ? "yes" : "no") << ","
		<< (result.endpointDominatedSelection ? "yes" : "no") << ","
		<< (result.weakCurvatureContrast ? "yes" : "no") << ","
		<< result.failureIndicators << "\n";
}

double actualAlphaFromSplineCoordinate(
	const double splineAlpha,
	const bool useLogLog,
	const bool useRootNorm){

	if( useLogLog ){
		return useRootNorm ? std::pow(10.0, splineAlpha) : std::pow(10.0, 0.5 * splineAlpha);
	}
	return useRootNorm ? splineAlpha : std::sqrt(splineAlpha);
}

bool isCandidateInsideLegacyWindow(
	const double candidateAlpha,
	const double lowerAlpha,
	const double upperAlpha){

	return candidateAlpha <= upperAlpha && candidateAlpha >= lowerAlpha;
}

double curvatureFromSamples(
	const LCurveCubicSpline::Point& previous,
	const LCurveCubicSpline::Point& current,
	const LCurveCubicSpline::Point& next){

	const double alphaStepForward = next.alpha - current.alpha;
	const double alphaStepBackward = current.alpha - previous.alpha;
	if( std::fabs(alphaStepForward) <= std::numeric_limits<double>::epsilon() ||
		std::fabs(alphaStepBackward) <= std::numeric_limits<double>::epsilon() ){
		return -std::numeric_limits<double>::infinity();
	}

	const double roughnessPrime = (next.modelRoughness - current.modelRoughness) / alphaStepForward;
	const double misfitPrime = (next.dataMisfit - current.dataMisfit) / alphaStepForward;
	const double roughnessDoublePrime =
		(next.modelRoughness - 2.0 * current.modelRoughness + previous.modelRoughness) /
		alphaStepForward / alphaStepBackward;
	const double misfitDoublePrime =
		(next.dataMisfit - 2.0 * current.dataMisfit + previous.dataMisfit) /
		alphaStepForward / alphaStepBackward;
	const double denominator = std::pow(
		roughnessPrime * roughnessPrime + misfitPrime * misfitPrime,
		1.5);
	if( denominator <= std::numeric_limits<double>::epsilon() ){
		return -std::numeric_limits<double>::infinity();
	}
	return (misfitPrime * roughnessDoublePrime - misfitDoublePrime * roughnessPrime) / denominator;
}

LCurveCubicSpline::Point pointFromSpline(
	const NaturalCubicSpline1D& alphaSpline,
	const NaturalCubicSpline1D& misfitSpline,
	const NaturalCubicSpline1D& roughnessSpline,
	const double t,
	const bool useLogLog){

	const LCurveCubicSpline::Point rawPoint(
		alphaSpline.evaluate(t),
		misfitSpline.evaluate(t),
		roughnessSpline.evaluate(t));
	if( useLogLog ){
		return rawPoint;
	}
	return pow10Point(rawPoint);
}

bool passesLegacyMisfitGate(
	const LCurveCubicSpline::Point& point,
	const double previousDataMisfit,
	const bool useLogLog,
	const bool useRootNorm){

	if( useLogLog ){
		const double threshold = useRootNorm ?
			0.5 * std::log10(previousDataMisfit) :
			std::log10(previousDataMisfit);
		return point.dataMisfit < threshold;
	}

	const double threshold = useRootNorm ?
		previousDataMisfit * previousDataMisfit :
		previousDataMisfit;
	return point.dataMisfit < threshold;
}

}

LCurveCubicSpline::Point::Point():
	alpha(0.0),
	dataMisfit(0.0),
	modelRoughness(0.0)
{
}

LCurveCubicSpline::Point::Point(
	const double alphaIn,
	const double dataMisfitIn,
	const double modelRoughnessIn):
	alpha(alphaIn),
	dataMisfit(dataMisfitIn),
	modelRoughness(modelRoughnessIn)
{
}

LCurveCubicSpline::Options::Options():
	useLogLog(false),
	useRootNorm(false),
	interpolationSamples(1000),
	diagnosticPrefix(""),
	roughnessOperator("")
{
}

LCurveCubicSpline::Result::Result():
	selectedAlpha(-1.0),
	selectedPredictedDataMisfit(-1.0),
	selectedModelRoughness(-1.0),
	maxCurvature(-std::numeric_limits<double>::infinity()),
	curvatureContrast(-1.0),
	lowerAlpha(-1.0),
	upperAlpha(-1.0),
	inputPointCount(0),
	filteredPointCount(0),
	nonMonotonicPredictedMisfit(false),
	endpointDominatedSelection(false),
	weakCurvatureContrast(false),
	selected(false)
{
}

std::string LCurveCubicSpline::modeName(
	const bool useLogLog,
	const bool useRootNorm){

	if( useLogLog ){
		return useRootNorm ? "LOG_ROOT" : "LOG_SQUARED";
	}
	return useRootNorm ? "LINEAR_ROOT" : "LINEAR_SQUARED";
}

LCurveCubicSpline::Result LCurveCubicSpline::selectAlpha(
	const std::vector<Point>& points,
	const double previousDataMisfit,
	const int iteration,
	const Options& options) const{

	Result result;
	result.inputPointCount = static_cast<int>(points.size());

	std::vector<Point> sortedPoints(points);
	std::sort(sortedPoints.begin(), sortedPoints.end(),
		[](const Point& lhs, const Point& rhs){
			return lhs.alpha > rhs.alpha;
		});
	writeOriginalData(options.diagnosticPrefix, iteration, sortedPoints);

	if( points.size() < 4 ){
		appendFailureIndicator(result.failureIndicators, "INSUFFICIENT_POINTS");
		writeSelectionSummary(options, iteration, previousDataMisfit, result);
		return result;
	}
	if( options.interpolationSamples < 3 ){
		appendFailureIndicator(result.failureIndicators, "INSUFFICIENT_INTERPOLATION_SAMPLES");
		writeSelectionSummary(options, iteration, previousDataMisfit, result);
		return result;
	}
	if( !isPositiveFinite(previousDataMisfit) ){
		appendFailureIndicator(result.failureIndicators, "NONPOSITIVE_PREVIOUS_DATA_MISFIT");
		writeSelectionSummary(options, iteration, previousDataMisfit, result);
		return result;
	}

	std::vector<Point> filteredPoints;
	std::vector<bool> usedForSpline(sortedPoints.size(), false);
	std::vector<std::string> filterReasons(sortedPoints.size(), "used");
	bool stopAfterNonMonotonicMisfit = false;
	for( size_t iPoint = 0; iPoint + 1 < sortedPoints.size(); ++iPoint ){
		const Point& point = sortedPoints[iPoint];
		if( !stopAfterNonMonotonicMisfit &&
			isPositiveFinite(point.alpha) &&
			isPositiveFinite(point.dataMisfit) &&
			isPositiveFinite(point.modelRoughness) ){
			filteredPoints.push_back(transformForCurveNorm(point, options.useRootNorm));
			usedForSpline[iPoint] = true;
		}else if( stopAfterNonMonotonicMisfit ){
			filterReasons[iPoint] = "after_nonmonotonic_predicted_misfit";
		}else{
			filterReasons[iPoint] = "nonpositive_or_nonfinite_value";
		}
		if( sortedPoints[iPoint].alpha > sortedPoints[iPoint + 1].alpha &&
			sortedPoints[iPoint].dataMisfit < sortedPoints[iPoint + 1].dataMisfit ){
			stopAfterNonMonotonicMisfit = true;
			result.nonMonotonicPredictedMisfit = true;
		}
	}
	if( !stopAfterNonMonotonicMisfit && !sortedPoints.empty() ){
		const Point& point = sortedPoints.back();
		if( isPositiveFinite(point.alpha) &&
			isPositiveFinite(point.dataMisfit) &&
			isPositiveFinite(point.modelRoughness) ){
			filteredPoints.push_back(transformForCurveNorm(point, options.useRootNorm));
			usedForSpline[sortedPoints.size() - 1] = true;
		}else{
			filterReasons[sortedPoints.size() - 1] = "nonpositive_or_nonfinite_value";
		}
	}else if( !sortedPoints.empty() ){
		filterReasons[sortedPoints.size() - 1] = "after_nonmonotonic_predicted_misfit";
	}
	writeTrialSummary(options, iteration, sortedPoints, usedForSpline, filterReasons);
	result.filteredPointCount = static_cast<int>(filteredPoints.size());
	if( result.nonMonotonicPredictedMisfit ){
		appendFailureIndicator(result.failureIndicators, "NONMONOTONIC_MISFIT");
	}
	if( filteredPoints.size() < 4 ){
		appendFailureIndicator(result.failureIndicators, "INSUFFICIENT_POINTS");
		writeSelectionSummary(options, iteration, previousDataMisfit, result);
		return result;
	}

	std::vector<Point> splinePoints(filteredPoints);
	for( size_t iPoint = 0; iPoint < splinePoints.size(); ++iPoint ){
		if( !isPositiveFinite(splinePoints[iPoint].alpha) ||
			!isPositiveFinite(splinePoints[iPoint].dataMisfit) ||
			!isPositiveFinite(splinePoints[iPoint].modelRoughness) ){
			appendFailureIndicator(result.failureIndicators, "NONPOSITIVE_CURVE_COORDINATE");
			writeSelectionSummary(options, iteration, previousDataMisfit, result);
			return result;
		}
		splinePoints[iPoint] = log10Point(splinePoints[iPoint]);
	}

	std::vector<double> splineParameter;
	if( !buildChordLengthParameter(splinePoints, splineParameter) ){
		appendFailureIndicator(result.failureIndicators, "DEGENERATE_SPLINE_PARAMETER");
		writeSelectionSummary(options, iteration, previousDataMisfit, result);
		return result;
	}
	std::vector<double> alphaValues;
	std::vector<double> misfitValues;
	std::vector<double> roughnessValues;
	splitSplineCoordinates(splinePoints, alphaValues, misfitValues, roughnessValues);
	NaturalCubicSpline1D alphaSpline;
	NaturalCubicSpline1D misfitSpline;
	NaturalCubicSpline1D roughnessSpline;
	if( !alphaSpline.build(splineParameter, alphaValues) ||
		!misfitSpline.build(splineParameter, misfitValues) ||
		!roughnessSpline.build(splineParameter, roughnessValues) ){
		appendFailureIndicator(result.failureIndicators, "SPLINE_BUILD_FAILED");
		writeSelectionSummary(options, iteration, previousDataMisfit, result);
		return result;
	}

	result.upperAlpha = sortedPoints[1].alpha;
	result.lowerAlpha = sortedPoints[sortedPoints.size() - 2].alpha;
	double secondBestCurvature = -std::numeric_limits<double>::infinity();

	std::ofstream curvatureOutput;
	if( !options.diagnosticPrefix.empty() ){
		curvatureOutput.open(
			iterationFileName(options.diagnosticPrefix, "_interpolated_curvature", iteration).c_str());
		curvatureOutput
			<< "iteration,sample_index,mode,roughness_operator,"
			<< "curve_alpha,curve_data_misfit,curve_model_roughness,"
			<< "curvature,candidate_alpha,inside_search_window,passes_misfit_gate\n";
	}

	for( int iSample = 1; iSample < options.interpolationSamples - 1; ++iSample ){
		const double tPrevious = static_cast<double>(iSample - 1) /
			static_cast<double>(options.interpolationSamples - 1);
		const double tCurrent = static_cast<double>(iSample) /
			static_cast<double>(options.interpolationSamples - 1);
		const double tNext = static_cast<double>(iSample + 1) /
			static_cast<double>(options.interpolationSamples - 1);

		const Point previousPoint = pointFromSpline(
			alphaSpline,
			misfitSpline,
			roughnessSpline,
			tPrevious,
			options.useLogLog);
		const Point currentPoint = pointFromSpline(
			alphaSpline,
			misfitSpline,
			roughnessSpline,
			tCurrent,
			options.useLogLog);
		const Point nextPoint = pointFromSpline(
			alphaSpline,
			misfitSpline,
			roughnessSpline,
			tNext,
			options.useLogLog);
		const double curvature = curvatureFromSamples(previousPoint, currentPoint, nextPoint);
		const double candidateAlpha = actualAlphaFromSplineCoordinate(
			currentPoint.alpha,
			options.useLogLog,
			options.useRootNorm);
		const bool insideSearchWindow =
			isCandidateInsideLegacyWindow(candidateAlpha, result.lowerAlpha, result.upperAlpha);
		const bool passesMisfitGate =
			passesLegacyMisfitGate(
				currentPoint,
				previousDataMisfit,
				options.useLogLog,
				options.useRootNorm);

		if( curvatureOutput.is_open() ){
			curvatureOutput
				<< iteration << ","
				<< iSample << ","
				<< LCurveCubicSpline::modeName(options.useLogLog, options.useRootNorm) << ","
				<< options.roughnessOperator << ","
				<< currentPoint.alpha << ","
				<< currentPoint.dataMisfit << ","
				<< currentPoint.modelRoughness << ","
				<< curvature << ","
				<< candidateAlpha << ","
				<< (insideSearchWindow ? "yes" : "no") << ","
				<< (passesMisfitGate ? "yes" : "no") << "\n";
		}

		if( std::isfinite(curvature) &&
			insideSearchWindow &&
			passesMisfitGate ){
			if( curvature > result.maxCurvature ){
				secondBestCurvature = result.maxCurvature;
			}else if( curvature > secondBestCurvature ){
				secondBestCurvature = curvature;
			}
		}

		if( std::isfinite(curvature) &&
			curvature > result.maxCurvature &&
			insideSearchWindow &&
			passesMisfitGate ){
			const Point originalCandidate = originalPointFromCurvatureCoordinates(
				currentPoint,
				options.useLogLog,
				options.useRootNorm);
			result.maxCurvature = curvature;
			result.selectedAlpha = candidateAlpha;
			result.selectedPredictedDataMisfit = originalCandidate.dataMisfit;
			result.selectedModelRoughness = originalCandidate.modelRoughness;
			result.selected = true;
		}
	}

	if( result.selected ){
		result.endpointDominatedSelection = isEndpointDominatedSelection(
			result.selectedAlpha,
			result.lowerAlpha,
			result.upperAlpha);
		if( result.endpointDominatedSelection ){
			appendFailureIndicator(result.failureIndicators, "ENDPOINT_DOMINATED");
		}
		if( std::isfinite(secondBestCurvature) ){
			result.curvatureContrast = result.maxCurvature - secondBestCurvature;
			if( std::fabs(result.curvatureContrast) <= std::fabs(result.maxCurvature) * 0.05 ){
				result.weakCurvatureContrast = true;
				appendFailureIndicator(result.failureIndicators, "LOW_CURVATURE_CONTRAST");
			}
		}
	}else{
		appendFailureIndicator(result.failureIndicators, "NO_SELECTED_ALPHA");
	}
	writeSelectionSummary(options, iteration, previousDataMisfit, result);
	return result;
}
