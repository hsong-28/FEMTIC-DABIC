#ifndef DBLDEF_DATA_FIT_COOLING_POLICY
#define DBLDEF_DATA_FIT_COOLING_POLICY

#include <algorithm>
#include <cmath>

namespace DataFitCoolingPolicy {

inline bool shouldReverseInitialBracketSearch(
	const double previousRms,
	const double trialRms)
{
	return std::isfinite(previousRms) && std::isfinite(trialRms) &&
		trialRms >= previousRms;
}

inline double relativeRmsDecrease(const double previousRms, const double currentRms)
{
	if (!std::isfinite(previousRms) || !std::isfinite(currentRms) || previousRms <= 0.0)
	{
		return 0.0;
	}
	return (previousRms - currentRms) / previousRms;
}

inline bool isRmsImproved(const double previousRms, const double currentRms)
{
	return std::isfinite(previousRms) && std::isfinite(currentRms) &&
		previousRms > 0.0 && currentRms < previousRms;
}

inline bool isMeaningfulRmsDecrease(
	const double previousRms,
	const double currentRms,
	const double relativeTolerance)
{
	return previousRms > currentRms &&
		relativeRmsDecrease(previousRms, currentRms) >= relativeTolerance;
}

inline bool shouldCool(
	const double previousRms,
	const double currentRms,
	const double decreaseThreshold)
{
	const double decrease = relativeRmsDecrease(previousRms, currentRms);
	return previousRms > currentRms && decrease > 0.0 && decrease < decreaseThreshold;
}

inline double nextHigherBracketLog10Alpha(
	const double currentLog10Alpha,
	const double log10Span)
{
	return currentLog10Alpha + log10Span;
}

inline double nextLowerBracketLog10Alpha(
	const double currentLog10Alpha,
	const double minimumLog10Alpha,
	const double log10Span)
{
	if (currentLog10Alpha <= minimumLog10Alpha)
	{
		return minimumLog10Alpha;
	}
	return std::max(minimumLog10Alpha, currentLog10Alpha - log10Span);
}

inline double cooledAlpha(
	const double currentAlpha,
	const double coolingFactor,
	const double minimumAlpha)
{
	return std::max(minimumAlpha, coolingFactor * currentAlpha);
}

}

#endif
