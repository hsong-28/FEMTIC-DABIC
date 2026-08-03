#ifndef DBLDEF_OBSERVED_DATA_SD_MASKING_POLICY
#define DBLDEF_OBSERVED_DATA_SD_MASKING_POLICY

namespace ObservedDataSDMaskingPolicy {

inline bool isActive(const double standardDeviation)
{
	return standardDeviation > 0.0;
}

inline double safeStandardDeviation(const double standardDeviation)
{
	return isActive(standardDeviation) ? standardDeviation : 1.0;
}

class GuardedSensitivityMatrix {
public:
	explicit GuardedSensitivityMatrix(double* const values)
		: m_values(values),
		  m_inactiveValue(0.0)
	{
	}

	double& operator[](const long long index)
	{
		if (index < 0)
		{
			m_inactiveValue = 0.0;
			return m_inactiveValue;
		}
		return m_values[index];
	}

private:
	double* const m_values;
	double m_inactiveValue;
};

inline int assignDataID(const double standardDeviation, int& count)
{
	if (!isActive(standardDeviation))
	{
		return -1;
	}
	return count++;
}

inline double normalizedResidual(
	const double observed,
	const double calculated,
	const double standardDeviation)
{
	return isActive(standardDeviation)
		? (observed - calculated) / standardDeviation
		: 0.0;
}

inline void writeResidual(
	const int dataID,
	const int offset,
	const double value,
	double* const vector)
{
	if (dataID >= 0)
	{
		vector[offset + dataID] = value;
	}
}

inline void writeSensitivity(
	const int dataID,
	const int numberOfModelParameters,
	const int modelParameterID,
	const double value,
	double* const sensitivityMatrix)
{
	if (dataID >= 0)
	{
		sensitivityMatrix[
			static_cast<long long>(numberOfModelParameters) *
			static_cast<long long>(dataID) + modelParameterID] = value;
	}
}

}

#endif
