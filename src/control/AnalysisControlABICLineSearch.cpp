/* -------------------------------------------------------------------------------------------------------
 * FEMTIC-DABIC ABIC line-search member definitions split from AnalysisControl.cpp.
 * This file is branch-specific and must preserve the existing trial-state side effects.
 * ------------------------------------------------------------------------------------------------------- */
#include "AnalysisControl.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "ObservedData.h"
#include "OutputFiles.h"
#include "ResistivityBlock.h"

double AnalysisControl::getInitialABICLog10BracketSpan() const
{
	if (!usesInexactABICSearch())
	{
		return 0.25;
	}
	return m_iterationNumCurrent == m_iterationNumInit ? 0.5 : 0.2;
}

bool AnalysisControl::usesInexactABICSearch() const
{
	return m_abicSearchMode == ABIC_SEARCH_INEXACT;
}

bool AnalysisControl::shouldUseInexactABICBracketOnly(const int) const
{
	return usesInexactABICSearch();
}

bool AnalysisControl::shouldReuseInexactABICAlphaOnCutback(const int cutbackCount) const
{
	return usesInexactABICSearch() && cutbackCount > 0;
}

void AnalysisControl::runReducedStepTrialWithCurrentABICAlpha()
{
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData *const ptrObservedData = ObservedData::getInstance();
	m_ptrInversion->inversionCalculation();
	m_abic = m_ptrInversion->getabic();
	ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
	ptrObservedData->copyDistortionParamsCurToPWK1();
	ptrObservedData->cacheSelectedTrialForwardResponse();
}

// subroutine of ABIC inversion
void AnalysisControl::minbrkABIC()
{
	/*	minbrkABIC brackets a univariate minimum of a function.
		To be used prior to a univariate minimisation routine.
		Modified so that the model associated with the misfits and ABIC is carried around
		for use in the minimisation routines and possibly ultimately kept as the result of this iteration.
		This subroutine is modified based on a subroutine in the OCCAM 3.0 Package.
	References:
		[1] Myer et al., 2007 : OCCAM 3.0 release notes.
	*/
	double gold = sqrt(1.618034);
	int myProcessID = getMyPE();
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData *const ptrObservedData = ObservedData::getInstance();
	m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterABICB);
	m_ptrInversion->inversionCalculation();
	ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
	ptrObservedData->copyDistortionParamsCurToPWK1();
	ptrObservedData->cacheSelectedTrialForwardResponse();

	m_ABICB = m_ptrInversion->getabic();
	m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterABICA);
	m_ptrInversion->inversionCalculation();
	m_ABICA = m_ptrInversion->getabic();

	if (m_ABICB[0] > m_ABICA[0])
	{
		double tem = m_tradeOffParameterABICA;
		m_tradeOffParameterABICA = m_tradeOffParameterABICB;
		m_tradeOffParameterABICB = tem;
		std::vector<double> temVec = m_ABICA;
		m_ABICA = m_ABICB;
		m_ABICB = temVec;
		ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
		ptrObservedData->copyDistortionParamsCurToPWK1();
		ptrObservedData->cacheSelectedTrialForwardResponse();

	} // keep m_ABICB < m_ABICA;
	m_tradeOffParameterABICC = m_tradeOffParameterABICB + gold * (m_tradeOffParameterABICB - m_tradeOffParameterABICA);
	m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterABICC);
	m_ptrInversion->inversionCalculation();
	m_ABICC = m_ptrInversion->getabic();

	// If m_ABICC > m_ABICB && m_ABICA > m_ABICB, the univariate minimum is already bracketed.
	// But, if m_ABICC < m_ABICB, we still need some effort here.
	while (m_ABICB[0] > m_ABICC[0])
	{
		ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
		ptrObservedData->copyDistortionParamsCurToPWK1();
		ptrObservedData->cacheSelectedTrialForwardResponse();
		double R = (m_tradeOffParameterABICB - m_tradeOffParameterABICA) * (m_ABICB[0] - m_ABICC[0]);
		double Q = (m_tradeOffParameterABICB - m_tradeOffParameterABICC) * (m_ABICB[0] - m_ABICA[0]);
		double BmA = m_tradeOffParameterABICB - m_tradeOffParameterABICA;
		double BmC = m_tradeOffParameterABICB - m_tradeOffParameterABICC;
		double U = m_tradeOffParameterABICB - (BmC * Q - BmA * R) / (2. * sign(std::max(std::fabs(Q - R), 1.E-32), Q - R));
		double Ulim = m_tradeOffParameterABICB + 100. * (-1.0 * BmC);
		// double m_ABICU(0.0);
		std::vector<double> m_ABICU({0.0, 0.0});
		if ((m_tradeOffParameterABICB - U) * (U - m_tradeOffParameterABICC) > 0)
		{
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			m_ABICU = m_ptrInversion->getabic();
			if (m_ABICU[0] < m_ABICC[0])
			{
				// Fu < Fc <= Fb <= Fa
				// Make: A=B & B=U (want Fb < Fc & Fb <= Fa)
				m_tradeOffParameterABICA = m_tradeOffParameterABICB;
				m_ABICA = m_ABICB;
				m_tradeOffParameterABICB = U;
				m_ABICB = m_ABICU;
				ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
				ptrObservedData->copyDistortionParamsCurToPWK1();
				ptrObservedData->cacheSelectedTrialForwardResponse();
				return;
			}
			else if (m_ABICU[0] > m_ABICB[0])
			{
				m_tradeOffParameterABICC = U;
				m_ABICC = m_ABICU;
				return;
			}
			U = m_tradeOffParameterABICC + gold * (m_tradeOffParameterABICC - m_tradeOffParameterABICB);
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			m_ABICU = m_ptrInversion->getabic();
		}
		else if ((m_tradeOffParameterABICC - U) * (U - Ulim) > 0)
		{
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			m_ABICU = m_ptrInversion->getabic();
			if (m_ABICU[0] < m_ABICC[0])
			{
				m_tradeOffParameterABICB = m_tradeOffParameterABICC;
				m_tradeOffParameterABICC = U;
				U = m_tradeOffParameterABICC + gold * (m_tradeOffParameterABICC - m_tradeOffParameterABICB);
				m_ABICB = m_ABICC;
				m_ABICC = m_ABICU;
				ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
				ptrObservedData->copyDistortionParamsCurToPWK1();
				ptrObservedData->cacheSelectedTrialForwardResponse();
				m_tradeOffParameterForResistivityValue = pow(10.0, U);
				m_ptrInversion->inversionCalculation();
				m_ABICU = m_ptrInversion->getabic();
			}
		}
		else if ((U - Ulim) * (Ulim - m_tradeOffParameterABICC) > 0)
		{
			U = Ulim;
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			m_ABICU = m_ptrInversion->getabic();
		}
		else
		{
			U = m_tradeOffParameterABICC + gold * (m_tradeOffParameterABICC - m_tradeOffParameterABICB);
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			m_ABICU = m_ptrInversion->getabic();
		}
		m_tradeOffParameterABICA = m_tradeOffParameterABICB;
		m_tradeOffParameterABICB = m_tradeOffParameterABICC;
		m_tradeOffParameterABICC = U;
		m_ABICA = m_ABICB;
		m_ABICB = m_ABICC;
		m_ABICC = m_ABICU;
	};
}

// subroutine of ABIC inversion
double AnalysisControl::sign(double val, double ref)
{
	return (ref >= 0) ? fabs(val) : -fabs(val);
}

double AnalysisControl::frootABIC()
{
	/* FROOT FINDS THE POINT AT WHICH A UNIVARIATE FUNCTION ATTAINS A GIVEN VALUE (m_tolreq)
References:
	[1] Myer et al., 2007: OCCAM 3.0 release notes.
	*/
	// Resistivity block instance
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData *const ptrObservedData = ObservedData::getInstance();
	const int ITMAX = 100;	  // Maximum iterations
	const double EPS = 3.E-8; // Convergence threshold
	const double tol = 0.1;	  // Tolerance for convergence
	std::vector<double> fa({0.0, 0.0});
	std::vector<double> fb({0.0, 0.0});
	std::vector<double> fc({0.0, 0.0});

	double aa = m_stepsizelb; // Lower bound
	double b = m_stepsizeub;  // Upper bound
	fa[0] = m_ABIClb[0];
	fb[0] = m_ABICub[0];
	fa[1] = m_ABIClb[1] - m_tolreq; // Function value at lower bound
	fb[1] = m_ABICub[1] - m_tolreq; // Function value at upper bound
	fc = fb;						// Function value at midpoint
	double c = 0.0, dd = 0.0, e = 0.0;

	int myProcessID = getMyPE();
	if (fa[1] * fb[1] > 0.0)
	{
		if (myProcessID == 0)
		{
			std::cout << "ROOT NOT BRACKETED IN FROOT" << std::endl;
		}
		return b; // Return upper bound as fallback
	}

	// Copy initial models
	ptrResistivityBlock->copyPWK2NotFixedToPWK3();
	ptrObservedData->copyDistortionParamsPWK2ToPWK3();

	for (int iter = 0; iter < ITMAX; ++iter)
	{
		// Ensure b and c are on opposite sides of the root
		if (fb[1] * fc[1] > 0.0)
		{ // if1
			c = aa;
			fc = fa;
			dd = b - aa;
			e = dd;
			ptrResistivityBlock->copyPWK1NotFixedToPWK3();
			ptrObservedData->copyDistortionParamsPWK1ToPWK3();
		}

		if (std::fabs(fc[1]) < std::fabs(fb[1]))
		{ // if2
			// Rotate a, b, c values; ensure fb is close to the target;
			aa = b;
			b = c;
			c = aa;
			fa = fb;
			fb = fc;
			fc = fa;
			ptrResistivityBlock->copyPWK2NotFixedToPWK1();
			ptrObservedData->copyDistortionParamsPWK2ToPWK1();
			ptrResistivityBlock->copyPWK3NotFixedToPWK2();
			ptrObservedData->copyDistortionParamsPWK3ToPWK2();
			ptrResistivityBlock->copyPWK1NotFixedToPWK3();
			ptrObservedData->copyDistortionParamsPWK1ToPWK3();
		}

		double tol1 = 2.0 * EPS * std::fabs(b) + 0.5 * tol;
		double xm = 0.5 * (c - b);

		// Check for convergence
		if (std::fabs(xm) <= tol1 || std::fabs(fb[1]) < 0.001 * m_tolreq)
		{
			m_abic = fb;
			ptrObservedData->cacheSelectedTrialForwardResponse();
			return b; // Found root
		}

		// Attempt inverse quadratic interpolation
		if (std::fabs(e) >= tol1 && std::fabs(fa[1]) > std::fabs(fb[1]))
		{
			double s = fb[1] / fa[1];
			double p, q;

			if (aa == c)
			{
				p = 2.0 * xm * s;
				q = 1.0 - s;
			}
			else
			{
				double q_prev = fa[1] / fc[1];
				double r = fb[1] / fc[1];
				p = s * (2.0 * xm * q_prev * (q_prev - r) - (b - aa) * (r - 1.0));
				q = (q_prev - 1.0) * (r - 1.0) * (s - 1.0);
			}

			if (p > 0.0)
				q = -q;
			p = std::fabs(p);

			if (2.0 * p < std::min(3.0 * xm * q - std::fabs(tol1 * q), std::fabs(e * q)))
			{
				e = dd;
				dd = p / q;
			}
			else
			{
				dd = xm;
				e = dd;
			}
		}
		else
		{
			// Bisection step
			dd = xm;
			e = dd;
		}

		// Update a, b, fa, fb
		aa = b;
		fa = fb;
		ptrResistivityBlock->copyPWK2NotFixedToPWK1();
		ptrObservedData->copyDistortionParamsPWK2ToPWK1();
		if (std::fabs(dd) > tol1)
		{
			b += dd;
		}
		else
		{
			b += (xm > 0.0 ? tol1 : -tol1);
		}

		m_stepLengthDampingFactorCur = b;
		m_ptrInversion->inversionCalculation();
		fb = m_ptrInversion->getabic();
		fb[1] = fb[1] - m_tolreq; // Recalculate function value at new b
		ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
		ptrObservedData->copyDistortionParamsCurToPWK2();
	}

	// Maximum iterations exceeded
	if (myProcessID == 0)
	{
		std::cerr << "MAXIMUM ITERATIONS EXCEEDED IN FROOT" << std::endl;
	}
	m_abic = fb;
	ptrObservedData->cacheSelectedTrialForwardResponse();
	return b; // Return best estimate
}

// subroutine of ABIC inversion
std::vector<double> AnalysisControl::fminbrentABIC(const int icut)
{
	/*	fminbrent returns the minimum value of a function within a specified interval.
		This function implements Brent's method for function minimization, using
		parabolic interpolation and golden section search.
		This subroutine is based on Brent's (1973) minimizing method
		and modified based on a subroutine in the OCCAM 3.0 Package.
	References:
		[1] Brent, R. P., 1973. Chapter 4: An Algorithm with Guaranteed Convergence for Finding a Zero of a Function,
	Algorithms for Minimization without Derivatives, Englewood Cliffs, NJ: Prentice-Hall, ISBN 0-13-022335-2
		[2] Myer et al., 2007: OCCAM 3.0 release notes.
		*/
	// Constants
	const int ITMAX = 100;			// Maximum number of iterations
	const double CGOLD = 0.3819660; // Golden ratio constant
	const double ZEPS = 1.0E-10;	// Small number to prevent division by zero
	const double tol = 0.2;			// Tolerance for convergence

	// Interval bounds and initial values
	double lowerBound = std::min(m_tradeOffParameterABICC, m_tradeOffParameterABICA);
	double upperBound = std::max(m_tradeOffParameterABICC, m_tradeOffParameterABICA);
	double x = m_tradeOffParameterABICB; // Initial guess
	double w = x, v = x;				 // Secondary points
	double e = 0.0;						 // Distance moved in the last step
	std::vector<double> fx = m_ABICB;
	std::vector<double> fw = fx;
	std::vector<double> fv = fx;

	// Iteration variables
	double midpoint = 0.0, tol1 = 0.0, tol2 = 0.0;
	double r = 0.0, q = 0.0, p = 0.0, etemp = 0.0, step = 0.0, u = 0.0;
	std::vector<double> fu({0.0, 0.0});

	// Resistivity block instance
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData *const ptrObservedData = ObservedData::getInstance();

	for (int iter = 0; iter < ITMAX; ++iter)
	{
		// Calculate midpoint and tolerances
		midpoint = 0.5 * (lowerBound + upperBound);
		tol1 = tol * std::abs(x) + ZEPS;
		tol2 = 2.0 * tol1;

		// Check convergence
		// both the interval is small and x is close to the center
		// abs(x - midpoint) + 0.5 * (upperBound - lowerBound) <= tol2
		if (std::abs(x - midpoint) <= (tol2 - 0.5 * (upperBound - lowerBound)) || shouldUseInexactABICBracketOnly(icut))
		{
			m_tradeOffParameterForResistivityValue = pow(10.0, x);
			return fx; // Minimum found
					   //
		}

		// Attempt parabolic interpolation
		if (std::abs(e) > tol1)
		{
			r = (x - w) * (fx[0] - fv[0]);
			q = (x - v) * (fx[0] - fw[0]);
			p = (x - v) * q - (x - w) * r;
			q = 2.0 * (q - r);
			if (q > 0.0)
				p = -p; // Ensure correct direction
			q = std::abs(q);

			etemp = e;
			e = step;

			if (std::abs(p) < std::abs(0.5 * q * etemp) &&
				p > q * (lowerBound - x) &&
				p < q * (upperBound - x))
			{
				step = p / q;
				u = x + step;

				// Ensure step size is larger than the tolerance
				if ((u - lowerBound) < tol2 || (upperBound - u) < tol2)
				{
					step = std::copysign(tol1, midpoint - x);
				}
			}
			else
			{
				// Fall back to golden section search
				e = (x >= midpoint) ? (lowerBound - x) : (upperBound - x);
				step = CGOLD * e;
			}
		}
		else
		{
			// Golden section search
			e = (x >= midpoint) ? (lowerBound - x) : (upperBound - x);
			step = CGOLD * e;
		}

		// Calculate new trial point
		u = (std::abs(step) >= tol1) ? (x + step) : (x + std::copysign(tol1, step));
		m_tradeOffParameterForResistivityValue = pow(10.0, u);
		m_ptrInversion->inversionCalculation();
		fu = m_ptrInversion->getabic();
		// double fu = func(u); // Evaluate the function at the new point

		// Update the interval and points based on the function value
		if (fu[0] <= fx[0])
		{
			if (u >= x)
				lowerBound = x;
			else
				upperBound = x;

			v = w;
			fv = fw;
			w = x;
			fw = fx;
			x = u;
			fx = fu;
			ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
			ptrObservedData->copyDistortionParamsCurToPWK1();
			ptrObservedData->cacheSelectedTrialForwardResponse();
		}
		else
		{
			if (u < x)
				lowerBound = u;
			else
				upperBound = u;

			if (fu[0] <= fw[0] || w == x)
			{
				v = w;
				fv = fw;
				w = u;
				fw = fu;
			}
			else if (fu[0] <= fv[0] || v == x || v == w)
			{
				v = u;
				fv = fu;
			}
		}
	}

	// If maximum iterations are reached, warn the user
	std::cerr << "Warning: Maximum iterations exceeded in fminbrentABIC" << std::endl;
	m_tradeOffParameterForResistivityValue = pow(10.0, x);
	return fx; // Return the best found value
}
