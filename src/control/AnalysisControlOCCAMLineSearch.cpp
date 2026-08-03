/* -------------------------------------------------------------------------------------------------------
 * FEMTIC-DABIC OCCAM line-search member definitions split from AnalysisControl.cpp.
 * This file restores the legacy v1.0 OCCAM search helpers and the minimal
 * scheduler helper used by the Difference and Laplacian regularization branches.
 * ------------------------------------------------------------------------------------------------------- */
#include "AnalysisControl.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "ObservedData.h"
#include "OutputFiles.h"
#include "ResistivityBlock.h"

void AnalysisControl::runOCCAMLineSearch(const char* regularizationLabel)
{
	const int myProcessID = getMyPE();
	ObservedData *const ptrObservedData = ObservedData::getInstance();
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();

	if (myProcessID == 0)
	{
		std::cout << " # Entering OCCAM (" << regularizationLabel << ")." << std::endl;
		if (m_inexactMinimizationOfOCCAM)
		{
			std::cout << " # OCCAM line search mode: inexact Phase-I RMS minimization." << std::endl;
		}
		else
		{
			std::cout << " # OCCAM line search mode: exact Brent/legacy." << std::endl;
		}
	}
	const int numDataThisPE = ptrObservedData->getNumObservedDataThisPETotal();
	OutputFiles::m_logFile << "# Number of data of this PE : " << numDataThisPE << std::endl;
	if (m_residualVectorThisPE != NULL)
	{
		delete[] m_residualVectorThisPE;
		m_residualVectorThisPE = NULL;
	}
	m_residualVectorThisPE = new double[numDataThisPE];
	ptrObservedData->calculateResidualVectorOfDataThisPE(m_residualVectorThisPE);
	m_tradeOffParameterForResistivityValuePre = m_tradeOffParameterForResistivityValue;

	if (!m_OCCAMsmoothing)
	{
		if (myProcessID == 0)
		{
			std::cout << " # OCCAM Phase I: Searching for the trade-off parameter that minimizes data RMS" << std::endl;
			std::cout << " # ...Bracketing Minimum..." << std::endl;
		}
		m_tradeOffParameterOCCA = log10(m_tradeOffParameterForResistivityValuePre);
		m_tradeOffParameterOCCB = m_tradeOffParameterOCCA - 0.25;
		minbrkOCC();
		if (m_rmsOCCB < m_tolreq)
		{
			m_OCCAMsmoothing = true;
			m_rms = m_rmsOCCB;
			m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterOCCB);
		}
		else
		{
			if (myProcessID == 0)
			{
				if (m_inexactMinimizationOfOCCAM)
				{
					std::cout << " # ...Accepting the bracketed Phase-I RMS minimum without Brent refinement..." << std::endl;
				}
				else
				{
					std::cout << " # ...Finding minimum by Brent's minimizing method..." << std::endl;
				}
			}
			m_rms = fminbrentOCC();

			if (myProcessID == 0)
			{
				if (m_inexactMinimizationOfOCCAM)
				{
					std::cout << " # Inexact minimum RMS is at trade-off parameter = "
							  << m_tradeOffParameterForResistivityValue << std::endl;
				}
				else
				{
					std::cout << " # Minimum RMS from fminbrent is at trade-off parameter = "
							  << m_tradeOffParameterForResistivityValue << std::endl;
				}
			}
			if (m_rms < m_tolreq)
			{
				m_OCCAMsmoothing = true;
			}
		}

		if (m_OCCAMsmoothing)
		{
			if (myProcessID == 0)
			{
				std::cout << " # Finding Intercept: bracketing the root (RMS - m_tolreq = 0)..." << std::endl;
			}
			m_tradeOffParameterOCClb = log10(m_tradeOffParameterForResistivityValue);
			m_rmsOCClb = m_rms;
			m_tradeOffParameterOCCub = log10(m_tradeOffParameterForResistivityValue);
			m_rmsOCCub = m_rms;
			while (m_rmsOCCub < m_tolreq)
			{
				m_tradeOffParameterOCCub = m_tradeOffParameterOCCub + 0.30103;
				m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterOCCub);
				m_ptrInversion->inversionCalculation();
				m_rmsOCCub = m_ptrInversion->getrms();
			}
			if (myProcessID == 0)
			{
				std::cout << " # Finding Intercept: approaching the root (RMS - m_tolreq = 0)..." << std::endl;
			}
			ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
			ptrObservedData->copyDistortionParamsCurToPWK2();
			m_tradeOffParameterForResistivityValue = pow(10.0, frootOCC());
			if (myProcessID == 0)
			{
				std::cout << " # Tolerance is met (approximately) at trade-off parameter = "
						  << m_tradeOffParameterForResistivityValue << std::endl;
				std::cout << " # Tolerance met. Next iteration begins smoothing." << std::endl;
			}
			ptrResistivityBlock->copyPWK2NotFixedToPWK1();
			ptrObservedData->copyDistortionParamsPWK2ToPWK1();
		}
		ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
		ptrObservedData->copyDistortionParamsPWK1ToCur();
	}
	else
	{
		if (myProcessID == 0)
		{
			std::cout << " # OCCAM Phase II: Searching for the smoothest model that meets the tolerance" << std::endl;
		}
		m_ptrInversion->inversionCalculation();
		m_rmsOCClb = m_ptrInversion->getrms();
		m_tradeOffParameterOCClb = log10(m_tradeOffParameterForResistivityValue);
		if (m_rmsOCClb < m_tolreq)
		{
			ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
			ptrObservedData->copyDistortionParamsCurToPWK1();
			if (myProcessID == 0)
			{
				std::cout << " # Finding smoother model..." << std::endl;
				std::cout << " # Finding Intercept: bracketing the root (RMS - m_tolreq = 0)..." << std::endl;
			}
			m_rmsOCCub = m_ptrInversion->getrms();
			m_tradeOffParameterOCCub = log10(m_tradeOffParameterForResistivityValue);
			while (m_rmsOCCub < m_tolreq)
			{
				m_tradeOffParameterOCCub = m_tradeOffParameterOCCub + 0.30103;
				m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterOCCub);
				m_ptrInversion->inversionCalculation();
				m_rmsOCCub = m_ptrInversion->getrms();
			}
			if (myProcessID == 0)
			{
				std::cout << " # Finding Intercept: approaching the root (RMS - m_tolreq = 0)..." << std::endl;
			}
			ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
			ptrObservedData->copyDistortionParamsCurToPWK2();
			m_tradeOffParameterForResistivityValue = pow(10.0, frootOCC());
			if (myProcessID == 0)
			{
				std::cout << " # Smoother model is found at trade-off parameter = "
						  << m_tradeOffParameterForResistivityValue << std::endl;
			}
			ptrResistivityBlock->copyPWK2NotFixedToPWK1();
			ptrObservedData->copyDistortionParamsPWK2ToPWK1();
		}
		else
		{
			m_tradeOffParameterOCClb = m_tradeOffParameterOCClb + 0.30103;
			m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterOCClb);
			m_ptrInversion->inversionCalculation();
			m_rmsOCClb = m_ptrInversion->getrms();
			if (m_rmsOCClb >= m_tolreq)
			{
				if (myProcessID == 0)
				{
					std::cout << " # The smoother model may not be found." << std::endl;
				}
				m_leavingOCCAM = true;
				m_OCCAMsmoothing = false;
			}
			else
			{
				m_rmsOCCub = m_ptrInversion->getrms();
				m_tradeOffParameterOCCub = log10(m_tradeOffParameterForResistivityValue);
				while (m_rmsOCCub < m_tolreq)
				{
					m_tradeOffParameterOCCub = m_tradeOffParameterOCCub + 0.30103;
					m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterOCCub);
					m_ptrInversion->inversionCalculation();
					m_rmsOCCub = m_ptrInversion->getrms();
				}
				if (myProcessID == 0)
				{
					std::cout << " # Finding Intercept: approaching the root (RMS - m_tolreq = 0)..." << std::endl;
				}
				ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
				ptrObservedData->copyDistortionParamsCurToPWK2();
				m_tradeOffParameterForResistivityValue = pow(10.0, frootOCC());
				if (myProcessID == 0)
				{
					std::cout << " # Smoother model is found at trade-off parameter = "
							  << m_tradeOffParameterForResistivityValue << std::endl;
				}
				ptrResistivityBlock->copyPWK2NotFixedToPWK1();
				ptrObservedData->copyDistortionParamsPWK2ToPWK1();
			}
		}
		ptrResistivityBlock->copyPWK1NotFixedToResistivityValues();
		ptrObservedData->copyDistortionParamsPWK1ToCur();
	}
}

// subroutine of OCCAM inversion
void AnalysisControl::minbrkOCC()
{
	/*	minbrkOCC brackets a univariate minimum of the RMS.
		To be used prior to a univariate minimisation routine.
		Modified so that the model associated with the misfits is carried around
		for use in the minimisation routines and possibly ultimately kept as the result of this iteration.
		This subroutine is modified based on a subroutine in the OCCAM 3.0 Package.
	References:
		[1] Myer et al., 2007 : OCCAM 3.0 release notes.
	*/
	const double gold = sqrt(1.618034);
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData *const ptrObservedData = ObservedData::getInstance();

	m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterOCCB);
	m_ptrInversion->inversionCalculation();
	ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
	ptrObservedData->copyDistortionParamsCurToPWK1();
	m_rmsOCCB = m_ptrInversion->getrms();

	m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterOCCA);
	m_ptrInversion->inversionCalculation();
	m_rmsOCCA = m_ptrInversion->getrms();

	if (m_rmsOCCB > m_rmsOCCA)
	{
		double temp = m_tradeOffParameterOCCA;
		m_tradeOffParameterOCCA = m_tradeOffParameterOCCB;
		m_tradeOffParameterOCCB = temp;
		temp = m_rmsOCCA;
		m_rmsOCCA = m_rmsOCCB;
		m_rmsOCCB = temp;
		ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
		ptrObservedData->copyDistortionParamsCurToPWK1();
	}

	m_tradeOffParameterOCCC = m_tradeOffParameterOCCB + gold * (m_tradeOffParameterOCCB - m_tradeOffParameterOCCA);
	m_tradeOffParameterForResistivityValue = pow(10.0, m_tradeOffParameterOCCC);
	m_ptrInversion->inversionCalculation();
	m_rmsOCCC = m_ptrInversion->getrms();

	while (m_rmsOCCB > m_rmsOCCC)
	{
		ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
		ptrObservedData->copyDistortionParamsCurToPWK1();
		const double R = (m_tradeOffParameterOCCB - m_tradeOffParameterOCCA) * (m_rmsOCCB - m_rmsOCCC);
		const double Q = (m_tradeOffParameterOCCB - m_tradeOffParameterOCCC) * (m_rmsOCCB - m_rmsOCCA);
		const double BmA = m_tradeOffParameterOCCB - m_tradeOffParameterOCCA;
		const double BmC = m_tradeOffParameterOCCB - m_tradeOffParameterOCCC;
		double U = m_tradeOffParameterOCCB - (BmC * Q - BmA * R) /
			(2.0 * sign(std::max(std::fabs(Q - R), 1.E-32), Q - R));
		const double Ulim = m_tradeOffParameterOCCB + 100.0 * (-1.0 * BmC);
		double rmsOCCU = 0.0;

		if ((m_tradeOffParameterOCCB - U) * (U - m_tradeOffParameterOCCC) > 0)
		{
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			rmsOCCU = m_ptrInversion->getrms();
			if (rmsOCCU < m_rmsOCCC)
			{
				m_tradeOffParameterOCCA = m_tradeOffParameterOCCB;
				m_rmsOCCA = m_rmsOCCB;
				m_tradeOffParameterOCCB = U;
				m_rmsOCCB = rmsOCCU;
				ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
				ptrObservedData->copyDistortionParamsCurToPWK1();
				return;
			}
			else if (rmsOCCU > m_rmsOCCB)
			{
				m_tradeOffParameterOCCC = U;
				m_rmsOCCC = rmsOCCU;
				return;
			}
			U = m_tradeOffParameterOCCC + gold * (m_tradeOffParameterOCCC - m_tradeOffParameterOCCB);
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			rmsOCCU = m_ptrInversion->getrms();
		}
		else if ((m_tradeOffParameterOCCC - U) * (U - Ulim) > 0)
		{
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			rmsOCCU = m_ptrInversion->getrms();
			if (rmsOCCU < m_rmsOCCC)
			{
				m_tradeOffParameterOCCB = m_tradeOffParameterOCCC;
				m_tradeOffParameterOCCC = U;
				U = m_tradeOffParameterOCCC + gold * (m_tradeOffParameterOCCC - m_tradeOffParameterOCCB);
				m_rmsOCCB = m_rmsOCCC;
				m_rmsOCCC = rmsOCCU;
				ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
				ptrObservedData->copyDistortionParamsCurToPWK1();
				m_tradeOffParameterForResistivityValue = pow(10.0, U);
				m_ptrInversion->inversionCalculation();
				rmsOCCU = m_ptrInversion->getrms();
			}
		}
		else if ((U - Ulim) * (Ulim - m_tradeOffParameterOCCC) > 0)
		{
			U = Ulim;
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			rmsOCCU = m_ptrInversion->getrms();
		}
		else
		{
			U = m_tradeOffParameterOCCC + gold * (m_tradeOffParameterOCCC - m_tradeOffParameterOCCB);
			m_tradeOffParameterForResistivityValue = pow(10.0, U);
			m_ptrInversion->inversionCalculation();
			rmsOCCU = m_ptrInversion->getrms();
		}
		m_tradeOffParameterOCCA = m_tradeOffParameterOCCB;
		m_tradeOffParameterOCCB = m_tradeOffParameterOCCC;
		m_tradeOffParameterOCCC = U;
		m_rmsOCCA = m_rmsOCCB;
		m_rmsOCCB = m_rmsOCCC;
		m_rmsOCCC = rmsOCCU;
	}
}

// subroutine of OCCAM inversion
double AnalysisControl::frootOCC()
{
	/*	FROOT FINDS THE POINT AT WHICH A UNIVARIATE FUNCTION ATTAINS A GIVEN VALUE (m_tolreq).
	References:
		[1] Myer et al., 2007: OCCAM 3.0 release notes.
	*/
	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData *const ptrObservedData = ObservedData::getInstance();
	const int ITMAX = 100;
	const double EPS = 3.E-8;
	const double tol = 0.1;

	double aa = m_tradeOffParameterOCClb;
	double b = m_tradeOffParameterOCCub;
	double fa = m_rmsOCClb - m_tolreq;
	double fb = m_rmsOCCub - m_tolreq;
	double fc = fb;
	double c = 0.0;
	double dd = 0.0;
	double e = 0.0;

	const int myProcessID = getMyPE();
	if (fa * fb > 0.0)
	{
		if (myProcessID == 0)
		{
			std::cout << "ROOT NOT BRACKETED IN FROOT" << std::endl;
		}
		return b;
	}

	ptrResistivityBlock->copyPWK2NotFixedToPWK3();
	ptrObservedData->copyDistortionParamsPWK2ToPWK3();

	for (int iter = 0; iter < ITMAX; ++iter)
	{
		if (fb * fc > 0.0)
		{
			c = aa;
			fc = fa;
			dd = b - aa;
			e = dd;
			ptrResistivityBlock->copyPWK1NotFixedToPWK3();
			ptrObservedData->copyDistortionParamsPWK1ToPWK3();
		}

		if (std::fabs(fc) < std::fabs(fb))
		{
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

		const double tol1 = 2.0 * EPS * std::fabs(b) + 0.5 * tol;
		const double xm = 0.5 * (c - b);

		if (std::fabs(xm) <= tol1 || fb == 0.0)
		{
			return b;
		}

		if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb))
		{
			const double s = fb / fa;
			double p = 0.0;
			double q = 0.0;

			if (aa == c)
			{
				p = 2.0 * xm * s;
				q = 1.0 - s;
			}
			else
			{
				const double qPrevious = fa / fc;
				const double r = fb / fc;
				p = s * (2.0 * xm * qPrevious * (qPrevious - r) - (b - aa) * (r - 1.0));
				q = (qPrevious - 1.0) * (r - 1.0) * (s - 1.0);
			}

			if (p > 0.0)
			{
				q = -q;
			}
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
			dd = xm;
			e = dd;
		}

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

		m_tradeOffParameterForResistivityValue = pow(10.0, b);
		m_ptrInversion->inversionCalculation();
		fb = m_ptrInversion->getrms() - m_tolreq;
		ptrResistivityBlock->copyResistivityValuesNotFixedToPWK2();
		ptrObservedData->copyDistortionParamsCurToPWK2();
	}

	if (myProcessID == 0)
	{
		std::cerr << "MAXIMUM ITERATIONS EXCEEDED IN FROOT" << std::endl;
	}
	return b;
}

// subroutine of OCCAM inversion
double AnalysisControl::fminbrentOCC()
{
	/*	fminbrentOCC returns the minimum RMS value within a specified interval.
		This function implements Brent's method for function minimization, using
		parabolic interpolation and golden section search.
		This subroutine is based on Brent's (1973) minimizing method
		and modified based on a subroutine in the OCCAM 3.0 Package.
	References:
		[1] Brent, R. P., 1973. Chapter 4: An Algorithm with Guaranteed Convergence for Finding a Zero of a Function,
	Algorithms for Minimization without Derivatives, Englewood Cliffs, NJ: Prentice-Hall, ISBN 0-13-022335-2
		[2] Myer et al., 2007: OCCAM 3.0 release notes.
	*/
	const int ITMAX = 100;
	const double CGOLD = 0.3819660;
	const double ZEPS = 1.0E-10;
	const double tol = 0.1;

	double lowerBound = std::min(m_tradeOffParameterOCCC, m_tradeOffParameterOCCA);
	double upperBound = std::max(m_tradeOffParameterOCCC, m_tradeOffParameterOCCA);
	double x = m_tradeOffParameterOCCB;
	double w = x;
	double v = x;
	double e = 0.0;
	double fx = m_rmsOCCB;
	double fw = fx;
	double fv = fx;

	double midpoint = 0.0;
	double tol1 = 0.0;
	double tol2 = 0.0;
	double r = 0.0;
	double q = 0.0;
	double p = 0.0;
	double etemp = 0.0;
	double step = 0.0;
	double u = 0.0;
	double fu = 0.0;

	ResistivityBlock *const ptrResistivityBlock = ResistivityBlock::getInstance();
	ObservedData *const ptrObservedData = ObservedData::getInstance();

	for (int iter = 0; iter < ITMAX; ++iter)
	{
		midpoint = 0.5 * (lowerBound + upperBound);
		tol1 = tol * std::abs(x) + ZEPS;
		tol2 = 2.0 * tol1;

		if (std::abs(x - midpoint) <= (tol2 - 0.5 * (upperBound - lowerBound)) ||
			m_inexactMinimizationOfOCCAM)
		{
			m_tradeOffParameterForResistivityValue = pow(10.0, x);
			return fx;
		}

		if (std::abs(e) > tol1)
		{
			r = (x - w) * (fx - fv);
			q = (x - v) * (fx - fw);
			p = (x - v) * q - (x - w) * r;
			q = 2.0 * (q - r);
			if (q > 0.0)
			{
				p = -p;
			}
			q = std::abs(q);

			etemp = e;
			e = step;

			if (std::abs(p) < std::abs(0.5 * q * etemp) &&
				p > q * (lowerBound - x) &&
				p < q * (upperBound - x))
			{
				step = p / q;
				u = x + step;

				if ((u - lowerBound) < tol2 || (upperBound - u) < tol2)
				{
					step = std::copysign(tol1, midpoint - x);
				}
			}
			else
			{
				e = (x >= midpoint) ? (lowerBound - x) : (upperBound - x);
				step = CGOLD * e;
			}
		}
		else
		{
			e = (x >= midpoint) ? (lowerBound - x) : (upperBound - x);
			step = CGOLD * e;
		}

		u = (std::abs(step) >= tol1) ? (x + step) : (x + std::copysign(tol1, step));
		m_tradeOffParameterForResistivityValue = pow(10.0, u);
		m_ptrInversion->inversionCalculation();
		fu = m_ptrInversion->getrms();

		if (fu <= fx)
		{
			if (u >= x)
			{
				lowerBound = x;
			}
			else
			{
				upperBound = x;
			}

			v = w;
			fv = fw;
			w = x;
			fw = fx;
			x = u;
			fx = fu;
			ptrResistivityBlock->copyResistivityValuesNotFixedToPWK1();
			ptrObservedData->copyDistortionParamsCurToPWK1();
		}
		else
		{
			if (u < x)
			{
				lowerBound = u;
			}
			else
			{
				upperBound = u;
			}

			if (fu <= fw || w == x)
			{
				v = w;
				fv = fw;
				w = u;
				fw = fu;
			}
			else if (fu <= fv || v == x || v == w)
			{
				v = u;
				fv = fu;
			}
		}
	}

	std::cerr << "Warning: Maximum iterations exceeded in fminbrentOCC" << std::endl;
	m_tradeOffParameterForResistivityValue = pow(10.0, x);
	return fx;
}
