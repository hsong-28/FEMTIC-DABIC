# Changelog

## v2.7.1 - Oct. 3, 2026

Integrated Volker Rath's HDF5 output
and Python readers, optional input-file mapping, control-file enhancements,
sensitivity exports, and signed model-resolution diagonal correction.
Follow-up maintenance unifies the Makefile, validates control inputs, and
checks optional exports while retaining the existing output formats.

## Earlier Releases

***v2.7*** Aug. 3, 2026: Simplified the ABIC interface while preserving the
bracket-only inexact workflow and same-alpha reduced-step retry.

***v2.6*** Jul. 16, 2026: Implemented scalar negative-standard-deviation
masking for raw MT and VTF real and imaginary data, so inactive observations no
longer contribute to the residual, sensitivity matrix, RMS, or model update.

***v2.5*** Jun. 27, 2026: Added production model-resolution and
covariance-diagonal appraisal.

***v2.3*** Jun. 23, 2026: Unified fixed-alpha, ABIC, OCCAM, linear
cubic-spline L-curve, and nonlinear cubic-spline L-curve inversion under one
maintained control and diagnostic framework.

***v1.4*** Jan. 12, 2026: I've revised the D-DABIC workflow to ensure it is
capable of incorporating the distortion correction functionality.

***v1.3*** Sep. 13, 2025: Added Minimum Norm (MN) Stabilizer with Depth of
Investigation (DOI) Support. Introduced a new regularization option
(`|m - m_r|`) to constrain inversion toward a reference model (`m_r`); the
primary purpose of this option (for now) is to enable DOI analysis for model
appraisal.

***v1.2*** Sep. 11, 2025: Reference Model (`m_r`) Configuration Option. Added
support for defining a user-provided reference model (`m_r`), enabling
physics-based constraints in the inversion.

***v1.1*** Dec. 30, 2024: Laplacian Filter (LF) for Marginal Likelihood
Maximization. Enabled the LF as an alternative regularization during the
D-DABIC optimization.

***v1.0*** Nov. 28, 2024: Core FEMTIC-DABIC Framework. Implemented a 3-D
data-space inversion method using a data-space variant of Akaike's Bayesian
Information Criterion (D-DABIC).
