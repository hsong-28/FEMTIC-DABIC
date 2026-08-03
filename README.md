# FEMTIC-DABIC

FEMTIC-DABIC is a general-purpose 3-D magnetotelluric (MT) inversion program
derived from [FEMTIC](https://github.com/yoshiya-usui/femtic). Its principal
inversion method is a data-space variant of Akaike's Bayesian Information
Criterion (D-DABIC); OCCAM and nonlinear cubic-spline L-curve inversion are
also supported, together with L2, L1-style, and L0-style regularization.

Current release: **v2.7.0**.

## Main Features

- 3-D MT forward modeling and Gauss-Newton inversion based on FEMTIC;
- exact and inexact D-DABIC/ABIC inversion;
- exact and inexact OCCAM inversion;
- nonlinear cubic-spline L-curve inversion;
- L2, L1-style, and L0-style regularization with Difference-filter support;
- Laplacian L2 roughness as an alternative to the Difference filter;
- fixed-alpha, linear L-curve, and data-fit cooling as additional modes;
- optional reference-model/minimum-norm constraints and galvanic distortion
  estimation;
- optional Levenberg-Marquardt damping for stabilizing model updates;
- model-resolution and covariance-diagonal appraisal;
- runtime summaries for reviewing inversion results.

## Documentation

The complete installation, input, control-keyword, inversion-method, and output
reference is available in:

- [FEMTIC-DABIC User Manual v2.7.0](FEMTIC-DABIC_UserManual_v2.7.0.pdf)

## Build

The maintained Makefile targets Linux or WSL with Intel oneAPI MPI, OpenMP,
and MKL ILP64.

```bash
source /opt/intel/oneapi/setvars.sh
cd src
make check-env
make -j2
```

The executable is generated as:

```text
src/femtic-dabic
```

## Minimal Run

A standard run directory contains:

```text
control.dat
mesh.dat
observe.dat
resistivity_block_iter0.dat
```

Run the program from that directory so all native inputs and outputs remain
traceable to the same case:

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mpirun -np 2 /path/to/FEMTIC-DABIC/src/femtic-dabic
```

`Referencemodel.dat` is required only when reference-model/minimum-norm
stabilization is enabled. See the User Manual before preparing a scientific or
production run.

## Repository Layout

```text
src/       program source and Makefile
LICENSE    MIT license and FEMTIC/FEMTIC-DABIC attribution
```

Datasets, inversion outputs, debug runs, and server scratch directories are
not included in the GitHub source release.

## Release Note

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

## Citation

Publications using the data-space ABIC, OCCAM, or other inversion capabilities
added and maintained in FEMTIC-DABIC should cite:

- Song, H., Yu, P., Usui, Y., Uyeshima, M., Diba, D., and Zhang, L. (2026).
  Three-dimensional magnetotelluric inversion based on a data-space variant of
  Akaike's Bayesian information criterion. *Geophysics*, 91(3), E111-E126.
  https://doi.org/10.1190/geo-2025-0233
- Usui, Y., Ogawa, Y., Aizawa, K., Kanda, W., Hashimoto, T., Koyama, T.,
  Yamaya, Y., and Kagiyama, T. (2017). Three-dimensional resistivity structure
  of Asama Volcano revealed by data-space magnetotelluric inversion using
  unstructured tetrahedral elements. *Geophysical Journal International*,
  208(3), 1359-1372. https://doi.org/10.1093/gji/ggw459
- Usui, Y. (2015). 3-D inversion of magnetotelluric data using unstructured
  tetrahedral elements: applicability to data affected by topography.
  *Geophysical Journal International*, 202(2), 828-849.
  https://doi.org/10.1093/gji/ggv186

If the non-conforming deformed hexahedral mesh (`MESH_TYPE=2`) is used, also
cite:

- Usui, Y., Uyeshima, M., Hase, H., Ichihara, H., Aizawa, K., Koyama, T.,
  Sakanaka, S., et al. (2024). Three-dimensional electrical resistivity
  structure beneath a strain concentration area in the back-arc side of the
  northeastern Japan Arc. *Journal of Geophysical Research: Solid Earth*,
  129(5), e2023JB028522. https://doi.org/10.1029/2023JB028522

## Relationship to FEMTIC

FEMTIC-DABIC is a maintained derivative of Yoshiya Usui's FEMTIC, not an
independent or clean-room implementation. FEMTIC provides the underlying mesh,
data, forward-modeling, inversion, sparse-linear-algebra, solver, and native
I/O architecture. FEMTIC-DABIC adds and maintains the D-DABIC/ABIC, OCCAM,
L-curve, Lp regularization, appraisal, reporting, and workflow extensions.

## License and Attribution

FEMTIC-DABIC is distributed under the MIT License. See [LICENSE](LICENSE).

- Original FEMTIC source: Copyright (c) 2021 Yoshiya Usui
- FEMTIC-DABIC modifications: Copyright (c) 2025-2026 Han Song

Files derived from upstream FEMTIC retain the original attribution. External
dependencies, including MPI and Intel oneAPI/MKL, are governed by their own
licenses.
