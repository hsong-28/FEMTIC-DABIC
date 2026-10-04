# FEMTIC-DABIC

FEMTIC-DABIC is a 3-D magnetotelluric (MT) inversion program
derived from [FEMTIC](https://github.com/yoshiya-usui/femtic). It uses a data-space
variant of Akaike's Bayesian Information Criterion (DABIC) for statistically
guided selection of the regularization parameter and model. OCCAM and nonlinear
cubic-spline L-curve inversion offer alternative paths.

Current release: **v2.7.1**.

## Main Features

- FEMTIC-based 3-D MT forward modeling and mesh support;
- DABIC/ABIC and OCCAM inversion with exact and inexact options, plus
  cubic-spline L-curve inversion;
- configurable regularization, reference-model constraints, and galvanic
  distortion correction;
- model-resolution and covariance-diagonal appraisal;
- optional HDF5 export of resistivity models, predicted responses, the
  data-weighted Jacobian, and the roughening matrix.

## Documentation

The base user manual and v2.7.1 additions are available in:

- [FEMTIC-DABIC User Manual v2.7.0](docs/FEMTIC-DABIC_UserManual_v2.7.0.pdf)
- [v2.7.1 HDF5 and workflow additions](src/README_hdf5.md)

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

`src/Makefile` is the only build entry point. HDF5 and input-file remapping
are optional and disabled by default:

```bash
make HDF5_OUT=yes HDF5_JAC=yes
make HDF5_OUT=yes HDF5_JAC=yes INPUT_FILE_MAP=yes
```

Both commands build `femtic-dabic_h5_results_exchange.x`. HDF5 builds require
the HDF5 development library; writing files additionally requires the
corresponding `control.dat` switches. See [HDF5 build and output options](src/README_hdf5.md).

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

## Examples

A runnable example set based on the simplified Atotsugawa Fault model is
provided in [`examples/AtotsugawaFault`](examples/AtotsugawaFault). It includes
exact ABIC, fixed-alpha inversion with a fixed reference model, and exact ABIC
with distortion correction. The mesh, observations, and initial model are
stored once under `shared_inputs`.

```bash
case_name=exact_ABIC
run_dir=/tmp/femtic-dabic-atotsugawa/${case_name}
mkdir -p "${run_dir}"
cp examples/AtotsugawaFault/shared_inputs/* "${run_dir}/"
cp examples/AtotsugawaFault/${case_name}/* "${run_dir}/"
cd "${run_dir}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mpirun -np 2 /path/to/FEMTIC-DABIC/src/femtic-dabic
```

Set `case_name` to `fixed_alpha_reference` or `ABIC_with_distortion` to prepare
the other cases. All supplied controls run iterations 0-2. See the example
README before changing inversion settings.

[`examples/BrokenHill`](examples/BrokenHill) provides six field-data ABIC cases
covering mesh refinement, Difference-L1 regularization, and galvanic distortion.
It includes native inputs, reference convergence histories, a report, and an
A-A' plotting script. See its README for usage and validation scope.

## Repository Layout

```text
src/       program source and Makefile
docs/      user manual
examples/  Atotsugawa Fault and Broken Hill examples
LICENSE    MIT license and FEMTIC/FEMTIC-DABIC attribution
```

Full inversion outputs, debug runs, and server scratch directories are not
included in the GitHub source release.

## Release Note

***v2.7.1*** Oct. 3, 2026: Integrated Volker Rath's HDF5 output
and Python readers, optional input-file mapping, control-file enhancements,
sensitivity exports, and signed model-resolution diagonal correction.
Follow-up maintenance unifies the Makefile, validates control inputs, and
checks optional exports while retaining the existing output formats.

For earlier changes, see the [CHANGELOG](CHANGELOG.md).

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
L-curve, appraisal, reporting, and workflow extensions.

## License and Attribution

FEMTIC-DABIC is distributed under the MIT License. See [LICENSE](LICENSE).

- Original FEMTIC source: Copyright (c) 2021 Yoshiya Usui
- FEMTIC-DABIC modifications: Copyright (c) 2025-2026 Han Song

[Volker Rath](https://github.com/volkerrath) (DIAS) contributed the HDF5 and
related workflow extensions described above. His original commits and
file-level attribution are retained.

Files derived from upstream FEMTIC retain the original attribution. External
dependencies, including MPI and Intel oneAPI/MKL, are governed by their own
licenses.
