# femtic_hdf5_readers.py

Python (3.11+) readers for the HDF5 output of FEMTIC, for the three active
source trees `femtic_v4_src`, `femtic_v5_src` and `femtic_dabic_v2.7_src`.
All three trees write the same layout from `OutputHDF5.cpp`. The only
difference is in v5 anisotropic runs, which add tensor columns to
`/model/blocks` and a `@anisotropic` attribute.

Requirements: `h5py` (mandatory) and `scipy` (optional; without it
`read_rough_hdf5` returns the raw CSR arrays).

Contributed by Volker Rath (DIAS). Validate against representative solver
outputs before production use.

---

## Files

| File | C++ writer | Build flag | Written |
|---|---|---|---|
| `results_iter<N>.h5` | `outputResultsToHDF5()` | `HDF5_OUT` | each iteration N > `ITERATION_NUM_INIT` |
| `exchange.h5` | `outputJacobianToHDF5()` | `HDF5_JAC` | written at iteration `ITERATION_NUM_MAX - 1` or at an early converged iteration |

`exchange.h5` contains `/jacobian`, `/rough` (roughening matrix, CSR) and
`/mesh`. It replaces the former `jacobian.h5`, `rough.h5` and `mesh.h5`
(merged 2026-10-01; v2.7 tree so far).

The Jacobian describes its recorded linearization iteration, not necessarily
the final model. Check `/metadata` and `/jacobian/metadata` `iterNum` against
the run log before pairing it with a results file. The scheduled final
iteration is forward-only. Failed exports may leave an older valid file;
file existence alone is not evidence of a current export.

---

## Quick start

```python
import numpy as np
from femtic_hdf5_readers import (
    read_results_hdf5, read_exchange_hdf5,
    read_jacobian_hdf5, read_rough_hdf5, read_mesh_hdf5,
    read_femtic_hdf5, inspect_file,
)

res  = read_results_hdf5("results_iter5.h5")
rho  = res.model.rho                               # (nBlocks,) Ohm.m
sens = res.model.sensitivity_volume_normalised     # (nBlocks,) or None
mt   = res.data_of_type("MT")                      # structured array

ex   = read_exchange_hdf5("exchange.h5")           # everything in one call
jac, R, mesh = ex.jacobian, ex.rough, ex.mesh      # jac.J (nData, nModel); R csr_matrix
# or one group at a time:
jac  = read_jacobian_hdf5("exchange.h5")
R    = read_rough_hdf5("exchange.h5")
mesh = read_mesh_hdf5("exchange.h5")

obj  = read_femtic_hdf5("some_file.h5")            # detects the kind by content
print(inspect_file("some_file.h5"))                # tree dump for debugging
```

Running the module as a script prints a summary table of all results
files. There are no CLI arguments; edit the USER SECTION at the end of the
file:

```
RESULTS_GLOB = "results_iter*.h5"
SENS_FILL    = None       # or np.nan
PRINT_TREE   = False
```

Example output:

```
------------------------------------------------------------------------------
file                      iter  nBlocks   nFree  sens    nData  nMiss  dist
------------------------------------------------------------------------------
results_iter3.h5             3        5       3   yes        3      0   yes
results_iter10.h5           10        5       3    no        3      0    no
------------------------------------------------------------------------------
```

---

## Indexing: blocks vs. model parameters

- **blockID** runs over all nBlocks resistivity blocks, fixed ones
  included (air, sea, any other fixed region). It is used by
  `/model/blocks`, `/model/sensitivity/*` and `/model/element_block_map`.
- In isotropic DABIC, **resistivity modelID** runs over free blocks (type 0
  or 3) in ascending blockID order. These occupy the leading Jacobian columns;
  estimated distortion parameters follow them and have no resistivity blockID.
- **`/rough`** uses blockID on both axes and includes fixed blocks. It is
  the pre-degeneration operator, not the solver's weighted free-parameter
  operator. Fixed-block contributions, filter/IRLS weights, reference terms,
  regularization weights and distortion penalties are separate. Do not form
  `J.T @ J + R.T @ R` directly from these two raw exports.

`FemticModel.model_to_block` gives the blockID of each free resistivity
parameter only. It returns `None` for v5 anisotropic runs. For example,
the small isotropic test has 10,694 blocks but only 10,693 free resistivity
parameters: `/rough` has 10,694 rows while J has 10,693 columns without distortion.

```python
m = res.model
J_blocks = np.zeros((jac.n_data, m.n_blocks))
n_free_rho = len(m.model_to_block)
J_blocks[:, m.model_to_block] = jac.J[:, :n_free_rho]
J_distortion = jac.J[:, n_free_rho:]  # separate parameters, possibly zero columns
```

---

## results_iter<N>.h5

### Layout

```
/model/metadata                       @iterNum @nElem @nNodes @nBlocks [@anisotropic]
/model/element_block_map              int[nElem]
/model/blocks                         compound[nBlocks]: blockID, resistivity, rho_min,
                                      rho_max, weight, type [, rho_xx, rho_yy, rho_zz,
                                      strike, dip, slant]
/model/mesh/node_coords               double[nNodes,3]
/model/mesh/elem_nodes                int[nElem,nNodesPerElem]
/model/sensitivity/raw                double[nBlocks]     only if computed
/model/sensitivity/volume_normalised  double[nBlocks]     only if computed
/data/metadata                        @nRows @iterNum @nMissingCalc
/data/data                            compound[nRows]: freq, datatype, site_id,
                                      site_x, site_y, site_z, re_val, im_val,
                                      re_err, im_err, cal_re, cal_im, component
/distortion/metadata                  @iterNum @nRows @type    only if distortion estimated
/distortion/params                    compound[nSites]: site_id, param1..4, isFixed
```

### Sensitivity

- The sensitivity arrays have the same length (nBlocks) and blockID
  indexing as `/model/blocks`.
- `raw` is sum over data of |J[:, m]|, reduced over all MPI ranks.
  `volume_normalised` is `raw` divided by the block volume (m^-3).
- Fixed blocks, and free blocks with |S| <= 1e-20, hold
  `SENSITIVITY_FLOOR = 1e-20`. Use `model.free_mask` to exclude fixed
  blocks.
- The reader raises `FemticHDF5SchemaError` if the sensitivity length
  differs from the number of blocks.

**When sensitivity is absent.** `/model/sensitivity` is written only if
`AnalysisControl::doesCalculateSensitivity(iter)` is true, which is
`iter < ITERATION_NUM_MAX` in all three trees. Results files are written
only for `iter > ITERATION_NUM_INIT`, and only after the retrial loop of
that iteration succeeded. For a run from i0 to N:

```
results_iter<i0>.h5        not written
results_iter<k>.h5         i0 < k < N : with sensitivity
results_iter<N>.h5         without sensitivity (forward-only final iteration)
convergence at k < N       last file results_iter<k>.h5 has sensitivity
retrials exhausted at k    no file for k
```

`read_results_hdf5(path, sens_fill=np.nan)` returns NaN-filled `(nBlocks,)`
arrays for a file without sensitivity, so every iteration has the same
shape. The default `sens_fill=None` returns `None`.

**Retrial accumulation, fixed 2026-09-25 (all three trees).** The
sensitivity accumulator was zeroed once per iteration, but
`calculateSensitivityMatrix()` adds to it on every retrial attempt. For an
iteration that needed retrials, the stored sensitivity was the sum over
all retrial models tried, not just the accepted one -- affecting the HDF5
output, `sensitivity_iterN.dat` and the VTK output. The accumulator is now
re-zeroed at the start of every retrial (moved from once-per-iteration,
before the retrial loop, to once-per-attempt, inside it), so it always
reflects only the model that was ultimately accepted. This is a C++-only
change; the file layout above is unaffected. Files written before the fix
may still hold the summed-over-retrials values for iterations that needed
a retrial; there is no attribute distinguishing old files from new ones.

### `read_results_hdf5(path, sens_fill=None) -> FemticResults`

| Field | Type | Description |
|---|---|---|
| `model` | `FemticModel` | the `/model` group (table below) |
| `data` | structured `ndarray (nRows,)` | `/data/data` |
| `data_attrs` | `dict` | `nRows`, `iterNum`, `nMissingCalc` |
| `distortion` | structured `ndarray` or `None` | `/distortion/params` |
| `distortion_attrs` | `dict` or `None` | `iterNum`, `nRows`, `type` |
| `iteration` | `int` or `None` | iteration number |
| `sensitivity_computed` | `bool` (property) | `/model/sensitivity` present |
| `distortion_type` | `str` or `None` (property) | name from `DISTORTION_TYPE_NAMES` |
| `data_of_type(t)` | method | rows for a datatype code or name (`"MT"`, `"VTF"`, ...) |

### `FemticModel`

| Field | Type | Description |
|---|---|---|
| `rho`, `rho_min`, `rho_max` | `(nBlocks,)` | resistivity and bounds, Ohm.m |
| `weight` | `(nBlocks,)` | roughening weight (1.0 placeholder for v5 anisotropic blocks) |
| `block_type` | `(nBlocks,)` int | `BLOCK_*` codes: 0 free/constrained, 1 fixed/isolated, 2 fixed/constrained, 3 free/isolated |
| `element_block_map` | `(nElem,)` | blockID of each element |
| `node_coords` | `(nNodes, 3)` | node coordinates |
| `elem_nodes` | `(nElem, nNodesPerElem)` | element connectivity |
| `sensitivity_raw` | `(nBlocks,)` or `None` | see above |
| `sensitivity_volume_normalised` | `(nBlocks,)` or `None` | see above |
| `sensitivity_computed` | `bool` | sensitivity present in the file |
| `iteration`, `attrs`, `raw_blocks` | | `@iterNum`, `/model/metadata`, full `/model/blocks` |
| `rho_xx` ... `slant`, `is_anisotropic` | `(nBlocks,)` or `None` | v5 anisotropic runs only |
| `n_blocks` | property | number of blocks |
| `free_mask` | property, `(nBlocks,)` bool | True for free blocks |
| `model_to_block` | property, `(nFreeResistivity,)` or `None` | blockID of each free resistivity parameter; excludes distortion columns |
| `sensitivity_per_element(volume_normalised=True)` | method | sensitivity mapped to elements |

### Data codes

`datatype`: 0 MT, 1 APP_RES_AND_PHS, 2 HTF, 3 VTF, 4 PT, 5 NMT, 6 NMT2,
7 NMT2_APP_RES_AND_PHS (`DATATYPE_NAMES`).

`component` (`COMPONENT_NAMES`):

```
MT, NMT2                 0 Zxx   1 Zxy   2 Zyx   3 Zyy
APP_RES_AND_PHS,         0 rhoXX 1 rhoXY 2 rhoYX 3 rhoYY
NMT2_APP_RES_AND_PHS     4 phsXX 5 phsXY 6 phsYX 7 phsYY
HTF                      0 Txx   1 Txy   2 Tyx   3 Tyy
VTF                      0 Tzx   1 Tzy
PT                       0 PTxx  1 PTxy  2 PTyx  3 PTyy
NMT                      0 Yx    1 Yy
```

For real-valued types (APP_RES_AND_PHS, PT), `im_val`, `im_err` and `cal_im`
are 0. `cal_re` and `cal_im` are NaN only if no rank reported a calculated
value for that row; the number of such rows is `@nMissingCalc`.

`/distortion/params` param1..4 depend on `@type`:

```
1 ESTIMATE_DISTORTION_MATRIX_DIFFERENCE   Cxx, Cxy, Cyx, Cyy
2 ESTIMATE_GAINS_AND_ROTATIONS            ExGain, EyGain, ExRot (deg), EyRot (deg)
3 ESTIMATE_GAINS_ONLY                     ExGain, EyGain
```

---

## exchange.h5

```
/metadata                @iterNum @exchangeVersion(=1)

/jacobian/metadata       @iterNum @nData @nModel @weighted(=1)
/jacobian/values         double[nData, nModel]    J = Cd^{-1/2} dF/dm
/jacobian/data_errors    double[nData]            SD of each row

/rough/metadata          @nRows @nNonZeros @format="CSR"
/rough/row_ptr           int[nRows+1]
/rough/col_ind           int[nNonZeros]
/rough/values            double[nNonZeros]

/mesh/metadata           @meshType @nNodes @nElem @nNodesPerElem @neighborFormat [@nNeighborElem]
/mesh/node_coords        double[nNodes,3]
/mesh/elem_nodes         int[nElem,nNodesPerElem]
/mesh/neighbor_elements  int[nElem,nNeighborElem]   (neighborFormat 0: HEXA, TETRA)
                         int[nnz]                   (neighborFormat 1: non-conforming hexa, CSR)
/mesh/neighbor_face_ptr  int[nElem*6+1]             (neighborFormat 1 only)
```

`read_exchange_hdf5(path) -> FemticExchange` with `jacobian`, `rough`,
`mesh`, `iteration` and `attrs`.

- `jacobian` is a `FemticJacobian` (`J`, `data_errors`, `iteration`,
  `n_data`, `n_model`, `weighted`, `attrs`). `unweighted()` returns
  dF/dm = J * SD. The rows follow the inversion's global data order, which
  is not the row order of `/data/data`.
- `rough` is an nModel x nModel `scipy.sparse.csr_matrix`, or a dict of the
  raw arrays if scipy is not installed.
- `mesh` is a `FemticMesh`; `neighbors_of(e)` returns the neighbors of
  element e for either layout.

`read_jacobian_hdf5`, `read_rough_hdf5` and `read_mesh_hdf5` read a single
group from the same file.

---

## Errors

Every reader raises `FemticHDF5SchemaError` when a file does not match the
layout above. This covers missing datasets, inconsistent lengths, and
sensitivity length != nBlocks. The message lists the keys that were found.
Run `inspect_file(path)` on the file to see its contents.

---

## Limitations

- Not yet tested against a FEMTIC-written file. h5py was unavailable in
  the environment where this was written, so tests used an in-memory h5py
  stand-in built from the C++ writer's schema. The 2026-10-01 exchange.h5
  reader changes were tested the same way.
- The legacy `model_iter<N>.h5` / `data_iter<N>.h5` files and the
  deprecated dabic v1.4/v1.5/v1.5.2 layouts are no longer supported.
