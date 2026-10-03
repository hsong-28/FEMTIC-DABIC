# FEMTIC-DABIC v2.7 — Modifications

**Modified by:** Volker Rath (DIAS) with the help of Claude.

This file documents the extensions made to FEMTIC-DABIC v2.7 (ABIC-based
regularization search by Han Song, cross-gradient/`ConstrainingModel`
features by Dieno Diba and Han Song, on top of Yoshiya Usui's original
FEMTIC) beyond the official release described in
`docs/FEMTIC-DABIC_UserManual_v2.7.0.pdf`. Three independent,
compile-time-optional features are covered:

1. [Building / oneAPI toolchain selection](#0-building)
2. [Input file name remapping and `#`-comments](#1-input-file-name-remapping-and--comments) (2026-08-05)
3. [HDF5 output extension](#2-hdf5-output-extension) (2026-08-21, ported from `femtic_v4_src`)

Each feature is off/on independently and gated behind its own preprocessor
flag, so any of them can be disabled without affecting the others or the
base build. See `ECOSYSTEM_STATUS.md` (one level up) for the technical
detail behind every change made across the whole FEMTIC/FEMTIC-DABIC
ecosystem, including cross-tree parity notes for `femtic_v4_src` and
`femtic_v5_src`.

---

## 0. Building

The Makefile is `Makefile_hdf5` (renamed 2026-10-01); use `make -f
Makefile_hdf5 ...`, or copy/symlink it to `Makefile`.

```bash
make -f Makefile_hdf5          # release build
make -f Makefile_hdf5 debug    # debug build, same toolchain
```

Requires Intel oneAPI 2024.0 or later: the compiler-agnostic MPI wrapper
`mpiicpx` (invoking the LLVM-based `icpx`) is used for `CXX` and `CC`.
Support for oneAPI 2023.1 and earlier (`ONEAPI_2023=yes`) was removed
2026-10-01.

`icpx`/`icx` use their own bundled libstdc++ headers by default; on systems
where the module-loaded/system GCC is a different major version than the
one oneAPI was validated against, set
`GCC_INSTALL_DIR=/path/to/gcc/version-dir` to pass `--gcc-install-dir=...`
(no flag is passed by default):

```bash
make -f Makefile_hdf5 GCC_INSTALL_DIR=/usr/lib/gcc/x86_64-linux-gnu/14
```

`MKL_ILP64` is always used (all MKL/LAPACK integers passed to
`LAPACKE_*`/`cblas_*` are `MKL_INT`, which resolves to a 64-bit integer
under this build); this is independent of the `HDF5_*`/`INPUT_FILE_MAP`
feature flags below.

`make check-env` verifies `$(CXX)` is on `PATH` and that `MKLROOT` points
at a real ILP64 MKL installation before you try a full build — useful
after moving to a new machine. `make print-config` echoes every relevant
variable (`CXX`, `CC`, `GCC_INSTALL_DIR`, `MKLROOT`, `INPUT_FILE_MAP`, the
`HDF5_*` flags, and the final `CXXFLAGS`/`DEBUG_CXXFLAGS`) without building
anything. (Both targets also need `-f Makefile_hdf5`.)

### Other build targets

```bash
make release            # same as plain `make`
make debug              # -g -O0, same CXX/CC selection
make cutback-test        # builds femtic-dabic-cutback-test into a separate
                         # object directory, with
                         # -DFEMTIC_DABIC_TEST_FORCE_FIRST_INEXACT_ABIC_CUTBACK
make clean               # removes build/ and both PROGRAM binaries
```

---

## 1. Input file name remapping and `#`-comments

**Added:** 2026-08-05.

`control.dat` (and the other FEMTIC-DABIC input files) can use `#` to start
a comment — either a whole comment line, or trailing text after real
content on the same line — instead of requiring every line to be pure
data. `control.dat` parsing is also order-independent (each keyword is
located by a `seekKeyword`-style lookup rather than a strict top-to-bottom
sequential read), so blocks can appear in any order, with one exception:
blocks with a genuine runtime dependency on another block having already
run (e.g. `PARAM_DISTORTION` depending on `DISTORTION` having set the
distortion type) must still appear in a compatible relative order.

On top of that, FEMTIC-DABIC's hard-coded input file names can be
overridden at run time, without touching the source, by placing an
optional file named **`inputfiles.dat`** in the working directory:

```
control = my_control_file.dat
mesh    = mesh_v2.dat
observe = observe_2024.dat
initial = resistivity_block_iter000_restart.dat
```

- Blank lines and `#`-comments are ignored, same as in `control.dat`.
- Keys are matched case-insensitively.
- Any key you omit, or the absence of `inputfiles.dat` altogether, falls
  back to the previous hard-coded default name for that file — existing
  working directories and job scripts are completely unaffected.

### Build flag

| Flag | Preprocessor define |
|---|---|
| `INPUT_FILE_MAP=yes` (default) | `-D_INPUT_FILE_MAP` |
| `INPUT_FILE_MAP=no` | feature compiled out entirely |

```bash
make                         # feature enabled (default)
make INPUT_FILE_MAP=no       # feature compiled out entirely
```

### Source files

`io/InputFileMap.h` / `io/InputFileMap.cpp` implement the resolver
(namespace-based, `<string>`/`<map>`/`<fstream>` only); `#ifdef
_INPUT_FILE_MAP` guards its one call site in `control/AnalysisControl.cpp`
(`control.dat` → key `control`). Comment/keyword-lookup support lives
directly in `control/AnalysisControl.cpp`'s `control.dat` reader.

---

## 2. HDF5 output extension

**Added:** 2026-08-21, ported from `femtic_v4_src` (see
`ECOSYSTEM_STATUS.md` for the full porting notes, including one
pre-existing v4/v5 limitation flagged rather than silently carried over).

Optional HDF5 output is selected in two steps; the base build is completely
unaffected when no build flag is set. (Reduced from four flags on
2026-10-01, see "exchange.h5" below; run-time keywords added 2026-10-02.)

**Step 1, at build time: is the HDF5 support compiled and linked in?** The
HDF5 library has to be linked, so this part cannot move into `control.dat`.

| Build flag | Preprocessor define | Compiles in support for |
|---|---|---|
| `HDF5_OUT=yes` | `-D_HDF5_OUT` | `results_iterX.h5` (model+data+distortion, since 2026-09-13) |
| `HDF5_JAC=yes` | `-D_HDF5_JAC` | `exchange.h5` (Jacobian + roughening matrix + mesh, one file; since 2026-10-01; executable suffix `_exchange`) |

**Step 2, at run time (since 2026-10-02): is the file written in this run?**
That is now decided in `control.dat`, no longer by the build:

| `control.dat` keyword | Needs build flag | Effect when set |
|---|---|---|
| `ACTIVATE_HDF5_RESULTS` | `HDF5_OUT=yes` | write `results_iterX.h5` every iteration |
| `ACTIVATE_HDF5_EXCHANGE` | `HDF5_JAC=yes` | write `exchange.h5` (last scheduled iteration or on convergence, as below) |

```
# control.dat (any position; the order of keywords does not matter)
ACTIVATE_HDF5_RESULTS            # keyword alone switches the feature on
ACTIVATE_HDF5_EXCHANGE 1         # explicit 1 (on) or 0 (off) is also accepted,
                                 # on the same line or the next one
```

- Both keywords are optional and default to **off**. A binary built with
  HDF5 support therefore writes no `.h5` file unless `control.dat` asks for
  it. (Before 2026-10-02 a build with `HDF5_OUT=yes` / `HDF5_JAC=yes` wrote
  the files in every run; existing `control.dat` files need the keyword
  added to keep that behaviour.)
- Any value other than `0` or `1` is an error (`exit(1)`, message in the log).
- Asking for a feature the executable was **not** built with
  (`ACTIVATE_HDF5_RESULTS` without `HDF5_OUT=yes`, `ACTIVATE_HDF5_EXCHANGE`
  without `HDF5_JAC=yes`) stops the run at start-up with an error naming the
  missing make flag, on the console (PE 0) and in the log, rather than
  silently producing no file. `ACTIVATE_HDF5_...  0` is always accepted.
- The start-up log lists both outputs as ON/OFF, and says "compiled in" or
  "not compiled in" for each, e.g.
  `# HDF5 exchange output (exchange.h5) : OFF (compiled in; add ACTIVATE_HDF5_EXCHANGE to control.dat to enable).`
- The switches are read by every PE from the same `control.dat`, so the
  collective MPI calls inside the gated code (the calculated-value gather for
  `results_iterX.h5`, the Jacobian assembly for `exchange.h5`) are entered by
  all PEs or by none.
- Implementation: `AnalysisControl::isHDF5ResultsActive()` /
  `isHDF5ExchangeActive()` (`control/AnalysisControl.h`, parsed in
  `inputControlData()`), used in `AnalysisControl::run()` (the gather and
  `outputResultsToHDF5()`; the `assembleAndWriteJacobianToHDF5()` call and its
  log notes) and in the four `writeJacobianHDF5ThisIter` definitions in
  `AnalysisControlOCCAMLineSearch.cpp`. The `#ifdef _HDF5_*` guards are
  unchanged and still keep all HDF5 code out of builds without support.

```bash
make HDF5_OUT=yes
make HDF5_OUT=yes HDF5_JAC=yes
make HDF5_JAC=yes
```

The Makefile is now named `Makefile_hdf5`: build with
`make -f Makefile_hdf5 HDF5_JAC=yes ...` (or copy/symlink it to `Makefile`).

`HDF5_ROUGH=yes` / `HDF5_MESH=yes` no longer exist; passing either prints a
Makefile warning and enables `HDF5_JAC` instead.

### `exchange.h5`: jacobian + rough + mesh in one file (2026-10-01)

`jacobian.h5`, `rough.h5` and `mesh.h5` are replaced by a single
`exchange.h5`, written by `outputJacobianToHDF5()` exactly when the
Jacobian is written. Since 2026-10-01 that is only (a) at the last
iteration for which sensitivity is computed, `iter == ITERATION_NUM_MAX - 1`,
or (b) at an early converged iteration (`convergenceFlag ==
INVERSIN_CONVERGED`); previously it was rewritten at every accepted
iteration. If the retrial limit is exhausted at that iteration, or the
data-fit-cooling early exit triggers, no exchange.h5 is written at all
(a log note says so). Older entries below that mention
`jacobian.h5`, `rough.h5` or `mesh.h5` describe the history of the same
writers; their content now lives in the corresponding group of `exchange.h5`.

```
/metadata                attrs: iterNum, exchangeVersion (=1)
/jacobian/metadata       attrs: iterNum, nData, nModel, weighted (=1)
/jacobian/values         double[nData][nModel]   J = Cd^{-1/2} dF/dm
/jacobian/data_errors    double[nData]
/rough/metadata          attrs: nRows, nNonZeros, format="CSR"
/rough/row_ptr|col_ind|values
/mesh/metadata           attrs: meshType, nNodes, nElem, nNodesPerElem,
                                neighborFormat [, nNeighborElem]
/mesh/node_coords|elem_nodes|neighbor_elements [|neighbor_face_ptr]
```

Changes versus the four-flag layout:

- `outputRougheningMatrixToHDF5()` and `outputMeshToHDF5()` are gone; the
  mesh (via `AnalysisControl::getPointerOfMeshData()`) and the roughening
  matrix (new `ResistivityBlock::getRougheningMatrix()`) are read from
  memory when the Jacobian is written. The one-time calls in
  `AnalysisControl::run()` (after `callInputMeshData()`) and
  `ResistivityBlock::calcRougheningMatrix()` were removed.
- `HDF5_JAC=yes` alone now builds: the results-file writer in
  `OutputHDF5.cpp` is gated by `_HDF5_OUT`, and the mesh/CSR accessors
  needed for the dump are enabled by `_HDF5_JAC` too. (A `_HDF5_JAC`-only
  build did not compile before, because those parts were not gated.)
- Mesh and roughening matrix are no longer written at start-up. A run that
  never reaches a Jacobian write (e.g. ends before the first accepted
  iteration with sensitivity) produces no mesh/rough file at all.
- File size: mesh and roughening matrix are rewritten with every Jacobian
  write (small compared with the dense Jacobian).
- `python/femtic_hdf5_readers.py`: new `read_exchange_hdf5()`.

HDF5 root auto-detection (first match wins): `HDF5_ROOT`, `HDF5_DIR`,
`HDF5HOME`, `EBROOTHDF5`, then `pkg-config` — same order as
`femtic_v4_src`/`femtic_v5_src`. `OutputHDF5.o` is only added to `OBJS`
when at least one flag is set.

The on-disk schema (groups, datasets, compound field layout for
`results_iterX.h5` -- which as of 2026-09-13 merges the former
`model_iterX.h5`/`data_iterX.h5` pair plus a `/distortion` group -- and for
`exchange.h5` (formerly `jacobian.h5`, `rough.h5`, `mesh.h5`), and the Python snippets to read them)
is **identical** to `femtic_v4_src`'s — this port reused `OutputHDF5.{h,cpp}`
verbatim, since v2.7, like v4, has a single isotropic `ResistivityBlock`
rather than v5's isotropic/anisotropic split. See
[`../femtic_v4_src/README.md`](../femtic_v4_src/README.md#1-hdf5-output-extension)
for the full dataset-by-dataset reference and Python read examples; only
the call sites differ (v2.7's `run()` and `calcRougheningMatrix()` live in
`control/AnalysisControl.cpp` and `model/ResistivityBlock.cpp`
respectively, versus v4's single `AnalysisControl.cpp`/
`ResistivityBlock.cpp` at the source-tree root).

**`jacobian.h5` is a fixed filename, written once (2026-08-30):** Femtic
Jacobians can be very large, so this file is no longer written
per-iteration. It is only written for the deterministic last iteration at
which the Jacobian is computed (`iter == ITERATION_NUM_MAX - 1`, since
sensitivity is never computed for the true final iteration by design), and
is truncated/overwritten on each write, so it always ends up holding
exactly that last iteration's Jacobian. If the inversion converges before
reaching `ITERATION_NUM_MAX - 1`, no `jacobian.h5` is written for that
run — a log note is printed in this case.

### `_OCCAM`, `_LCurve`, and `_ABIC` trade-off-parameter search variants

Unlike the base `InversionGaussNewtonDataSpace.cpp`/
`InversionGaussNewtonModelSpace.cpp` (which have direct v4/v5 counterparts
this port was modeled on), v2.7's `_OCCAM`, `_LCurve`, and `_ABIC`
inversion variants (`inversion/InversionGaussNewtonDataSpace_OCCAM.cpp`,
`InversionGaussNewtonDataSpaceLCurve.cpp`,
`InversionGaussNewtonDataSpace_ABIC.cpp`) search over many trade-off-
parameter candidates per outer iteration — each candidate triggers its own
`inversionCalculation()` call — before settling on one. Each of these three
files does have a `jacobian.h5` hook (ported 2026-08-22), and as of
2026-08-30 all of their many per-candidate calls (routed through
`control/AnalysisControlOCCAMLineSearch.cpp` for OCCAM, and directly
through `control/AnalysisControl.cpp` for ABIC/L-curve) share the same
"only on the last outer iteration" flag as the base Gauss-Newton path.
During that final outer iteration, `jacobian.h5` gets harmlessly
overwritten by each trial candidate in turn and ends up holding whichever
one was evaluated last — there is no way to single out "the winning
candidate's Jacobian" without deeper changes to each search algorithm, so
this is a known imprecision for these three variants specifically (not
present in the plain Gauss-Newton `TO_Fixed` path, where every call
corresponds to the actual accepted update).

### Fixed: MPI_Allreduce deadlock in `model_iterX.h5` sensitivity output (2026-09-09)

Same root cause and fix as `femtic_v4_src`/`femtic_v5_src` (this code was
reused verbatim, so it carried the same bug): `Inversion::getSensitivityScalarValuesReduced()`
performs a collective `MPI_Allreduce()`, but was called only from PE 0's
`if( myProcessID == 0 && ... )` branch in `control/AnalysisControl.cpp`,
deadlocking the run the first time `model_iterX.h5` was written with
sensitivity data (any iteration where `doesCalculateSensitivity(iter)` is
true — typically iteration 1). Because PE 0 never returned from that call,
`outputDataToHDF5()` — called right after, in the same branch — was never
reached, so `data_iterX.h5` was never written even though
`model_iterX.h5`, `rough.h5`, and `mesh.h5` were.

**Fix:** the reduction now runs on every PE, unconditionally on
`myProcessID`, immediately before the `myProcessID == 0` block, gated only
by `doesCalculateSensitivity(iter)`. Only PE 0's copy is passed into
`outputModelToHDF5()` (signature changed from `(iterNum, const Inversion*)`
to `(iterNum, const double* sensitivityScalarValuesReduced)`); every PE
frees its own copy afterward. Files touched: `io/OutputHDF5.h`,
`io/OutputHDF5.cpp`, `control/AnalysisControl.cpp` (comment-only
clarification in `inversion/Inversion.cpp`).

### Fixed: MPI_Gatherv deadlock in `jacobian.h5` output, ModelSpace path (2026-09-09)

Same root cause as `femtic_v4_src`/`femtic_v5_src` — this code was ported
verbatim and carried the same bug, which the porting comment at the time
had flagged as a known limitation without fixing: in
`inversion/InversionGaussNewtonModelSpace.cpp`, the Jacobian-output block
— including an `MPI_Gatherv()` call gathering each PE's per-datum
error/SD vector — was nested entirely inside the
`if( myProcessID == 0 ){ ... }` branch, so only PE 0 ever called the
collective `MPI_Gatherv`, deadlocking the run whenever
`numProcessTotal > 1` and `writeJacobianHDF5` became true (the
deterministic last iteration at which the Jacobian is computed). Unlike
the v4/v5 versions of this bug, this tree's `numDataLocal`/`displacements`
arrays were already freed at the correct, later point, so there was no
accompanying use-after-free here — only the deadlock.
(`inversion/InversionGaussNewtonDataSpace.cpp` already performed this
gather correctly and needed no change.)

**Fix:** the error-vector gather now runs collectively, on every PE,
before branching into the `myProcessID == 0`-only block, gated only on
`writeJacobianHDF5`. Only PE 0's result (`errVecTotalForJac`) is kept and
used later, inside the PE-0-only block, to call `outputJacobianToHDF5()`.

**Files touched:** `inversion/InversionGaussNewtonModelSpace.cpp`.

### Fixed: `make` silently reuses stale objects across HDF5_* flag changes (2026-09-10)

Same root cause and fix as `femtic_v4_src`/`femtic_v5_src` (see the shared
`ECOSYSTEM_STATUS.md` for the full writeup): this tree's `-MMD`/`-MP`
generated dependency files track header changes but not `-D_HDF5_*`
command-line macros, so switching flag combinations without `make clean`
silently relinks objects in `build/obj/` built with the old flags. A new
`.cxxflags` signature file, which every object now depends on, forces a
full rebuild whenever any tracked flag (or the `debug` goal) changes.
**Files touched:** `Makefile`.

### Fixed: jacobian.h5 never written on early convergence (2026-09-11)

Same root cause and fix as `femtic_v4_src`/`femtic_v5_src` (see the
shared `ECOSYSTEM_STATUS.md` for the full writeup): `jacobian.h5` was
only ever written from inside `inversionCalculation()`, called at the
deterministic scheduled iteration `ITERATION_NUM_MAX - 1` — but that
function is never called for the iteration at which convergence is
detected (a converged model needs no further update), so early
convergence silently skipped the Jacobian output entirely, even though
the needed sensitivity data had already been computed by
`calcForwardComputation(iter)` and was still on disk.

**Fix:** extracted the Jacobian assembly-and-write logic into one shared
function, `Inversion::assembleAndWriteJacobianToHDF5(iterNum)`, called
from `InversionGaussNewtonModelSpace::inversionCalculation()`, **both**
of `InversionGaussNewtonDataSpace`'s algorithm variants
(`inversionCalculationByNewMethod()` and
`inversionCalculationByNewMethodUsingInvRTRMatrix()` — this tree, unlike
v4/v5, already threaded `writeJacobianHDF5` through to the latter, so
both needed updating), and, new, from `AnalysisControl::run()`'s
early-convergence branch.

Unlike v4/v5, this tree's shutdown cleanup already called
`deleteOutOfCoreFileAll()` *before* deleting `m_ptrInversion`, so it did
not have the null-pointer-after-delete bug fixed in those two trees.

**Files touched:** `inversion/Inversion.h`, `inversion/Inversion.cpp`,
`inversion/InversionGaussNewtonModelSpace.cpp`,
`inversion/InversionGaussNewtonDataSpace.cpp`,
`control/AnalysisControl.cpp`.

### Fixed: jacobian.h5 also never written when the retrial limit is reached (2026-09-12)

Same root cause and fix as `femtic_v4_src`/`femtic_v5_src` (see the
shared `ECOSYSTEM_STATUS.md` for the full writeup): reaching the
cutback/retrial limit (`"# Reach maximum retrial number."`) in the main
Gauss-Newton loop (`control/AnalysisControl.cpp`, around line 1102) is a
third termination path, distinct from `INVERSIN_CONVERGED` and "reach
max iteration number", that broke out of the iteration loop before any
per-iteration output — including the Jacobian check. Fixed by calling
`assembleAndWriteJacobianToHDF5(iter)` at that point, gated on
`doesCalculateSensitivity(iter)`, mirroring the `INVERSIN_CONVERGED`
branch fix. `model.h5`/`data.h5`/resistivity output are still not
written for that iteration — only the Jacobian gap was closed.

**Not addressed:** this tree's OCCAM/L-curve/ABIC trade-off-parameter
search code (see the "known imprecision" note elsewhere in this file)
has its own, separate `"# Reach maximum retrial number."` handling
(around lines 1539 and 1862), with a substantially different control
flow that sets `m_leavingABIC = true` rather than breaking the main loop
directly. Not touched by this fix.

**Files touched:** `control/AnalysisControl.cpp`.

### Fixed: jacobian.h5 also never written on the DATA_FIT_COOLING early-exit (2026-09-12)

Found while auditing this tree's main iteration loop for *every*
outer-loop exit point (in response to the user asking whether any other
termination path remained uncovered). `femtic_dabic_v2.7_src` has a
fourth exit, exclusive to this tree (`m_typeOfTradeOffParam ==
AnalysisControl::TO_DATA_FIT_COOLING` combined with
`m_inversionMethod == Inversion::DATA_FIT_COOLING_DATA_SPECE` — see
`isDataFitCoolingMode()`): when the data-fit-cooling procedure can't find
an acceptable full-step alpha, `m_stopAfterDataFitCooling` is set and the
main loop breaks with `"# Stop inversion loop because the selected
full-step cooling response was not reproducible."`, *before* the retrial
check and all per-iteration output — the same shape of gap as the
retrial-exhaustion fix above.

**Fix:** identical pattern — call `assembleAndWriteJacobianToHDF5(iter)`
at that break point, gated on `doesCalculateSensitivity(iter)`.

**Not affected:** this whole code path only executes when
`isDataFitCoolingMode()` is true, i.e. only for that specific
`TypeOfTradeOffParam`/`InversionMethod` combination — standard
Gauss-Newton (`TO_Fixed`) runs, which is what this conversation's actual
reported issues have all been about, never reach this branch at all, so
this fix has no effect on the standard path.

### Found, NOW FIXED (2026-09-12, superseding the entry below): the trade-off-parameter-search modes

Following up on the "found, NOT fixed" entry below at the user's
request, all six `TypeOfTradeOffParam`/`InversionMethod` combinations
were traced in full: `TO_Fixed`, `TO_ABIC_LS` (→
`InversionGaussNewtonDataSpace_ABIC`), `TO_OCCAM_LS` (→
`InversionGaussNewtonDataSpace_OCCAM`), `TO_LINEAR_LCURVE` /
`TO_NONLINEAR_LCURVE` (both → `InversionGaussNewtonDataSpaceLCurve`, plus
a companion plain `InversionGaussNewtonDataSpace` object), and
`TO_DATA_FIT_COOLING` (→ plain `InversionGaussNewtonDataSpace`).

**Already correct, no fix needed:** `TO_Fixed`, `TO_ABIC_LS`, and
`TO_OCCAM_LS` all consistently pass `writeJacobianHDF5ThisIter` into
every `inversionCalculation()` call in their respective code
(`AnalysisControl.cpp`, the ABIC bracketing/root-finding loop, and
`AnalysisControlOCCAMLineSearch.cpp` respectively), and the underlying
classes (`InversionGaussNewtonDataSpace_ABIC`,
`InversionGaussNewtonDataSpace_OCCAM`) both already implement the
Jacobian output block with the *correct*, non-buggy MPI pattern (never
touched by the earlier deadlock/use-after-free fixes because those two
classes didn't exist yet at that point, or were out of scope — this
audit is what found them). `TO_LINEAR_LCURVE` was also already correct:
its diagnostic probe call on `m_ptrInversion` carries the flag, and its
companion "final apply" call intentionally doesn't need to.

**Two genuine gaps found and fixed:**

1. **`TO_NONLINEAR_LCURVE`: never wrote jacobian.h5 at all, on any
   iteration.** `runNonlinearLCurveDiagnostics()` never calls
   `inversionCalculation()` on the real `m_ptrInversion` object; the
   *only* `inversionCalculation()` reached in this mode is a companion
   call on the plain `m_ptrInversiondataspace` object, made with no
   argument (defaulting to `false`) — unconditionally, regardless of
   iteration. **Fix:** that companion call now receives
   `writeJacobianHDF5ThisIter` (in both the "Difference Filter" and
   "Laplacian Filter" branches).

2. **`TO_DATA_FIT_COOLING`: never wrote jacobian.h5 at all, on any
   iteration.** `runInitialDataFitCoolingBracket()`/
   `runPersistentDataFitCoolingAlpha()` evaluate multiple candidate alpha
   values per outer iteration via `runDataFitCoolingTrial()`, which calls
   `inversionCalculation()` with no argument *by design* — see its
   "without sensitivity work" comment: trials deliberately reuse the
   sensitivity already computed by `calcForwardComputation(iter)` rather
   than recomputing it per trial, so there is no single "the" trial call
   to attach the flag to. **Fix:** call
   `m_ptrInversion->assembleAndWriteJacobianToHDF5()` directly, once,
   right after the bracket/alpha search call, since every trial within
   one outer iteration shares that same fixed sensitivity data regardless
   of which alpha is ultimately selected.

**A third, related gap, only visible once the above two were traced
through to their conclusion:** both `TO_NONLINEAR_LCURVE` and
`TO_DATA_FIT_COOLING` can decide to *stop the whole run* partway through
an iteration (`m_stopAfterNonlinearLCurveDiagnostics` /
`m_stopAfterDataFitCooling`), at the "fifth exit" identified below —
and in both modes, that stop decision *skips* the fix just described
(the L-curve companion call is only made `if (!m_stopAfter...)`; the
data-fit-cooling direct write above is keyed to
`writeJacobianHDF5ThisIter`, which has nothing to do with *this*
iteration being the one that decided to stop). **Fix:** the fifth exit
itself (see below) now also does a direct
`assembleAndWriteJacobianToHDF5()` call, covering both stop reasons
uniformly, using this iteration's already-computed sensitivity data
(trials/diagnostics reuse it without recomputing it, so it still
correctly reflects the state being stopped at). A write here can
harmlessly coincide with one from the ordinary schedule (jacobian.h5 has
a fixed filename; a redundant overwrite is wasted work, not a
correctness issue).

**Files touched:** `control/AnalysisControl.cpp`.

### Found, NOT fixed (superseded by the entry above, kept for history)

Near the very end of the same main loop (`control/AnalysisControl.cpp`,
around line 1988), there's a further exit —
`if (m_stopAfterNonlinearLCurveDiagnostics || m_stopAfterDataFitCooling)
{ ...; break; }` — reachable from the `TO_OCCAM_LS`, `TO_LINEAR_LCURVE`,
`TO_NONLINEAR_LCURVE`, and `TO_DATA_FIT_COOLING` trade-off-parameter
branches. Unlike the other four exits, this one sits *after* the block
that runs `inversionCalculation()`/`runOCCAMLineSearch()`/etc. for the
current iteration, so it's not structurally identical to the gaps fixed
above — it may already be covered by the ordinary
`writeJacobianHDF5ThisIter` mechanism in some of these branches (e.g.
`TO_LINEAR_LCURVE` explicitly calls
`m_ptrInversion->inversionCalculation(writeJacobianHDF5ThisIter)`), and
in others may not apply at all (e.g. it's unclear whether
`doesCalculateSensitivity(iter)`/the per-frequency
`calculateSensitivityMatrix()` calls happen in the same way inside
`runOCCAMLineSearch()`'s or `runNonlinearLCurveDiagnostics()`'s own
internal forward-computation cycles). Extending the same
`assembleAndWriteJacobianToHDF5()` pattern here without tracing those
functions in detail risks either a redundant/no-op call or asserting on
stale sensitivity data from a different sub-iteration than intended.
**Left unaddressed** pending confirmation this trade-off-search
functionality is actually in use; this file's existing "known
imprecision" note (above) already flags this general area as less
rigorously verified than the standard `TO_Fixed` path.

**Files touched:** `control/AnalysisControl.cpp`.

---

### CRITICAL: exit(1) on a missing sensMatFreq<N> file crashed the whole MPI job (2026-09-12)

Same root cause and fix as `femtic_v4_src`/`femtic_v5_src` (found via a
real crashed 24-rank run of `femtic_v4_src`; see the shared
`ECOSYSTEM_STATUS.md` for the full writeup): a missing/mismatched
`sensMatFreq<N>` file used to call `exit(1)`, which under `mpirun`
SIGKILLs every other rank rather than cleanly stopping the job. Fixed in
`inversion/Inversion.cpp`'s `assembleAndWriteJacobianToHDF5()`, and also
in `InversionGaussNewtonDataSpace_ABIC.cpp`, `_OCCAM.cpp`, and
`InversionGaussNewtonDataSpaceLCurve.cpp`'s own inline Jacobian blocks
(found to have the identical pre-existing pattern during the same audit
that added Jacobian support to those classes) — all now log a warning
and skip just that iteration's Jacobian dump instead of crashing.

**Files touched:** `inversion/Inversion.cpp`,
`inversion/InversionGaussNewtonDataSpace_ABIC.cpp`,
`inversion/InversionGaussNewtonDataSpace_OCCAM.cpp`,
`inversion/InversionGaussNewtonDataSpaceLCurve.cpp`.

### 2026-09-12 — SUPERSEDED: jacobian.h5 now written on every successful iteration, not at termination

This supersedes essentially every jacobian.h5-timing fix above from
"jacobian.h5 never written on early convergence" onward, with a better
design, at the user's suggestion: instead of attempting the (race-prone)
assembly separately at each special termination event -- retrial
exhaustion, `INVERSIN_CONVERGED`, the data-fit-cooling early exit, the
fifth exit near the end of the trade-off dispatch, and inside the
`TO_NONLINEAR_LCURVE`/`TO_DATA_FIT_COOLING` dispatch blocks themselves
-- `jacobian.h5` is now written directly at the point each iteration's
retrial loop *succeeds*, uniformly across every trade-off mode (this
point runs before any mode-specific dispatch). On any iteration that
does NOT succeed, nothing further is attempted: the correct artifact is
whatever was written on the last successful iteration, already on disk.
All of the special-case write attempts listed above have been removed.
See the shared `ECOSYSTEM_STATUS.md` for the full writeup, including
confirmation that every trade-off mode (`TO_Fixed`, `TO_ABIC_LS`,
`TO_OCCAM_LS`, `TO_LINEAR_LCURVE`, `TO_NONLINEAR_LCURVE`,
`TO_DATA_FIT_COOLING`) is correctly covered by this single change.

**Files touched:** `control/AnalysisControl.cpp`.

---

### 2026-10-02: HDF5 outputs switched on from `control.dat` (`ACTIVATE_HDF5_RESULTS`, `ACTIVATE_HDF5_EXCHANGE`)

The build flags `HDF5_OUT` / `HDF5_JAC` no longer decide whether
`results_iterX.h5` / `exchange.h5` are written; they only compile the
support in. Each run chooses with the new `control.dat` keywords, both
default off (details and the error behaviour in section 2 above).
Files touched: `control/AnalysisControl.h` (two enum IDs, two members, two
getters), `control/AnalysisControl.cpp` (parsing, start-up log lines, run-time
gates around the results gather/write and the exchange write),
`control/AnalysisControlGetters.cpp`, `control/AnalysisControlOCCAMLineSearch.cpp`
(`writeJacobianHDF5ThisIter` now also requires the exchange switch),
`Makefile_hdf5` (comment only), this README.
Behaviour change to be aware of: HDF5-enabled builds no longer write the
`.h5` files unless `control.dat` contains the keywords.
Verification: syntax check only (`g++ -fsyntax-only -Wall -Wextra`, stub MPI/MKL
headers) of the three `control/AnalysisControl*.cpp` files in all four
`_HDF5_OUT`/`_HDF5_JAC` combinations (no errors, warning count equal to the
base commit), plus a stand-alone test of the keyword parser on 12 `control.dat`
snippets. Not built with the real MPI/MKL/HDF5 toolchain and not run.

---

## Pristine originals

Pristine originals of every file touched by any of the above live in
`orig/` (prefixed `v2.7_`), so each change can be diffed against the
unmodified FEMTIC-DABIC v2.7 release.
