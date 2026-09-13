# Atotsugawa Fault Examples

These examples use one shared Atotsugawa Fault mesh, observation file, and
initial resistivity model. Three inversion configurations are provided:

| Case | Configuration |
| --- | --- |
| `exact_ABIC` | Exact ABIC with L2 Difference-filter regularization |
| `fixed_alpha_reference` | Fixed `alpha = 10.0` with an L2 fixed-reference-model constraint of weight `1.0` |
| `ABIC_with_distortion` | Exact ABIC with L2 Difference-filter regularization and distortion correction of weight `1.0` |

Prepare a clean run directory so generated outputs do not mix with the tracked
example files. From the repository root, run:

```bash
case_name=exact_ABIC
run_dir=/tmp/femtic-dabic-atotsugawa/${case_name}
mkdir -p "${run_dir}"
cp examples/AtotsugawaFault/shared_inputs/* "${run_dir}/"
cp examples/AtotsugawaFault/${case_name}/* "${run_dir}/"
```

Set `case_name` to `fixed_alpha_reference` or `ABIC_with_distortion` for the
other configurations. Then run the selected case:

```bash
cd "${run_dir}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mpirun -np 2 /path/to/FEMTIC-DABIC/src/femtic-dabic
```

The supplied controls run iterations 0-2. Check the log, convergence file, and
model outputs before increasing the iteration limit or changing regularization
weights.

In `fixed_alpha_reference`, `Referencemodel.dat` is intentionally identical to
the supplied initial model. This provides a reproducible fixed reference with
the same element/block mapping and positive resistivity values.

In `ABIC_with_distortion`, `distortion_iter0.dat` initializes the four
distortion-matrix differences to zero for every MT station. The final column is
zero, so the distortion parameters are adjustable during inversion.
