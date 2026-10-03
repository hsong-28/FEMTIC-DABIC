"""
femtic_hdf5_readers.py
======================

Python (3.11+) readers for the HDF5 files written by the optional HDF5
output of FEMTIC, for the three active source trees ``femtic_v4_src``,
``femtic_v5_src`` and ``femtic_dabic_v2.7_src`` (``OutputHDF5.cpp``).

Files
-----
::

    File                  C++ writer (OutputHDF5.cpp)       Flag          Written
    --------------------  --------------------------------  ------------  --------------------------
    results_iter<N>.h5    outputResultsToHDF5()             _HDF5_OUT     each iteration N > INIT
    exchange.h5           outputJacobianToHDF5()            _HDF5_JAC     whenever the Jacobian is
                                                                          written (last accepted
                                                                          iteration)

``exchange.h5`` holds three groups: ``/jacobian``, ``/rough`` (roughening
matrix, CSR) and ``/mesh``. It replaces the former ``jacobian.h5``,
``rough.h5`` and ``mesh.h5`` (merged 2026-10-01).

Quick start
-----------
>>> from femtic_hdf5_readers import read_femtic_hdf5, read_results_hdf5
>>> res = read_results_hdf5("results_iter3.h5")
>>> res.model.rho                               # (nBlocks,) Ohm.m
>>> res.model.sensitivity_volume_normalised     # (nBlocks,), None if not computed
>>> res.data_of_type("MT")                      # structured array of MT rows
>>> ex = read_exchange_hdf5("exchange.h5")      # ex.jacobian.J, ex.rough, ex.mesh
>>> obj = read_femtic_hdf5("exchange.h5")       # auto-detects file kind

Block / model indexing
----------------------
Per-block arrays (``/model/blocks``, ``/model/sensitivity/*``) have length
nBlocks and include fixed blocks (air, sea, other fixed regions); they are
indexed by blockID as in ``/model/element_block_map``. In isotropic DABIC,
the leading Jacobian columns correspond to free resistivity blocks (type
0 or 3) in ascending blockID order; ``FemticModel.model_to_block`` maps
only these columns. Any estimated distortion parameters follow them and
are not resistivity blocks. The exported ``/rough`` matrix is the
pre-degeneration nBlocks x nBlocks operator, including fixed blocks.
It is not the weighted free-parameter regularization operator used by
the solver: fixed-block terms, filter/IRLS weights, reference constraints,
regularization weights and distortion penalties must be handled separately.
Do not combine the raw export directly with J or assume matching dimensions.

``exchange.h5`` describes the Jacobian's linearization iteration, recorded
in ``/metadata`` and ``/jacobian/metadata``. It need not match the last
results file; the scheduled final iteration is forward-only. Check the
iteration attributes and run log before combining outputs. A failed export
can leave a valid but stale file from an earlier run or iteration.

When is sensitivity absent from results_iter<N>.h5?
----------------------------------------------------
``/model/sensitivity`` is written only if
``AnalysisControl::doesCalculateSensitivity(iter)``, i.e.
``iter < ITERATION_NUM_MAX`` (identical in all three trees). Results files
are written only for ``iter > ITERATION_NUM_INIT`` and only after a
successful retrial loop. For a run from i0 to N:

* ``results_iter<i0>.h5``: not written;
* ``results_iter<k>.h5``, i0 < k < N: with sensitivity;
* ``results_iter<N>.h5``: without sensitivity (forward-only final iteration);
* convergence at k < N: last file ``results_iter<k>.h5`` has sensitivity;
* retrials exhausted at k: no file for k.

For an iteration that needed retrials, ``sensitivity_raw`` /
``sensitivity_volume_normalised`` reflect only the accepted retrial's
model (fixed 2026-09-25 in the C++ writer, all three trees: the
accumulator is now re-zeroed at the start of every retrial attempt, not
just once per iteration). Files written before that fix may still hold
the sum over all retrials tried that iteration; there is no way to tell
the two cases apart from the file's contents alone.

Contributor: Volker Rath (DIAS).

Validation note: the original reader tests used an in-memory HDF5 stand-in,
not solver-written files. Validate against representative solver outputs
before production use. Legacy separate model/data file layouts are unsupported.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "femtic_hdf5_readers requires h5py (`pip install h5py`)"
    ) from exc

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None  # read_rough_hdf5 then returns the raw CSR arrays


class FemticHDF5SchemaError(RuntimeError):
    """Raised when an HDF5 file does not match the expected FEMTIC layout."""


# ---------------------------------------------------------------------------
# Constants (must match the C++ sources)
# ---------------------------------------------------------------------------

# ResistivityBlock::TypeOfResistivityBlock -> /model/blocks["type"]
BLOCK_FREE_AND_CONSTRAINED = 0
BLOCK_FIXED_AND_ISOLATED = 1
BLOCK_FIXED_AND_CONSTRAINED = 2
BLOCK_FREE_AND_ISOLATED = 3
_FIXED_BLOCK_TYPES = (BLOCK_FIXED_AND_ISOLATED, BLOCK_FIXED_AND_CONSTRAINED)

# Value written to /model/sensitivity/* for fixed blocks and for free blocks
# with |S| <= this value (OutputHDF5.cpp::writeModelGroup(), "criteria").
SENSITIVITY_FLOOR = 1.0e-20

# /data/data["datatype"] codes (FemticHDF5CalcTypes.h)
DATATYPE_NAMES = {
    0: "MT", 1: "APP_RES_AND_PHS", 2: "HTF", 3: "VTF", 4: "PT",
    5: "NMT", 6: "NMT2", 7: "NMT2_APP_RES_AND_PHS",
}

# /data/data["component"] codes per datatype (OutputHDF5.h)
COMPONENT_NAMES = {
    "MT": ("Zxx", "Zxy", "Zyx", "Zyy"),
    "NMT2": ("Zxx", "Zxy", "Zyx", "Zyy"),
    "APP_RES_AND_PHS": ("rhoXX", "rhoXY", "rhoYX", "rhoYY",
                        "phsXX", "phsXY", "phsYX", "phsYY"),
    "NMT2_APP_RES_AND_PHS": ("rhoXX", "rhoXY", "rhoYX", "rhoYY",
                             "phsXX", "phsXY", "phsYX", "phsYY"),
    "HTF": ("Txx", "Txy", "Tyx", "Tyy"),
    "VTF": ("Tzx", "Tzy"),
    "PT": ("PTxx", "PTxy", "PTyx", "PTyy"),
    "NMT": ("Yx", "Yy"),
}

# /distortion/metadata attr "type" (AnalysisControl::TypeOfDistortion)
DISTORTION_TYPE_NAMES = {
    0: "NO_DISTORTION",
    1: "ESTIMATE_DISTORTION_MATRIX_DIFFERENCE",   # param1..4 = Cxx, Cxy, Cyx, Cyy
    2: "ESTIMATE_GAINS_AND_ROTATIONS",            # param1..4 = ExGain, EyGain, ExRot, EyRot (deg)
    3: "ESTIMATE_GAINS_ONLY",                     # param1..2 = ExGain, EyGain
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _require(g: h5py.File | h5py.Group, *paths: str, context: str = "") -> None:
    """Raise FemticHDF5SchemaError listing missing paths, instead of a bare KeyError."""
    missing = [p for p in paths if p not in g]
    if missing:
        raise FemticHDF5SchemaError(
            f"{context or g.name}: missing expected path(s) {missing}. "
            f"Keys present: {list(g.keys())}. Run inspect_file() on this file."
        )


def _attrs(obj: h5py.File | h5py.Group | h5py.Dataset) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.attrs.items():
        v = v.item() if hasattr(v, "item") else v
        if isinstance(v, bytes):
            v = v.decode("ascii", errors="replace").rstrip("\x00")
        out[k] = v
    return out


def inspect_file(path: str | Path, max_depth: int = 4) -> str:
    """Tree dump of an HDF5 file: groups, datasets (shape, dtype), attributes."""
    lines: list[str] = []

    def _visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        depth = name.count("/")
        if depth > max_depth:
            return
        indent = "  " * (depth + 1)
        if isinstance(obj, h5py.Dataset):
            lines.append(f"{indent}{name.split('/')[-1]}  <dataset {obj.shape} {obj.dtype}>")
        else:
            lines.append(f"{indent}{name.split('/')[-1]}/")
        for k, v in obj.attrs.items():
            lines.append(f"{indent}  @{k} = {v!r}")

    with h5py.File(path, "r") as f:
        lines.append(Path(path).name)
        for k, v in f.attrs.items():
            lines.append(f"  @{k} = {v!r}")
        f.visititems(_visit)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# results_iter<N>.h5  -- /model, /data, [/distortion]
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FemticModel:
    """
    The ``/model`` group of ``results_iter<N>.h5``.

    All per-block arrays have length nBlocks and are indexed by blockID
    (fixed blocks included). Sensitivity entries of fixed blocks hold
    ``SENSITIVITY_FLOOR``; use ``free_mask`` to exclude them.
    """

    rho: np.ndarray                      # (nBlocks,) Ohm.m
    rho_min: np.ndarray                  # (nBlocks,)
    rho_max: np.ndarray                  # (nBlocks,)
    weight: np.ndarray                   # (nBlocks,) roughening weight (1.0 placeholder, v5 anisotropic)
    block_type: np.ndarray               # (nBlocks,) BLOCK_* codes
    element_block_map: np.ndarray        # (nElem,) blockID per element
    node_coords: np.ndarray              # (nNodes, 3)
    elem_nodes: np.ndarray               # (nElem, nNodesPerElem)
    sensitivity_raw: np.ndarray | None             # (nBlocks,) sum|J[:,m]|, MPI-reduced
    sensitivity_volume_normalised: np.ndarray | None   # (nBlocks,) raw / block volume (m^-3)
    sensitivity_computed: bool           # True iff /model/sensitivity exists in the file
    iteration: int | None
    attrs: dict[str, Any]                # /model/metadata attributes
    raw_blocks: np.ndarray               # full /model/blocks structured array
    # femtic_v5_src anisotropic runs only (None otherwise)
    rho_xx: np.ndarray | None = None
    rho_yy: np.ndarray | None = None
    rho_zz: np.ndarray | None = None
    strike: np.ndarray | None = None     # degrees
    dip: np.ndarray | None = None        # degrees
    slant: np.ndarray | None = None      # degrees
    is_anisotropic: bool | None = None   # /model/metadata @anisotropic (v5 only)

    @property
    def n_blocks(self) -> int:
        return self.rho.shape[0]

    @property
    def free_mask(self) -> np.ndarray:
        """(nBlocks,) bool, True for free (inverted-for) blocks."""
        return ~np.isin(self.block_type, _FIXED_BLOCK_TYPES)

    @property
    def model_to_block(self) -> np.ndarray | None:
        """
        (nFreeResistivity,) blockID of each free resistivity parameter,
        in ascending blockID order. These are the leading isotropic
        Jacobian columns, not the appended distortion columns or the
        all-block roughening indices. None for anisotropic runs.
        """
        if self.is_anisotropic:
            return None
        return np.flatnonzero(self.free_mask)

    def sensitivity_per_element(self, volume_normalised: bool = True) -> np.ndarray | None:
        """Sensitivity mapped to mesh elements via element_block_map (None if absent)."""
        s = self.sensitivity_volume_normalised if volume_normalised else self.sensitivity_raw
        return None if s is None else s[self.element_block_map]


@dataclasses.dataclass
class FemticResults:
    """Contents of one ``results_iter<N>.h5`` file."""

    model: FemticModel
    data: np.ndarray                     # structured (nRows,): freq, datatype, site_id,
                                          # site_x, site_y, site_z, re_val, im_val,
                                          # re_err, im_err, cal_re, cal_im, component
    data_attrs: dict[str, Any]            # nRows, iterNum, nMissingCalc
    distortion: np.ndarray | None         # structured (nSites,): site_id, param1..4, isFixed
    distortion_attrs: dict[str, Any] | None  # iterNum, nRows, type; None without /distortion
    iteration: int | None

    @property
    def sensitivity_computed(self) -> bool:
        return self.model.sensitivity_computed

    @property
    def distortion_type(self) -> str | None:
        if self.distortion_attrs is None:
            return None
        return DISTORTION_TYPE_NAMES.get(int(self.distortion_attrs.get("type", -1)))

    def data_of_type(self, datatype: int | str) -> np.ndarray:
        """Rows of ``data`` for one datatype, given as code or name (e.g. "MT")."""
        if isinstance(datatype, str):
            inv = {v: k for k, v in DATATYPE_NAMES.items()}
            if datatype not in inv:
                raise KeyError(f"unknown datatype {datatype!r}; known: {sorted(inv)}")
            datatype = inv[datatype]
        return self.data[self.data["datatype"] == datatype]


def _read_model_group(g: h5py.Group, context: str, sens_fill: float | None) -> FemticModel:
    _require(g, "metadata", "element_block_map", "blocks",
             "mesh/node_coords", "mesh/elem_nodes", context=context)

    blocks = g["blocks"][()]
    names = blocks.dtype.names or ()
    need = ("resistivity", "rho_min", "rho_max", "weight", "type")
    if not set(need) <= set(names):
        raise FemticHDF5SchemaError(
            f"{context}: blocks lacks field(s) {sorted(set(need) - set(names))}; "
            f"fields present: {names}."
        )
    has_tensor = "rho_xx" in names

    meta = _attrs(g["metadata"])
    n_blocks = blocks.shape[0]
    if "nBlocks" in meta and int(meta["nBlocks"]) != n_blocks:
        raise FemticHDF5SchemaError(
            f"{context}: metadata nBlocks={meta['nBlocks']} but blocks has {n_blocks} rows."
        )

    eb_map = g["element_block_map"][()]
    if eb_map.size and (eb_map.min() < 0 or eb_map.max() >= n_blocks):
        raise FemticHDF5SchemaError(
            f"{context}: element_block_map values outside [0, {n_blocks})."
        )

    sens_raw = g["sensitivity/raw"][()] if "sensitivity/raw" in g else None
    sens_vol = (g["sensitivity/volume_normalised"][()]
                if "sensitivity/volume_normalised" in g else None)
    sensitivity_computed = sens_raw is not None or sens_vol is not None

    # Sensitivity is indexed by blockID over ALL blocks, exactly like
    # /model/blocks. Any other length means a writer/reader mismatch.
    for name, arr in (("sensitivity/raw", sens_raw),
                      ("sensitivity/volume_normalised", sens_vol)):
        if arr is not None and arr.shape != (n_blocks,):
            raise FemticHDF5SchemaError(
                f"{context}: {name} has shape {arr.shape}, expected ({n_blocks},) "
                f"= number of rows in blocks."
            )

    if not sensitivity_computed and sens_fill is not None:
        sens_raw = np.full(n_blocks, sens_fill, dtype=float)
        sens_vol = np.full(n_blocks, sens_fill, dtype=float)

    is_aniso = meta.get("anisotropic")
    return FemticModel(
        rho=blocks["resistivity"],
        rho_min=blocks["rho_min"],
        rho_max=blocks["rho_max"],
        weight=blocks["weight"],
        block_type=blocks["type"],
        element_block_map=eb_map,
        node_coords=g["mesh/node_coords"][()],
        elem_nodes=g["mesh/elem_nodes"][()],
        sensitivity_raw=sens_raw,
        sensitivity_volume_normalised=sens_vol,
        sensitivity_computed=sensitivity_computed,
        iteration=meta.get("iterNum"),
        attrs=meta,
        raw_blocks=blocks,
        rho_xx=blocks["rho_xx"] if has_tensor else None,
        rho_yy=blocks["rho_yy"] if has_tensor else None,
        rho_zz=blocks["rho_zz"] if has_tensor else None,
        strike=blocks["strike"] if has_tensor else None,
        dip=blocks["dip"] if has_tensor else None,
        slant=blocks["slant"] if has_tensor else None,
        is_anisotropic=None if is_aniso is None else bool(is_aniso),
    )


def read_results_hdf5(path: str | Path, sens_fill: float | None = None) -> FemticResults:
    """
    Read ``results_iter<N>.h5`` written by ``outputResultsToHDF5()``.

    sens_fill: if not None and the file has no /model/sensitivity (final
        iteration), return (nBlocks,) sensitivity arrays filled with this
        value (e.g. ``np.nan``) instead of None.
    """
    with h5py.File(path, "r") as f:
        _require(f, "model", "data/data", context=str(path))

        model = _read_model_group(f["model"], f"{path}:/model", sens_fill)

        data = f["data/data"][()]
        data_attrs = _attrs(f["data/metadata"]) if "data/metadata" in f else {}
        if "nRows" in data_attrs and int(data_attrs["nRows"]) != data.shape[0]:
            raise FemticHDF5SchemaError(
                f"{path}: /data/metadata nRows={data_attrs['nRows']} but "
                f"/data/data has {data.shape[0]} rows."
            )

        distortion = None
        distortion_attrs = None
        if "distortion" in f:
            distortion_attrs = (_attrs(f["distortion/metadata"])
                                if "distortion/metadata" in f else {})
            if "distortion/params" in f:
                distortion = f["distortion/params"][()]

    iteration = model.iteration if model.iteration is not None else data_attrs.get("iterNum")
    return FemticResults(
        model=model,
        data=data,
        data_attrs=data_attrs,
        distortion=distortion,
        distortion_attrs=distortion_attrs,
        iteration=iteration,
    )


# ---------------------------------------------------------------------------
# exchange.h5  (/jacobian, /rough, /mesh)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FemticJacobian:
    """
    ``/jacobian`` of ``exchange.h5``: the SD-weighted Jacobian
    J = Cd^{-1/2} dF/dm at the exported linearization iteration.

    Rows follow the inversion's global data ordering (PE-gathered), which
    is NOT the row order of ``/data/data`` in results_iter<N>.h5. Columns
    contain free resistivity modelIDs followed by any estimated distortion
    parameters. ``FemticModel.model_to_block`` maps only the resistivity part.
    """

    J: np.ndarray                 # (nData, nModel), SD-weighted
    data_errors: np.ndarray       # (nData,) SD of each row; squared = diag(Cd)
    iteration: int | None
    n_data: int
    n_model: int
    weighted: bool
    attrs: dict[str, Any]

    def unweighted(self) -> np.ndarray:
        """dF/dm = J * SD (row-wise)."""
        return self.J * self.data_errors[:, np.newaxis]


def _read_jacobian_group(g: h5py.Group, context: str) -> FemticJacobian:
    _require(g, "metadata", "values", "data_errors", context=context)
    J = g["values"][()]
    err = g["data_errors"][()]
    meta = _attrs(g["metadata"])

    n_data = int(meta.get("nData", J.shape[0]))
    n_model = int(meta.get("nModel", J.shape[1]))
    if J.shape != (n_data, n_model) or err.shape != (n_data,):
        raise FemticHDF5SchemaError(
            f"{context}: values {J.shape} / data_errors {err.shape} inconsistent "
            f"with metadata nData={n_data}, nModel={n_model}."
        )
    return FemticJacobian(
        J=J,
        data_errors=err,
        iteration=meta.get("iterNum"),
        n_data=n_data,
        n_model=n_model,
        weighted=bool(meta.get("weighted", 1)),
        attrs=meta,
    )


def _read_rough_group(g: h5py.Group, context: str):
    _require(g, "metadata", "row_ptr", "col_ind", "values", context=context)
    row_ptr = g["row_ptr"][()]
    col_ind = g["col_ind"][()]
    values = g["values"][()]
    meta = _attrs(g["metadata"])

    n = int(meta.get("nRows", len(row_ptr) - 1))
    nnz = int(meta.get("nNonZeros", len(values)))
    if len(row_ptr) != n + 1 or len(col_ind) != nnz or len(values) != nnz:
        raise FemticHDF5SchemaError(
            f"{context}: CSR arrays (row_ptr {len(row_ptr)}, col_ind {len(col_ind)}, "
            f"values {len(values)}) inconsistent with nRows={n}, nNonZeros={nnz}."
        )
    if sp is not None:
        return sp.csr_matrix((values, col_ind, row_ptr), shape=(n, n))
    return {"row_ptr": row_ptr, "col_ind": col_ind, "values": values, "shape": (n, n)}


@dataclasses.dataclass
class FemticMesh:
    """
    ``/mesh`` of ``exchange.h5`` (arrays as parsed from mesh.dat).

    neighbor_format 0 (HEXA, TETRA): dense ``neighbor_elements``
    (nElem, nNeighborElem). neighbor_format 1 (non-conforming hexa): CSR
    over (element, face) with ``neighbor_face_ptr`` (nElem*6+1,) and flat
    ``neighbor_elements``.
    """

    node_coords: np.ndarray            # (nNodes, 3)
    elem_nodes: np.ndarray             # (nElem, nNodesPerElem)
    neighbor_format: int               # 0 = dense, 1 = CSR
    neighbor_elements: np.ndarray
    neighbor_face_ptr: np.ndarray | None
    mesh_type: int | None              # /mesh/metadata @meshType
    attrs: dict[str, Any]

    def neighbors_of(self, elem: int) -> np.ndarray:
        """Neighbor element IDs of ``elem`` (negative entries = no neighbor, removed)."""
        if self.neighbor_format == 0:
            row = self.neighbor_elements[elem]
            return row[row >= 0]
        a, b = self.neighbor_face_ptr[elem * 6], self.neighbor_face_ptr[(elem + 1) * 6]
        return self.neighbor_elements[a:b]


def _read_mesh_group(g: h5py.Group, context: str) -> FemticMesh:
    _require(g, "metadata", "node_coords", "elem_nodes", "neighbor_elements",
             context=context)
    meta = _attrs(g["metadata"])
    fmt = int(meta.get("neighborFormat", 0))
    face_ptr = g["neighbor_face_ptr"][()] if "neighbor_face_ptr" in g else None
    if fmt == 1 and face_ptr is None:
        raise FemticHDF5SchemaError(
            f"{context}: neighborFormat=1 (CSR) but neighbor_face_ptr is missing."
        )
    return FemticMesh(
        node_coords=g["node_coords"][()],
        elem_nodes=g["elem_nodes"][()],
        neighbor_format=fmt,
        neighbor_elements=g["neighbor_elements"][()],
        neighbor_face_ptr=face_ptr,
        mesh_type=meta.get("meshType"),
        attrs=meta,
    )


@dataclasses.dataclass
class FemticExchange:
    """Everything in ``exchange.h5``."""

    jacobian: FemticJacobian
    rough: Any                         # scipy csr_matrix, or dict of raw CSR arrays
    mesh: FemticMesh
    iteration: int | None              # /metadata @iterNum
    attrs: dict[str, Any]              # /metadata attributes


def _open_group(f: h5py.File, name: str, path: str | Path) -> h5py.Group:
    _require(f, name, context=str(path))
    return f[name]


def read_exchange_hdf5(path: str | Path) -> FemticExchange:
    """Read all three groups of ``exchange.h5`` (``outputJacobianToHDF5()``)."""
    with h5py.File(path, "r") as f:
        _require(f, "jacobian", "rough", "mesh", context=str(path))
        meta = _attrs(f["metadata"]) if "metadata" in f else {}
        return FemticExchange(
            jacobian=_read_jacobian_group(f["jacobian"], f"{path}:/jacobian"),
            rough=_read_rough_group(f["rough"], f"{path}:/rough"),
            mesh=_read_mesh_group(f["mesh"], f"{path}:/mesh"),
            iteration=meta.get("iterNum"),
            attrs=meta,
        )


def read_jacobian_hdf5(path: str | Path) -> FemticJacobian:
    """Read only ``/jacobian`` of ``exchange.h5``."""
    with h5py.File(path, "r") as f:
        return _read_jacobian_group(_open_group(f, "jacobian", path), f"{path}:/jacobian")


def read_rough_hdf5(path: str | Path):
    """
    Read only ``/rough`` of ``exchange.h5`` (roughening matrix R,
    nBlocks x nBlocks, CSR, before fixed-block degeneration in DABIC).
    Returns a ``scipy.sparse.csr_matrix``, or a dict of the raw CSR arrays
    if scipy is not installed. This is not the solver's weighted model-space
    regularization operator.
    """
    with h5py.File(path, "r") as f:
        return _read_rough_group(_open_group(f, "rough", path), f"{path}:/rough")


def read_mesh_hdf5(path: str | Path) -> FemticMesh:
    """Read only ``/mesh`` of ``exchange.h5``."""
    with h5py.File(path, "r") as f:
        return _read_mesh_group(_open_group(f, "mesh", path), f"{path}:/mesh")


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def read_femtic_hdf5(path: str | Path, sens_fill: float | None = None):
    """
    Detect the file kind from its contents (not its name) and return
    FemticResults (results_iter<N>.h5) or FemticExchange (exchange.h5).
    """
    with h5py.File(path, "r") as f:
        keys = set(f.keys())

    if "model" in keys and "data" in keys:
        return read_results_hdf5(path, sens_fill=sens_fill)
    if {"jacobian", "rough", "mesh"} <= keys:
        return read_exchange_hdf5(path)
    raise FemticHDF5SchemaError(
        f"{path}: not a FEMTIC HDF5 file (top-level keys {sorted(keys)}). "
        f"Run inspect_file() on it."
    )


__all__ = [
    "FemticHDF5SchemaError",
    "FemticModel", "FemticResults", "read_results_hdf5",
    "FemticExchange", "read_exchange_hdf5",
    "FemticJacobian", "read_jacobian_hdf5",
    "read_rough_hdf5",
    "FemticMesh", "read_mesh_hdf5",
    "read_femtic_hdf5", "inspect_file",
    "BLOCK_FREE_AND_CONSTRAINED", "BLOCK_FIXED_AND_ISOLATED",
    "BLOCK_FIXED_AND_CONSTRAINED", "BLOCK_FREE_AND_ISOLATED",
    "SENSITIVITY_FLOOR", "DATATYPE_NAMES", "COMPONENT_NAMES",
    "DISTORTION_TYPE_NAMES",
]


# ---------------------------------------------------------------------------
# Summary of results_iter<N>.h5 files (no CLI arguments -- edit USER SECTION)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob

    # ======================== USER SECTION ================================
    RESULTS_GLOB = "results_iter*.h5"   # results files to summarise (glob)
    SENS_FILL = None                    # None, or e.g. np.nan for missing sensitivity
    PRINT_TREE = False                  # True: also print inspect_file() tree
    # ======================== END USER SECTION ============================

    def _iter_key(p: str) -> int:
        digits = "".join(c for c in Path(p).stem if c.isdigit())
        return int(digits) if digits else -1

    files = sorted(glob.glob(RESULTS_GLOB), key=_iter_key)
    if not files:
        print(f"No files match {RESULTS_GLOB!r}")
    else:
        sep = "-" * 78
        print(sep)
        print(f"{'file':<24}{'iter':>6}{'nBlocks':>9}{'nFree':>8}{'sens':>6}"
              f"{'nData':>9}{'nMiss':>7}{'dist':>6}")
        print(sep)
        for fn in files:
            r = read_results_hdf5(fn, sens_fill=SENS_FILL)
            m = r.model
            print(f"{Path(fn).name:<24}{str(r.iteration):>6}{m.n_blocks:>9}"
                  f"{int(m.free_mask.sum()):>8}"
                  f"{('yes' if r.sensitivity_computed else 'no'):>6}"
                  f"{r.data.shape[0]:>9}{int(r.data_attrs.get('nMissingCalc', 0)):>7}"
                  f"{('yes' if r.distortion is not None else 'no'):>6}")
            if PRINT_TREE:
                print(inspect_file(fn))
        print(sep)
