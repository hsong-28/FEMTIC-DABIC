#!/usr/bin/env python3
"""Plot Broken Hill A-A' from native DHEXA/TETRA mesh and model files.

Standalone adaptation of the maintained native-cell plotting workflow:
  maintenance/tools/plot_brokenhill_native_mesh_comparison.py
  maintenance/tools/plot_brokenhill_native_five_profiles.py
  maintenance/tools/plot_brokenhill_mixed_mesh_results.py
The readers retain FEMTIC's [northing, easting, depth-positive-down] order.
DHEXA uses line/box intersections; TETRA uses PyVista plane intersections.
No cell-to-point conversion, spatial interpolation, or smoothing is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def indexed_row(stream, index, columns):
    fields = stream.readline().split()
    if len(fields) != columns or int(fields[0]) != index:
        raise ValueError(f"Invalid row {index}: expected {columns} fields and sequential IDs")
    return fields


def read_mesh(path):
    """Read the node/connectivity prefix; FEMTIC coordinates are in metres."""
    with Path(path).open(encoding="ascii") as stream:
        kind = stream.readline().strip()
        if kind not in ("DHEXA", "TETRA"):
            raise ValueError(f"Unsupported mesh header: {kind!r}")
        node_count = int(stream.readline().split()[0])
        if node_count <= 0:
            raise ValueError("Mesh must contain nodes")
        nodes = np.empty((node_count, 3))
        for index in range(node_count):
            nodes[index] = [float(v) * 0.001 for v in indexed_row(stream, index, 4)[1:]]
        element_count = int(stream.readline().split()[0])
        if element_count <= 0:
            raise ValueError("Mesh must contain elements")
        vertices_per_cell = 8 if kind == "DHEXA" else 4
        elements = np.empty((element_count, vertices_per_cell), dtype=np.int64)
        for index in range(element_count):
            row = indexed_row(stream, index, 9)
            elements[index] = [int(v) for v in row[-vertices_per_cell:]]
            if kind == "DHEXA":
                for _ in range(6):
                    if not stream.readline().strip():
                        raise ValueError("Incomplete DHEXA neighbour table")
    if not np.isfinite(nodes).all():
        raise ValueError("Non-finite mesh coordinates")
    if np.any(elements < 0) or np.any(elements >= node_count):
        raise ValueError("Mesh node index out of range")
    if np.any(np.diff(np.sort(elements, axis=1), axis=1) == 0):
        raise ValueError("Repeated vertex in a mesh element")
    return kind, nodes, elements


def read_model(path, element_count):
    """Read native block values; block 0 is air in these Broken Hill cases."""
    with Path(path).open(encoding="ascii") as stream:
        count, block_count = map(int, stream.readline().split())
        if count != element_count or block_count < 2:
            raise ValueError("Mesh/model element-count mismatch or invalid block count")
        block_ids = np.empty(count, dtype=np.int64)
        for index in range(count):
            block_ids[index] = int(indexed_row(stream, index, 2)[1])
        block_rho = np.empty(block_count)
        for index in range(block_count):
            row = indexed_row(stream, index, 6)
            block_rho[index] = float(row[1])
            if index == 0 and int(row[5]) != 1:
                raise ValueError("Expected fixed air block 0 for Broken Hill")
    if np.any(block_ids < 0) or np.any(block_ids >= block_count):
        raise ValueError("Resistivity block index out of range")
    if not np.isfinite(block_rho).all() or np.any(block_rho <= 0):
        raise ValueError("Resistivity must be finite and positive")
    values = np.log10(block_rho[block_ids])
    values[block_ids == 0] = np.nan
    return values


def read_profile(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    start = np.asarray(document["start_north_east_km"], dtype=float)
    end = np.asarray(document["end_north_east_km"], dtype=float)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all():
        raise ValueError("Profile endpoints must be finite [northing, easting] pairs")
    length = np.linalg.norm(end - start)
    if length <= 0 or not document.get("geometry_status"):
        raise ValueError("Distinct endpoints and geometry_status are required")
    stations = np.asarray(
        [[s["northing_km"], s["easting_km"]] for s in document["stations"]], dtype=float
    )
    if stations.ndim != 2 or stations.shape[1] != 2 or not np.isfinite(stations).all():
        raise ValueError("Invalid station coordinates")
    return document, start, end, float(length), stations


def dhexa_section(nodes, elements, values, start, end, depth):
    """Exact intersections for axis-aligned DHEXA cells, as in the report."""
    vertices = nodes[elements]
    low, high = vertices.min(axis=1), vertices.max(axis=1)
    if np.any(high <= low) or not np.all((vertices == low[:, None]) | (vertices == high[:, None])):
        raise ValueError("This plotter requires axis-aligned DHEXA cells")
    delta = end - start
    length = np.linalg.norm(delta)
    polygons, selected = [], []
    candidates = np.flatnonzero(np.isfinite(values) & (high[:, 2] > 0) & (low[:, 2] < depth))
    for cell in candidates:
        entry, exit_ = 0.0, 1.0
        for axis in range(2):
            if abs(delta[axis]) <= 1.0e-15:
                if not low[cell, axis] <= start[axis] < high[cell, axis]:
                    exit_ = entry
                    break
                continue
            first = (low[cell, axis] - start[axis]) / delta[axis]
            second = (high[cell, axis] - start[axis]) / delta[axis]
            entry = max(entry, min(first, second))
            exit_ = min(exit_, max(first, second))
        if exit_ <= entry:
            continue
        left, right = entry * length, exit_ * length
        top, bottom = max(0.0, low[cell, 2]), min(depth, high[cell, 2])
        polygons.append(np.asarray([[left, top], [right, top], [right, bottom], [left, bottom]]))
        selected.append(cell)
    return polygons, values[selected], np.asarray(selected, dtype=np.int64)


def tetra_section(nodes, elements, values, start, end, depth):
    """Preserve source-cell values through the validated PyVista slicing path."""
    import pyvista as pv

    direction = (end - start) / np.linalg.norm(end - start)
    normal = np.asarray([-direction[1], direction[0]])
    relative = nodes[:, :2] - start
    points = np.column_stack((relative @ direction, relative @ normal, nodes[:, 2]))
    cells = np.column_stack((np.full(len(elements), 4), elements)).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(elements), pv.CellType.TETRA, dtype=np.uint8), points)
    grid.cell_data["log10_resistivity"] = values
    grid.cell_data["source_element_id"] = np.arange(len(elements), dtype=np.int64)
    length = float(np.linalg.norm(end - start))
    section = grid.slice(normal=(0, 1, 0), origin=(length / 2, 0, depth / 2))
    section = section.clip_box(bounds=(0, length, -0.01, 0.01, 0, depth), invert=False)
    section = section.extract_cells(np.isfinite(section.cell_data["log10_resistivity"]))
    source_ids = np.asarray(section.cell_data["source_element_id"], dtype=np.int64)
    section_values = np.asarray(section.cell_data["log10_resistivity"])
    if not np.array_equal(section_values, values[source_ids]):
        raise ValueError("Sliced values differ from the source-cell values")
    polygons = [
        np.asarray(section.points[section.get_cell(i).point_ids])[:, [0, 2]]
        for i in range(section.n_cells)
    ]
    return polygons, section_values, source_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output prefix, without extension")
    parser.add_argument("--label", default="Broken Hill", help="Case label; the model filename is also shown")
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("profile_AA.json"))
    parser.add_argument("--depth-max-km", type=float, default=10.0)
    parser.add_argument("--color-limits", nargs=2, type=float, default=(1.2, 4.5), metavar=("MIN", "MAX"))
    args = parser.parse_args()
    if not np.isfinite([args.depth_max_km, *args.color_limits]).all() or args.depth_max_km <= 0:
        parser.error("Depth must be positive and plot limits must be finite")
    if args.color_limits[0] >= args.color_limits[1]:
        parser.error("Color MIN must be smaller than MAX")
    outputs = [Path(str(args.output) + suffix) for suffix in (".png", ".pdf", ".json")]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Choose a new output prefix; file already exists: {path}")

    geometry, start, end, length, stations = read_profile(args.profile)
    kind, nodes, elements = read_mesh(args.mesh)
    values = read_model(args.model, len(elements))
    slicer = dhexa_section if kind == "DHEXA" else tetra_section
    polygons, section_values, source_ids = slicer(nodes, elements, values, start, end, args.depth_max_km)
    if not polygons:
        raise ValueError("A-A' does not intersect any non-air cells in the requested depth range")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    direction = (end - start) / length
    normal = np.asarray([-direction[1], direction[0]])
    along = (stations - start) @ direction
    offsets = np.abs((stations - start) @ normal)
    corridor = 1.5  # Same station-selection corridor as the report.
    selected = (along >= 0) & (along <= length) & (offsets <= corridor)
    figure, (location, section) = plt.subplots(
        1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 2.3]}, layout="constrained"
    )
    location.scatter(stations[:, 1], stations[:, 0], s=18, c="0.6", marker="v", label="MT stations")
    location.scatter(stations[selected, 1], stations[selected, 0], s=22, c="black", marker="v", label="Within 1.5 km")
    location.plot([start[1], end[1]], [start[0], end[0]], color="#d627a8", lw=1.7)
    for point, label in ((start, "A"), (end, "A'")):
        location.annotate(label, (point[1], point[0]), xytext=(5, 5), textcoords="offset points", weight="bold")
    location.set(xlabel="Easting (km)", ylabel="Northing (km)", title="Profile location")
    location.set_aspect("equal")
    location.margins(0.15)
    location.legend(loc="lower left", fontsize=8)
    collection = PolyCollection(
        polygons, array=section_values, cmap="jet_r", norm=Normalize(*args.color_limits),
        edgecolors="none", antialiaseds=False, rasterized=True,
    )
    section.add_collection(collection)
    section.scatter(along[selected], np.zeros(selected.sum()), marker="v", s=25, c="black", clip_on=False, zorder=3)
    section.set(xlim=(0, length), ylim=(args.depth_max_km, 0), xlabel="Distance from A (km)", ylabel="Depth (km)")
    section.set_aspect("equal")
    section.set_title("A", loc="left", weight="bold")
    section.set_title("A'", loc="right", weight="bold")
    figure.colorbar(collection, ax=section, orientation="horizontal", shrink=0.85, pad=0.12, label=r"$\log_{10}[\rho/(\Omega\,\mathrm{m})]$", extend="both")
    figure.suptitle(
        f"{args.label} | {kind} | {args.model.name}\n"
        "Native cell values; approximate digitized A-A'; flat z=0 surface", fontsize=11
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs[:2]:
        figure.savefig(path, dpi=240)
    plt.close(figure)
    record = {
        "mesh": {"path": str(args.mesh.resolve()), "sha256": sha256(args.mesh)},
        "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
        "profile": {"path": str(args.profile.resolve()), "sha256": sha256(args.profile)},
        "geometry_status": geometry["geometry_status"],
        "start_north_east_km": start.tolist(), "end_north_east_km": end.tolist(),
        "profile_length_km": length, "depth_max_km": args.depth_max_km,
        "coordinate_convention": "X=northing, Y=easting, Z=depth positive down; plotted in km",
        "mesh_type": kind, "model_label": args.label,
        "air_mask": "resistivity block 0", "spatial_interpolation": "none",
        "element_count": len(elements), "slice_polygon_count": len(polygons),
        "slice_source_element_count": int(np.unique(source_ids).size),
        "source_values_exact": bool(np.array_equal(section_values, values[source_ids])),
        "station_corridor_km": corridor, "projected_station_count": int(selected.sum()),
        "color_limits_log10_ohm_m": list(args.color_limits),
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
    }
    if kind == "TETRA":
        import pyvista as pv
        record["pyvista_version"] = pv.__version__
    outputs[2].write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
