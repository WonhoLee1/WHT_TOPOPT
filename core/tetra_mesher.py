"""Conformal 3D Tetrahedral Mesher & Unstructured Spatial Filter for TopOpt.

Fully integrated with Gmsh 4.x and SciPy cKDTree.
Capabilities:
1. Conformal tetrahedral volume mesh generation from arbitrary surface STL files (TET4 / TET10).
2. Boundary node classification via proximity matching against boundary STL surfaces (e.g. GE_fixed.STL, GE_force.STL).
3. Fast cKDTree-based unstructured spatial density filter matrix H.
4. Robust volume weighting for non-uniform mesh topology optimization.
"""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from wht_topopt.core.tetra_assembler import TetraMesh


# ==============================================================================
# 1. Gmsh-Based Conformal 3D Tetrahedral Mesher
# ==============================================================================

def mesh_stl_tetra(
    stl_path: Union[str, Path],
    target_elem_size: float,
    order: int = 1,
    min_elem_size: Optional[float] = None,
    max_elem_size: Optional[float] = None,
    verbose: bool = False,
) -> TetraMesh:
    """Generate a conformal 3D tetrahedral volume mesh from a surface STL file using Gmsh.

    Args:
        stl_path: Path to the watertight surface STL file.
        target_elem_size: Target characteristic element length in model units (mm).
        order: Element polynomial order (1 = TET4, 2 = TET10).
        min_elem_size: Minimum allowed element size (default: 0.5 * target).
        max_elem_size: Maximum allowed element size (default: 1.5 * target).
        verbose: If True, prints Gmsh console messages.

    Returns:
        TetraMesh instance containing 3D nodes and tetrahedral elements.
    """
    import gmsh

    stl_path = Path(stl_path).resolve()
    if not stl_path.exists():
        raise FileNotFoundError(f"Input STL file not found: {stl_path}")

    min_sz = min_elem_size if min_elem_size is not None else 0.5 * target_elem_size
    max_sz = max_elem_size if max_elem_size is not None else 1.5 * target_elem_size

    # Initialize Gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity", 2 if verbose else 0)
    gmsh.model.add(stl_path.stem)

    # 1. Load STL surface
    gmsh.merge(str(stl_path))

    # 2. Surface topology classification
    # Automatically creates discrete surfaces, curves, and points
    gmsh.model.mesh.classifySurfaces(
        angle=40.0 * np.pi / 180.0,
        boundary=True,
        forReparametrization=True,
        curveAngle=180.0 * np.pi / 180.0,
    )
    gmsh.model.mesh.createGeometry()

    # 3. Create closed surface loop and 3D volume
    surfaces = gmsh.model.getEntities(dim=2)
    surface_tags = [s[1] for s in surfaces]
    if len(surface_tags) == 0:
        gmsh.finalize()
        raise RuntimeError(f"Gmsh failed to identify 2D surfaces in STL: {stl_path.name}")

    loop_tag = gmsh.model.geo.addSurfaceLoop(surface_tags)
    vol_tag = gmsh.model.geo.addVolume([loop_tag])
    gmsh.model.geo.synchronize()

    # 4. Mesh settings
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_sz)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_sz)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    # Algorithm 3D: 1=Delaunay, 4=Frontal, 10=HXT (fast parallel)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    # 5. Generate 3D volume mesh
    gmsh.model.mesh.generate(3)

    # 6. Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

    # Create mapping from Gmsh 1-based nodeTag to contiguous 0-based index
    tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}

    # 7. Extract tetrahedral elements
    # In Gmsh: elementType 4 = 4-node TET4, elementType 11 = 10-node TET10
    target_type = 4 if order == 1 else 11
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

    elems_list = []
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == target_type:
            n_per_elem = 4 if order == 1 else 10
            conn = np.array([tag_to_idx[t] for t in enodes], dtype=np.int64).reshape(-1, n_per_elem)
            elems_list.append(conn)

    gmsh.model.remove()
    gmsh.finalize()

    if not elems_list:
        raise RuntimeError(f"No 3D tetrahedral elements of order {order} were generated for {stl_path.name}")

    elems = np.vstack(elems_list)
    elem_code = "TET4" if order == 1 else "TET10"
    return TetraMesh(nodes=nodes, elems=elems, element_type=elem_code)


# ==============================================================================
# 2. Boundary Condition Tagging from Auxiliary STL Files
# ==============================================================================

def tag_nodes_near_stl(
    mesh: TetraMesh,
    stl_path: Union[str, Path],
    tolerance: float = 1.0,
) -> np.ndarray:
    """Find mesh node indices that lie within a given tolerance of a boundary STL surface.

    Used to identify fixed bolt regions (GE_fixed.STL) and clevis pin surfaces (GE_force.STL).
    Uses scipy.spatial.cKDTree over boundary surface sample points for 100% self-contained execution
    without external rtree dependencies.

    Args:
        mesh: TetraMesh instance.
        stl_path: Path to auxiliary surface STL.
        tolerance: Proximity distance threshold in mm.

    Returns:
        1D array of matching global node indices.
    """
    import trimesh

    stl_path = Path(stl_path).resolve()
    if not stl_path.exists():
        raise FileNotFoundError(f"Boundary STL not found: {stl_path}")

    bnd_tri = trimesh.load_mesh(str(stl_path))
    # Collect dense surface sample points: vertices + face midpoints
    pts = [bnd_tri.vertices, bnd_tri.triangles.mean(axis=1)]
    # If face is large, also add edge midpoints
    v0 = bnd_tri.triangles[:, 0]
    v1 = bnd_tri.triangles[:, 1]
    v2 = bnd_tri.triangles[:, 2]
    pts.extend([0.5 * (v0 + v1), 0.5 * (v1 + v2), 0.5 * (v2 + v0)])
    sample_points = np.vstack(pts)

    tree = cKDTree(sample_points)
    distances, _ = tree.query(mesh.nodes, k=1)
    matching_nodes = np.flatnonzero(distances <= tolerance)
    return matching_nodes


# ==============================================================================
# 3. Unstructured Spatial Density Filter Matrix (cKDTree H)
# ==============================================================================

def build_tetra_density_filter(
    mesh: TetraMesh,
    rmin: float,
) -> sparse.csr_matrix:
    """Construct a volume-weighted spatial density filter matrix H for unstructured tetrahedra.

    Following the standard Bourdin/Sigmund formulation:
        w_ij = max(0, rmin - ||c_i - c_j||) * V_j
        H_ij = w_ij / sum_k(w_ik)

    Args:
        mesh: TetraMesh instance.
        rmin: Filter radius in model coordinates (mm).

    Returns:
        Sparse (n_elems, n_elems) normalized density filter matrix in CSR format.
    """
    centroids = mesh.centroids()  # (n_elems, 3)
    volumes = mesh.compute_volumes()  # (n_elems,)
    n_elems = mesh.n_elems

    if rmin <= 0.0:
        return sparse.eye(n_elems, format="csr", dtype=np.float64)

    tree = cKDTree(centroids)
    # Query all element pairs within distance rmin
    pairs = tree.query_pairs(r=rmin, output_type="ndarray")

    # Add self-interactions (i, i)
    self_indices = np.arange(n_elems)
    row_self = self_indices
    col_self = self_indices
    val_self = rmin * volumes

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        dist = np.linalg.norm(centroids[i_idx] - centroids[j_idx], axis=1)
        w_ij = (rmin - dist) * volumes[j_idx]
        w_ji = (rmin - dist) * volumes[i_idx]

        rows = np.concatenate([row_self, i_idx, j_idx])
        cols = np.concatenate([col_self, j_idx, i_idx])
        vals = np.concatenate([val_self, w_ij, w_ji])
    else:
        rows = row_self
        cols = col_self
        vals = val_self

    W = sparse.coo_matrix((vals, (rows, cols)), shape=(n_elems, n_elems)).tocsr()

    # Normalize rows so that H @ 1 = 1
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums < 1e-15] = 1.0
    inv_diag = sparse.diags(1.0 / row_sums, format="csr")
    H = (inv_diag @ W).tocsr()

    return H
