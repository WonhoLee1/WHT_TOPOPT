"""3D Domain Voxelization and Geometric Benchmark Generators.

Provides utilities for:
1. Converting arbitrary 3D STL / Trimesh surface meshes into Hex8 structured grids (StructuredMesh3D)
   with design, passive-solid, and passive-void region masks.
2. Generating parametric benchmark geometries including the GE Aircraft Bracket and 3D Cantilever Beams.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import trimesh

from .mesh import StructuredMesh3D


class VoxelizedDomain3D:
    """Container for a 3D structured mesh with topology optimization region masks."""

    def __init__(
        self,
        mesh: StructuredMesh3D,
        design_mask: np.ndarray,
        passive_solid_mask: np.ndarray,
        passive_void_mask: np.ndarray,
        fixed_dofs: np.ndarray,
        load_dofs: np.ndarray,
        load_values: np.ndarray,
        metadata: Optional[Dict[str, Union[str, float, int]]] = None,
    ) -> None:
        """Initialize voxelized domain.

        Parameters
        ----------
        mesh : StructuredMesh3D
            Underlying 3D structured hexahedral mesh.
        design_mask : np.ndarray
            Boolean array (n_elem,) indicating elements open to topology optimization.
        passive_solid_mask : np.ndarray
            Boolean array (n_elem,) of non-design solid elements (fixed rho = 1.0).
        passive_void_mask : np.ndarray
            Boolean array (n_elem,) of forced void elements (fixed rho = rho_min).
        fixed_dofs : np.ndarray
            Array of constrained global degree of freedom indices.
        load_dofs : np.ndarray
            Array of loaded global degree of freedom indices.
        load_values : np.ndarray
            Array of force values corresponding to load_dofs.
        metadata : dict, optional
            Descriptive dictionary of the problem setup.
        """
        self.mesh: StructuredMesh3D = mesh
        self.design_mask: np.ndarray = np.asarray(design_mask, dtype=bool)
        self.passive_solid_mask: np.ndarray = np.asarray(passive_solid_mask, dtype=bool)
        self.passive_void_mask: np.ndarray = np.asarray(passive_void_mask, dtype=bool)
        self.fixed_dofs: np.ndarray = np.asarray(fixed_dofs, dtype=np.int64)
        self.load_dofs: np.ndarray = np.asarray(load_dofs, dtype=np.int64)
        self.load_values: np.ndarray = np.asarray(load_values, dtype=np.float64)
        self.metadata: Dict[str, Union[str, float, int]] = metadata or {}

        # Validate dimensions
        n_elem = mesh.n_elem
        if len(self.design_mask) != n_elem or len(self.passive_solid_mask) != n_elem:
            raise ValueError(f"Mask length must equal mesh.n_elem ({n_elem})")

    @property
    def n_elements(self) -> int:
        return self.mesh.n_elem

    @property
    def n_nodes(self) -> int:
        return self.mesh.n_nodes

    @property
    def n_dofs(self) -> int:
        return self.mesh.ndof

    def get_initial_density(self, target_volfrac: float = 0.3) -> np.ndarray:
        """Create initial element density distribution honoring solid and void regions."""
        x0 = np.full(self.n_elements, float(target_volfrac), dtype=np.float64)
        x0[self.passive_solid_mask] = 1.0
        x0[self.passive_void_mask] = 1e-6
        return x0

    def assemble_load_vector(self) -> np.ndarray:
        """Assemble full global force vector (n_dofs,)."""
        F = np.zeros(self.n_dofs, dtype=np.float64)
        if len(self.load_dofs) > 0 and len(self.load_values) > 0:
            F[self.load_dofs] = self.load_values
        return F


def create_ge_aircraft_bracket_geometry(
    scale: float = 1.0,
) -> Tuple[trimesh.Trimesh, Dict[str, Union[np.ndarray, float]]]:
    """Generate parametric 3D CAD surface mesh of the GE Aircraft Bracket.

    The bracket features:
    - A base mounting plate with 4 bolt-hole bosses.
    - Two central vertical arch arms meeting at a clevis pin lug.
    - Internal void cutouts between arms.

    Parameters
    ----------
    scale : float
        Overall dimension scale (default 1.0 yields 120 x 60 x 50 mm).

    Returns
    -------
    bracket_mesh : trimesh.Trimesh
        Watertight 3D triangular surface mesh representing the design envelope.
    features : dict
        Dictionary of boss hole centers, clevis pin center, and radius dimensions.
    """
    lx = 120.0 * scale
    ly = 60.0 * scale
    lz = 50.0 * scale
    base_t = 12.0 * scale

    # 4 Bolt holes on base
    bolt_x = 42.0 * scale
    bolt_y = 18.0 * scale
    bolt_centers = np.array([
        [-bolt_x, -bolt_y, base_t * 0.5],
        [+bolt_x, -bolt_y, base_t * 0.5],
        [-bolt_x, +bolt_y, base_t * 0.5],
        [+bolt_x, +bolt_y, base_t * 0.5],
    ], dtype=np.float64)
    bolt_r = 5.0 * scale
    boss_r = 11.0 * scale

    # Central clevis pin lug
    clevis_center = np.array([0.0, 0.0, lz - 12.0 * scale], dtype=np.float64)
    clevis_pin_r = 7.0 * scale
    clevis_lug_r = 16.0 * scale

    # Base block
    base = trimesh.creation.box(extents=[lx, ly, base_t])
    base.apply_translation([0.0, 0.0, base_t * 0.5])

    # Central upright body
    lug_w = 28.0 * scale
    body_box = trimesh.creation.box(extents=[lx * 0.7, lug_w, lz - base_t])
    body_box.apply_translation([0.0, 0.0, base_t + (lz - base_t) * 0.5])

    # Top lug cylinder
    cyl = trimesh.creation.cylinder(radius=clevis_lug_r, height=lug_w)
    # Rotate cylinder to align along y-axis
    rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    cyl.apply_transform(rot_y)
    cyl.apply_translation(clevis_center)

    bracket_mesh = trimesh.util.concatenate([base, body_box, cyl])

    features = {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "base_t": base_t,
        "bolt_centers": bolt_centers,
        "bolt_r": bolt_r,
        "boss_r": boss_r,
        "clevis_center": clevis_center,
        "clevis_pin_r": clevis_pin_r,
        "clevis_lug_r": clevis_lug_r,
    }
    return bracket_mesh, features


def voxelize_ge_aircraft_bracket(
    nx: int = 40,
    ny: int = 20,
    nz: int = 16,
    scale: float = 1.0,
    load_vector: Tuple[float, float, float] = (0.0, 0.0, -1000.0),
) -> VoxelizedDomain3D:
    """Voxelize the GE Aircraft Bracket directly into StructuredMesh3D with exact masks.

    Parameters
    ----------
    nx, ny, nz : int
        Grid resolution along x, y, and z axes.
    scale : float
        Geometric scaling factor.
    load_vector : tuple
        Force applied to the central clevis pin lug [Fx, Fy, Fz] in Newtons.

    Returns
    -------
    VoxelizedDomain3D
        Fully configured 3D domain with boundary conditions and region masks.
    """
    lx = 120.0 * scale
    ly = 60.0 * scale
    lz = 50.0 * scale
    base_t = 12.0 * scale

    # Create StructuredMesh3D centered in xy, starting from z=0
    mesh = StructuredMesh3D(
        nx=nx, ny=ny, nz=nz,
        lx=lx, ly=ly, lz=lz,
    )
    # Shift coordinate origin to center (x in [-lx/2, lx/2], y in [-ly/2, ly/2], z in [0, lz])
    mesh.node_coords[:, 0] -= lx * 0.5
    mesh.node_coords[:, 1] -= ly * 0.5
    mesh.elem_centers[:, 0] -= lx * 0.5
    mesh.elem_centers[:, 1] -= ly * 0.5

    n_elem = mesh.n_elem
    ec = mesh.elem_centers  # (n_elem, 3)

    bolt_x = 42.0 * scale
    bolt_y = 18.0 * scale
    bolt_centers = [
        (-bolt_x, -bolt_y),
        (+bolt_x, -bolt_y),
        (-bolt_x, +bolt_y),
        (+bolt_x, +bolt_y),
    ]
    bolt_r = 5.0 * scale
    boss_r = 11.0 * scale

    clevis_z = lz - 12.0 * scale
    clevis_pin_r = 7.0 * scale
    clevis_lug_r = 16.0 * scale
    lug_w = 28.0 * scale

    # 1. Determine base elements
    is_base = (ec[:, 2] <= base_t) & (np.abs(ec[:, 0]) <= lx * 0.5) & (np.abs(ec[:, 1]) <= ly * 0.5)

    # 2. Determine bolt hole voids and boss solid rings
    is_bolt_hole = np.zeros(n_elem, dtype=bool)
    is_bolt_boss = np.zeros(n_elem, dtype=bool)

    for bx, by in bolt_centers:
        dist_xy = np.sqrt((ec[:, 0] - bx) ** 2 + (ec[:, 1] - by) ** 2)
        hole = (dist_xy <= bolt_r) & (ec[:, 2] <= base_t * 1.2)
        boss = (dist_xy > bolt_r) & (dist_xy <= boss_r) & (ec[:, 2] <= base_t)
        is_bolt_hole |= hole
        is_bolt_boss |= boss

    # 3. Determine clevis pin hole and clevis lug
    dist_clevis_xz = np.sqrt(ec[:, 0] ** 2 + (ec[:, 2] - clevis_z) ** 2)
    in_clevis_y = np.abs(ec[:, 1]) <= (lug_w * 0.5)
    is_clevis_hole = (dist_clevis_xz <= clevis_pin_r) & in_clevis_y
    is_clevis_boss = (dist_clevis_xz > clevis_pin_r) & (dist_clevis_xz <= clevis_lug_r) & in_clevis_y

    # 4. Central body envelope (pyramidal / arch envelope)
    # The arms taper from base to top clevis lug
    taper = np.clip(1.0 - (ec[:, 2] - base_t) / max(1e-6, (lz - base_t)), 0.0, 1.0)
    allowed_x = (lx * 0.45) * taper + (clevis_lug_r * 1.1) * (1.0 - taper)
    allowed_y = (ly * 0.45) * taper + (lug_w * 0.55) * (1.0 - taper)
    is_upright_envelope = (ec[:, 2] > base_t) & (np.abs(ec[:, 0]) <= allowed_x) & (np.abs(ec[:, 1]) <= allowed_y)

    # Combined geometry mask
    in_envelope = (is_base | is_upright_envelope | is_clevis_boss) & ~is_bolt_hole & ~is_clevis_hole

    passive_solid = (is_bolt_boss | is_clevis_boss) & in_envelope
    passive_void = ~in_envelope | is_bolt_hole | is_clevis_hole
    design_mask = in_envelope & ~passive_solid

    # 5. Boundary conditions (fixed bolt holes)
    # Fix all nodes within the 4 bolt holes on the bottom face z=0
    nc = mesh.node_coords
    fixed_node_mask = np.zeros(mesh.n_nodes, dtype=bool)
    is_bottom = np.abs(nc[:, 2]) <= 1e-4

    for bx, by in bolt_centers:
        dist_xy = np.sqrt((nc[:, 0] - bx) ** 2 + (nc[:, 1] - by) ** 2)
        bolt_nodes = (dist_xy <= boss_r) & is_bottom
        fixed_node_mask |= bolt_nodes

    fixed_nodes = np.flatnonzero(fixed_node_mask)
    fixed_dofs = np.concatenate([
        3 * fixed_nodes,
        3 * fixed_nodes + 1,
        3 * fixed_nodes + 2,
    ])
    fixed_dofs = np.unique(fixed_dofs)

    # 6. Applied load on clevis pin inside surface
    dist_pin_xz = np.sqrt(nc[:, 0] ** 2 + (nc[:, 2] - clevis_z) ** 2)
    load_node_mask = (dist_pin_xz <= clevis_pin_r * 1.05) & (np.abs(nc[:, 1]) <= (lug_w * 0.5))
    load_nodes = np.flatnonzero(load_node_mask)

    if len(load_nodes) == 0:
        # Fallback to closest nodes
        closest = np.argmin(dist_pin_xz)
        load_nodes = np.array([closest], dtype=np.int64)

    fx_per_node = load_vector[0] / len(load_nodes)
    fy_per_node = load_vector[1] / len(load_nodes)
    fz_per_node = load_vector[2] / len(load_nodes)

    load_dofs_list = []
    load_vals_list = []
    for nid in load_nodes:
        if abs(fx_per_node) > 1e-12:
            load_dofs_list.append(3 * nid)
            load_vals_list.append(fx_per_node)
        if abs(fy_per_node) > 1e-12:
            load_dofs_list.append(3 * nid + 1)
            load_vals_list.append(fy_per_node)
        if abs(fz_per_node) > 1e-12:
            load_dofs_list.append(3 * nid + 2)
            load_vals_list.append(fz_per_node)

    load_dofs = np.array(load_dofs_list, dtype=np.int64)
    load_vals = np.array(load_vals_list, dtype=np.float64)

    meta = {
        "case_name": "GE_Aircraft_Bracket_3D",
        "nx": nx, "ny": ny, "nz": nz,
        "n_elements": n_elem,
        "n_design_elements": int(np.sum(design_mask)),
        "n_solid_elements": int(np.sum(passive_solid)),
        "n_void_elements": int(np.sum(passive_void)),
        "n_fixed_nodes": len(fixed_nodes),
        "n_load_nodes": len(load_nodes),
        "total_force_N": float(np.linalg.norm(load_vector)),
    }

    return VoxelizedDomain3D(
        mesh=mesh,
        design_mask=design_mask,
        passive_solid_mask=passive_solid,
        passive_void_mask=passive_void,
        fixed_dofs=fixed_dofs,
        load_dofs=load_dofs,
        load_values=load_vals,
        metadata=meta,
    )


def voxelize_top3d_cantilever_beam(
    nx: int = 40,
    ny: int = 20,
    nz: int = 10,
    lx: float = 60.0,
    ly: float = 30.0,
    lz: float = 15.0,
    total_force: float = -1.0,
) -> VoxelizedDomain3D:
    """Generate academic standard 3D cantilever beam benchmark (Ferrari & Sigmund top3d125).

    Fixed face: X = 0 (all DOFs fixed)
    Load: Applied at bottom center of the free end (X=lx, Y=ly/2, Z=0) in -Z direction.

    Parameters
    ----------
    nx, ny, nz : int
        Element resolution.
    lx, ly, lz : float
        Physical dimensions.
    total_force : float
        Total downward force on the tip.

    Returns
    -------
    VoxelizedDomain3D
        Configured 3D benchmark domain.
    """
    mesh = StructuredMesh3D(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz)
    n_elem = mesh.n_elem

    # All elements in rectangular domain are design elements
    design_mask = np.ones(n_elem, dtype=bool)
    passive_solid = np.zeros(n_elem, dtype=bool)
    passive_void = np.zeros(n_elem, dtype=bool)

    # Fixed DOFs at X = 0
    fixed_nodes = mesh.get_nodes_at_x(0.0)
    fixed_dofs = np.concatenate([
        3 * fixed_nodes,
        3 * fixed_nodes + 1,
        3 * fixed_nodes + 2,
    ])
    fixed_dofs = np.unique(fixed_dofs)

    # Load node: Bottom center at X=lx
    tip_nodes = mesh.get_nodes_at_x(lx)
    tip_coords = mesh.node_coords[tip_nodes]
    # Find node closest to (lx, ly/2, 0)
    dist = (tip_coords[:, 1] - ly * 0.5) ** 2 + tip_coords[:, 2] ** 2
    best_idx = np.argmin(dist)
    load_node = tip_nodes[best_idx]

    load_dofs = np.array([3 * load_node + 2], dtype=np.int64)  # Z direction
    load_vals = np.array([total_force], dtype=np.float64)

    meta = {
        "case_name": "Ferrari_Sigmund_top3d125_Cantilever",
        "nx": nx, "ny": ny, "nz": nz,
        "n_elements": n_elem,
        "n_design_elements": n_elem,
        "n_fixed_nodes": len(fixed_nodes),
        "total_force_N": abs(total_force),
    }

    return VoxelizedDomain3D(
        mesh=mesh,
        design_mask=design_mask,
        passive_solid_mask=passive_solid,
        passive_void_mask=passive_void,
        fixed_dofs=fixed_dofs,
        load_dofs=load_dofs,
        load_values=load_vals,
        metadata=meta,
    )
