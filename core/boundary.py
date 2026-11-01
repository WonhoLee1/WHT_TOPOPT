"""Boundary condition management and benchmark problem presets for topology optimization.

Provides BoundaryConditionManager for defining Dirichlet constraints (fixed DOFs),
nodal point/distributed forces, passive solid/void design domains, and classic
benchmark builders (MBB, Cantilever 2D/3D, L-Bracket, Michell, Compliant Inverter).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from .mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh


class BoundaryConditionManager:
    """Manages boundary constraints, external loads, and passive solid/void domains."""

    def __init__(self, mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]) -> None:
        """Initialize BoundaryConditionManager on a mesh.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Computational mesh instance.
        """
        self.mesh = mesh
        self.ndof: int = mesh.ndof
        self.n_elem: int = mesh.n_elem
        self.ndof_per_node: int = mesh.ndof_per_node

        self._fixed_dofs_set: Set[int] = set()
        self._forces: Dict[int, Dict[int, float]] = {}  # {load_case: {dof: value}}

        self._passive_solid: np.ndarray = np.zeros(self.n_elem, dtype=bool)
        self._passive_void: np.ndarray = np.zeros(self.n_elem, dtype=bool)

    def fix_dofs(self, dofs: Union[int, Sequence[int], np.ndarray]) -> None:
        """Fix specified degrees of freedom to zero displacement."""
        if isinstance(dofs, (int, np.integer)):
            dofs = [int(dofs)]
        for d in dofs:
            d_idx = int(d)
            if not (0 <= d_idx < self.ndof):
                raise IndexError(f"DOF index {d_idx} out of range [0, {self.ndof})")
            self._fixed_dofs_set.add(d_idx)

    def fix_nodes(
        self,
        nodes: Union[int, Sequence[int], np.ndarray],
        components: Optional[Sequence[int]] = None,
    ) -> None:
        """Fix degrees of freedom on specified nodes.

        Parameters
        ----------
        nodes : int, sequence of int, or np.ndarray
            Node indices to fix.
        components : sequence of int, optional
            DOF components per node to fix (e.g., [0] for ux, [1] for uy).
            If None, fixes all DOFs for the node.
        """
        if isinstance(nodes, (int, np.integer)):
            nodes = [int(nodes)]

        comps = list(range(self.ndof_per_node)) if components is None else list(components)

        for n in nodes:
            n_idx = int(n)
            if not (0 <= n_idx < self.mesh.n_nodes):
                raise IndexError(f"Node index {n_idx} out of range [0, {self.mesh.n_nodes})")
            for c in comps:
                if not (0 <= c < self.ndof_per_node):
                    raise ValueError(f"Component {c} out of range [0, {self.ndof_per_node})")
                dof = n_idx * self.ndof_per_node + c
                self._fixed_dofs_set.add(dof)

    def add_nodal_force(
        self,
        nodes: Union[int, Sequence[int], np.ndarray],
        force_vector: Sequence[float],
        load_case: int = 0,
    ) -> None:
        """Apply a point or distributed nodal force.

        Parameters
        ----------
        nodes : int or sequence of int
            Target node indices.
        force_vector : sequence of float
            Force vector components per node (length ndof_per_node).
        load_case : int
            Subcase / load case index (default 0).
        """
        if isinstance(nodes, (int, np.integer)):
            nodes = [int(nodes)]

        f_vec = np.asarray(force_vector, dtype=np.float64)
        if len(f_vec) != self.ndof_per_node:
            raise ValueError(
                f"force_vector length {len(f_vec)} must match ndof_per_node {self.ndof_per_node}"
            )

        if load_case not in self._forces:
            self._forces[load_case] = {}

        lc_dict = self._forces[load_case]
        for n in nodes:
            n_idx = int(n)
            for c in range(self.ndof_per_node):
                val = float(f_vec[c])
                if val != 0.0:
                    dof = n_idx * self.ndof_per_node + c
                    lc_dict[dof] = lc_dict.get(dof, 0.0) + val

    def set_passive_solid(self, elements: Union[int, Sequence[int], np.ndarray]) -> None:
        """Mark elements as non-design solid domain (fixed density = 1)."""
        elems = np.asarray(elements, dtype=np.int64).ravel()
        self._passive_solid[elems] = True
        self._passive_void[elems] = False

    def set_passive_void(self, elements: Union[int, Sequence[int], np.ndarray]) -> None:
        """Mark elements as non-design void domain (fixed density = 0)."""
        elems = np.asarray(elements, dtype=np.int64).ravel()
        self._passive_void[elems] = True
        self._passive_solid[elems] = False

    @property
    def fixed_dofs(self) -> np.ndarray:
        """Sorted 1D array of fixed DOF indices."""
        return np.array(sorted(self._fixed_dofs_set), dtype=np.int64)

    @property
    def free_dofs(self) -> np.ndarray:
        """Sorted 1D array of unconstrained (free) DOF indices."""
        all_dofs = np.ones(self.ndof, dtype=bool)
        if self._fixed_dofs_set:
            fixed = list(self._fixed_dofs_set)
            all_dofs[fixed] = False
        return np.flatnonzero(all_dofs)

    @property
    def F(self) -> np.ndarray:
        """Force matrix of shape (ndof, n_load_cases)."""
        if not self._forces:
            return np.zeros((self.ndof, 1), dtype=np.float64)

        n_lc = max(self._forces.keys()) + 1
        F_mat = np.zeros((self.ndof, n_lc), dtype=np.float64)
        for lc, dof_vals in self._forces.items():
            for dof, val in dof_vals.items():
                F_mat[dof, lc] += val
        return F_mat

    @property
    def passive_solid_mask(self) -> np.ndarray:
        """Boolean mask of non-design solid elements."""
        return self._passive_solid

    @property
    def passive_void_mask(self) -> np.ndarray:
        """Boolean mask of non-design void elements."""
        return self._passive_void

    @property
    def active_elements(self) -> np.ndarray:
        """Boolean mask of elements that are free design variables (neither passive solid nor void)."""
        return ~(self._passive_solid | self._passive_void)

    # -----------------------------------------------------------------------
    # Static Preset Builders for Classic Benchmarks
    # -----------------------------------------------------------------------

    @staticmethod
    def mbb_half_2d(
        mesh: Union[StructuredMesh2D, UnstructuredMesh],
        F_mag: float = 1.0,
    ) -> BoundaryConditionManager:
        """Construct Half MBB Beam benchmark (symmetric half).

        Boundary conditions:
            - Left boundary (x = 0): Roller (fix u_x = 0 for symmetry).
            - Bottom-right corner (x = lx, y = 0): Pin/Roller (fix u_y = 0).
            - Top-left corner (x = 0, y = ly): Downward point load F_y = -F_mag.
        """
        bcm = BoundaryConditionManager(mesh)

        # Left symmetry line: x = 0 -> fix u_x (component 0)
        left_nodes = mesh.get_nodes_at_x(0.0)
        bcm.fix_nodes(left_nodes, components=[0])

        # Bottom right support: x = max(x), y = 0 -> fix u_y (component 1)
        lx = float(np.max(mesh.node_coords[:, 0]))
        br_nodes = mesh.get_nodes_in_box((lx - 1e-6, lx + 1e-6), (0.0 - 1e-6, 0.0 + 1e-6))
        bcm.fix_nodes(br_nodes, components=[1])

        # Top-left load: x = 0, y = ly
        ly = float(np.max(mesh.node_coords[:, 1]))
        tl_nodes = mesh.get_nodes_in_box((0.0 - 1e-6, 0.0 + 1e-6), (ly - 1e-6, ly + 1e-6))
        bcm.add_nodal_force(tl_nodes, [0.0, -float(F_mag)], load_case=0)

        return bcm

    @staticmethod
    def cantilever_2d(
        mesh: Union[StructuredMesh2D, UnstructuredMesh],
        F_mag: float = 1.0,
    ) -> BoundaryConditionManager:
        """Construct 2D Cantilever benchmark.

        Boundary conditions:
            - Left edge (x = 0): Fully clamped (fix u_x = 0, u_y = 0).
            - Right edge center (x = lx, y = ly/2): Downward point load F_y = -F_mag.
        """
        bcm = BoundaryConditionManager(mesh)

        # Clamped left edge
        left_nodes = mesh.get_nodes_at_x(0.0)
        bcm.fix_nodes(left_nodes)

        # Load at right edge center
        lx = float(np.max(mesh.node_coords[:, 0]))
        ly = float(np.max(mesh.node_coords[:, 1]))
        load_nodes = mesh.get_nodes_in_box(
            (lx - 1e-6, lx + 1e-6),
            (0.5 * ly - 0.5 * (ly / mesh.n_nodes**0.5) - 1e-5, 0.5 * ly + 0.5 * (ly / mesh.n_nodes**0.5) + 1e-5),
        )
        if len(load_nodes) == 0:
            # Fallback to bottom-right corner if no center node found
            load_nodes = mesh.get_nodes_in_box((lx - 1e-6, lx + 1e-6), (-1e-6, 1e-6))

        bcm.add_nodal_force(load_nodes, [0.0, -float(F_mag)], load_case=0)
        return bcm

    @staticmethod
    def l_bracket_2d(
        mesh: Union[StructuredMesh2D, UnstructuredMesh],
        F_mag: float = 1.0,
    ) -> BoundaryConditionManager:
        """Construct L-Bracket benchmark.

        Domain has passive void cut out in upper-right quadrant [0.4*lx, lx] x [0.4*ly, ly].
        Boundary conditions:
            - Top edge of left vertical arm: Clamped.
            - Right edge of horizontal arm: Downward point load.
        """
        bcm = BoundaryConditionManager(mesh)
        lx = float(np.max(mesh.node_coords[:, 0]))
        ly = float(np.max(mesh.node_coords[:, 1]))

        # Cutout passive void in upper-right quadrant
        void_elems = mesh.get_elements_in_box((0.4 * lx, lx), (0.4 * ly, ly))
        bcm.set_passive_void(void_elems)

        # Clamped top edge of left arm: x in [0, 0.4*lx], y = ly
        top_nodes = mesh.get_nodes_in_box((0.0, 0.4 * lx), (ly - 1e-6, ly + 1e-6))
        bcm.fix_nodes(top_nodes)

        # Load at right edge center of bottom arm: x = lx, y in [0.15*ly, 0.25*ly]
        load_nodes = mesh.get_nodes_in_box((lx - 1e-6, lx + 1e-6), (0.15 * ly, 0.25 * ly))
        if len(load_nodes) == 0:
            load_nodes = mesh.get_nodes_in_box((lx - 1e-6, lx + 1e-6), (0.0, 0.4 * ly))

        bcm.add_nodal_force(load_nodes, [0.0, -float(F_mag)], load_case=0)
        return bcm

    @staticmethod
    def michell_2d(
        mesh: Union[StructuredMesh2D, UnstructuredMesh],
        F_mag: float = 1.0,
    ) -> BoundaryConditionManager:
        """Construct Michell truss benchmark.

        Boundary conditions:
            - Bottom-left corner: Pinned support (fix u_x, u_y).
            - Bottom-right corner: Roller support (fix u_y).
            - Bottom-center node: Downward point load.
        """
        bcm = BoundaryConditionManager(mesh)
        lx = float(np.max(mesh.node_coords[:, 0]))

        # Bottom-left pin
        bl_nodes = mesh.get_nodes_in_box((-1e-6, 1e-6), (-1e-6, 1e-6))
        bcm.fix_nodes(bl_nodes, components=[0, 1])

        # Bottom-right roller
        br_nodes = mesh.get_nodes_in_box((lx - 1e-6, lx + 1e-6), (-1e-6, 1e-6))
        bcm.fix_nodes(br_nodes, components=[1])

        # Bottom-center load
        bc_nodes = mesh.get_nodes_in_box((0.5 * lx - 1e-3 * lx, 0.5 * lx + 1e-3 * lx), (-1e-6, 1e-6))
        bcm.add_nodal_force(bc_nodes, [0.0, -float(F_mag)], load_case=0)
        return bcm

    @staticmethod
    def compliant_inverter_2d(
        mesh: Union[StructuredMesh2D, UnstructuredMesh],
        F_in: float = 1.0,
        k_in: float = 1.0,
        k_out: float = 0.001,
    ) -> BoundaryConditionManager:
        """Construct Compliant Inverter mechanism (symmetric half or full).

        Boundary conditions:
            - Left top/bottom fixed pivots.
            - Input force at center left, output displacement measured at right.
        """
        bcm = BoundaryConditionManager(mesh)
        lx = float(np.max(mesh.node_coords[:, 0]))
        ly = float(np.max(mesh.node_coords[:, 1]))

        # Symmetric half: bottom edge y = 0 has symmetry u_y = 0
        sym_nodes = mesh.get_nodes_at_y(0.0)
        bcm.fix_nodes(sym_nodes, components=[1])

        # Pivot support at top-left: (x = 0, y = ly)
        pivot = mesh.get_nodes_in_box((-1e-6, 1e-6), (ly - 1e-6, ly + 1e-6))
        bcm.fix_nodes(pivot, components=[0, 1])

        # Input load at (x = 0, y = 0) pushing right
        in_nodes = mesh.get_nodes_in_box((-1e-6, 1e-6), (-1e-6, 1e-6))
        bcm.add_nodal_force(in_nodes, [float(F_in), 0.0], load_case=0)

        # Output dummy unit adjoint load at (x = lx, y = 0) pulling left
        out_nodes = mesh.get_nodes_in_box((lx - 1e-6, lx + 1e-6), (-1e-6, 1e-6))
        bcm.add_nodal_force(out_nodes, [-1.0, 0.0], load_case=1)

        return bcm

    force_inverter_2d = compliant_inverter_2d

    @staticmethod
    def cantilever_3d(
        mesh: Union[StructuredMesh3D, UnstructuredMesh],
        F_mag: float = 1.0,
    ) -> BoundaryConditionManager:
        """Construct 3D Cantilever benchmark.

        Boundary conditions:
            - Left face (x = 0): Fully clamped (fix u_x, u_y, u_z).
            - Right face center (x = lx, y = ly/2, z = 0): Downward point load F_z = -F_mag.
        """
        bcm = BoundaryConditionManager(mesh)

        # Clamped left face
        left_nodes = mesh.get_nodes_at_x(0.0)
        bcm.fix_nodes(left_nodes)

        # Load at center of right face
        lx = float(np.max(mesh.node_coords[:, 0]))
        ly = float(np.max(mesh.node_coords[:, 1]))
        lz = float(np.max(mesh.node_coords[:, 2]))

        load_nodes = mesh.get_nodes_in_box(
            (lx - 1e-6, lx + 1e-6),
            (0.5 * ly - 0.1 * ly, 0.5 * ly + 0.1 * ly),
            (0.0 - 1e-6, 0.1 * lz),
        )
        if len(load_nodes) == 0:
            load_nodes = mesh.get_nodes_in_box(
                (lx - 1e-6, lx + 1e-6),
                (-1e-6, 1e-6),
                (-1e-6, 1e-6),
            )

        bcm.add_nodal_force(load_nodes, [0.0, 0.0, -float(F_mag)], load_case=0)
        return bcm

    @staticmethod
    def compressed_column_2d(
        mesh: Union[StructuredMesh2D, UnstructuredMesh],
        F_mag: float = 1.0,
        load_mode: str = "distributed",
    ) -> BoundaryConditionManager:
        """Construct 2D compressed column benchmark for buckling analysis.

        Boundary conditions:
            - Bottom edge (y = 0): Fully clamped (fix u_x = 0, u_y = 0).
            - Top edge (y = ly): Downward compressive load F_y = -F_mag.
              If load_mode == 'distributed', load is distributed across top edge nodes.
              If load_mode == 'center', load is applied at top center node.
        """
        bcm = BoundaryConditionManager(mesh)
        bot_nodes = mesh.get_nodes_at_y(0.0)
        bcm.fix_nodes(bot_nodes)

        ly = float(np.max(mesh.node_coords[:, 1]))
        top_nodes = mesh.get_nodes_at_y(ly)

        if load_mode.lower() == "center":
            lx = float(np.max(mesh.node_coords[:, 0]))
            center_nodes = mesh.get_nodes_in_box(
                (0.5 * lx - 1e-4 * lx, 0.5 * lx + 1e-4 * lx),
                (ly - 1e-6, ly + 1e-6),
            )
            if len(center_nodes) == 0:
                center_nodes = [top_nodes[len(top_nodes) // 2]]
            bcm.add_nodal_force(center_nodes, [0.0, -float(F_mag)], load_case=0)
        else:
            n_top = len(top_nodes)
            f_per_node = float(F_mag) / max(1, n_top)
            bcm.add_nodal_force(top_nodes, [0.0, -f_per_node], load_case=0)

        return bcm

    @staticmethod
    def compressed_column_3d(
        mesh: Union[StructuredMesh3D, UnstructuredMesh],
        F_mag: float = 1.0,
    ) -> BoundaryConditionManager:
        """Construct 3D compressed column benchmark for buckling analysis.

        Boundary conditions:
            - Bottom face (z = 0): Fully clamped (fix u_x = 0, u_y = 0, u_z = 0).
            - Top face (z = lz): Downward compressive load F_z = -F_mag distributed across nodes.
        """
        bcm = BoundaryConditionManager(mesh)
        bot_nodes = mesh.get_nodes_at_z(0.0)
        bcm.fix_nodes(bot_nodes)

        lz = float(np.max(mesh.node_coords[:, 2]))
        top_nodes = mesh.get_nodes_at_z(lz)
        n_top = len(top_nodes)
        f_per_node = float(F_mag) / max(1, n_top)
        bcm.add_nodal_force(top_nodes, [0.0, 0.0, -f_per_node], load_case=0)

        return bcm
