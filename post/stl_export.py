"""STL and VTK mesh export and geometric quality verification for topology optimization.

Provides STLExporter with:
- Binary and ASCII STL file writers with face normal calculation.
- Legacy VTK 4.2 dataset exporter for structured/unstructured volumetric and surface meshes,
  supporting nodal and element scalar/vector fields (density, von Mises stress, displacement).
- Mesh quality verification using trimesh (watertightness, volume, area, Euler number, winding repair).
- Kratos TopologyOptimizationApplication IOUtilities wrapper for native ModelPart STL export.
"""

from __future__ import annotations

import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import trimesh

from ..core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh


class STLExporter:
    """Handles binary STL, ASCII STL, and legacy VTK export, plus topological verification."""

    @staticmethod
    def export_binary_stl(
        vertices: np.ndarray,
        faces: np.ndarray,
        filepath: Union[str, Path],
    ) -> Path:
        """Export triangle mesh to standard binary STL format.

        Parameters
        ----------
        vertices : np.ndarray of shape (V, 3)
            Vertex coordinates.
        faces : np.ndarray of shape (F, 3)
            Triangle node indices.
        filepath : str or Path
            Destination file path (.stl).

        Returns
        -------
        Path
            Resolved path of the created STL file.
        """
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        verts = np.asarray(vertices, dtype=np.float32)
        fcs = np.asarray(faces, dtype=np.int32)
        n_faces = len(fcs)

        # Compute facet normals: n = (v1 - v0) x (v2 - v0) / ||(v1 - v0) x (v2 - v0)||
        if n_faces > 0:
            v0 = verts[fcs[:, 0]]
            v1 = verts[fcs[:, 1]]
            v2 = verts[fcs[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(cross, axis=1, keepdims=True)
            normals = np.where(norm > 1e-12, cross / np.maximum(norm, 1e-12), 0.0).astype(np.float32)
        else:
            normals = np.empty((0, 3), dtype=np.float32)
            v0 = np.empty((0, 3), dtype=np.float32)
            v1 = np.empty((0, 3), dtype=np.float32)
            v2 = np.empty((0, 3), dtype=np.float32)

        with open(out_path, "wb") as f:
            # 80-byte header
            header = b"wht_topopt binary STL export"[:80].ljust(80, b"\0")
            f.write(header)
            # 4-byte unsigned integer count
            f.write(struct.pack("<I", n_faces))

            # Pack 50 bytes per triangle: 12 floats (normal + 3 verts) + 1 uint16 (attribute byte count = 0)
            for i in range(n_faces):
                data = struct.pack(
                    "<12fH",
                    normals[i, 0], normals[i, 1], normals[i, 2],
                    v0[i, 0], v0[i, 1], v0[i, 2],
                    v1[i, 0], v1[i, 1], v1[i, 2],
                    v2[i, 0], v2[i, 1], v2[i, 2],
                    0,
                )
                f.write(data)

        return out_path

    @classmethod
    def write_binary_stl(
        cls,
        filepath: Union[str, Path],
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Path:
        """Convenience wrapper accepting (filepath, vertices, faces) argument order."""
        return cls.export_binary_stl(vertices=vertices, faces=faces, filepath=filepath)

    @staticmethod
    def export_ascii_stl(
        vertices: np.ndarray,
        faces: np.ndarray,
        filepath: Union[str, Path],
        solid_name: str = "topopt_part",
    ) -> Path:
        """Export triangle mesh to human-readable ASCII STL format.

        Parameters
        ----------
        vertices : np.ndarray of shape (V, 3)
            Vertex coordinates.
        faces : np.ndarray of shape (F, 3)
            Triangle node indices.
        filepath : str or Path
            Destination file path.
        solid_name : str, default 'topopt_part'
            Identifier in solid statement.

        Returns
        -------
        Path
            Resolved path of the created STL file.
        """
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        verts = np.asarray(vertices, dtype=np.float64)
        fcs = np.asarray(faces, dtype=np.int64)
        n_faces = len(fcs)

        if n_faces > 0:
            v0 = verts[fcs[:, 0]]
            v1 = verts[fcs[:, 1]]
            v2 = verts[fcs[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(cross, axis=1, keepdims=True)
            normals = np.where(norm > 1e-12, cross / np.maximum(norm, 1e-12), 0.0)
        else:
            normals = np.empty((0, 3), dtype=np.float64)
            v0 = np.empty((0, 3), dtype=np.float64)
            v1 = np.empty((0, 3), dtype=np.float64)
            v2 = np.empty((0, 3), dtype=np.float64)

        lines = [f"solid {solid_name}\n"]
        for i in range(n_faces):
            nx, ny, nz = normals[i]
            x0, y0, z0 = v0[i]
            x1, y1, z1 = v1[i]
            x2, y2, z2 = v2[i]
            lines.append(
                f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n"
                f"    outer loop\n"
                f"      vertex {x0:.6e} {y0:.6e} {z0:.6e}\n"
                f"      vertex {x1:.6e} {y1:.6e} {z1:.6e}\n"
                f"      vertex {x2:.6e} {y2:.6e} {z2:.6e}\n"
                f"    endloop\n"
                f"  endfacet\n"
            )
        lines.append(f"endsolid {solid_name}\n")

        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return out_path

    @staticmethod
    def export_vtk(
        filepath: Union[str, Path],
        mesh: Optional[Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]] = None,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        nodal_fields: Optional[Dict[str, np.ndarray]] = None,
        element_fields: Optional[Dict[str, np.ndarray]] = None,
        title: str = "Topology Optimization Results",
    ) -> Path:
        """Export volumetric or surface mesh to legacy VTK 4.2 format.

        Supports both:
        1. Volumetric finite element mesh (Hex8, Quad4, Tet4) with element/nodal fields.
        2. Triangular boundary surface (vertices, faces) with point/cell data.

        Parameters
        ----------
        filepath : str or Path
            Destination file path (.vtk).
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh, optional
            If provided, exports the volumetric mesh.
        vertices : np.ndarray, optional
            Surface mesh vertices (if mesh is None).
        faces : np.ndarray, optional
            Surface mesh faces (if mesh is None).
        nodal_fields : dict of str -> np.ndarray, optional
            Point data arrays of shape (n_nodes,) or (n_nodes, 3).
        element_fields : dict of str -> np.ndarray, optional
            Cell data arrays of shape (n_elem,) or (n_elem, 3).
        title : str, default 'Topology Optimization Results'
            VTK header title.

        Returns
        -------
        Path
            Resolved path of the written VTK file.
        """
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if mesh is not None:
            # Volumetric FE mesh export
            nodes = mesh.node_coords
            if nodes.shape[1] == 2:
                # Pad 2D coordinates to 3D with z=0
                nodes_3d = np.column_stack([nodes, np.zeros(len(nodes), dtype=np.float64)])
            else:
                nodes_3d = nodes

            elements = mesh.elements
            n_cells = len(elements)
            nodes_per_cell = elements.shape[1]
            total_cell_ints = n_cells * (nodes_per_cell + 1)

            # Cell type mapping (VTK cell types)
            # 5: VTK_TRIANGLE, 9: VTK_QUAD, 10: VTK_TETRA, 12: VTK_HEXAHEDRON
            if nodes_per_cell == 8:
                cell_type = 12  # VTK_HEXAHEDRON
            elif nodes_per_cell == 4:
                cell_type = 9 if mesh.ndim == 2 else 10  # VTK_QUAD or VTK_TETRA
            elif nodes_per_cell == 3:
                cell_type = 5  # VTK_TRIANGLE
            else:
                cell_type = 7  # VTK_POLYGON

            cell_types = np.full(n_cells, cell_type, dtype=np.int32)

        elif vertices is not None and faces is not None:
            # Surface triangle mesh export
            verts = np.asarray(vertices, dtype=np.float64)
            if verts.shape[1] == 2:
                nodes_3d = np.column_stack([verts, np.zeros(len(verts), dtype=np.float64)])
            else:
                nodes_3d = verts

            elements = np.asarray(faces, dtype=np.int64)
            n_cells = len(elements)
            nodes_per_cell = 3
            total_cell_ints = n_cells * (3 + 1)
            cell_types = np.full(n_cells, 5, dtype=np.int32)  # VTK_TRIANGLE
        else:
            raise ValueError("Either 'mesh' or both 'vertices' and 'faces' must be provided.")

        n_pts = len(nodes_3d)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 4.2\n")
            f.write(f"{title}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            # POINTS
            f.write(f"POINTS {n_pts} float\n")
            for pt in nodes_3d:
                f.write(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")

            # CELLS
            f.write(f"CELLS {n_cells} {total_cell_ints}\n")
            for elem in elements:
                conn_str = " ".join(str(int(idx)) for idx in elem)
                f.write(f"{len(elem)} {conn_str}\n")

            # CELL_TYPES
            f.write(f"CELL_TYPES {n_cells}\n")
            for ct in cell_types:
                f.write(f"{ct}\n")

            # CELL_DATA
            if element_fields:
                f.write(f"CELL_DATA {n_cells}\n")
                for name, arr in element_fields.items():
                    arr_np = np.asarray(arr)
                    if arr_np.ndim == 1:
                        f.write(f"SCALARS {name} float 1\n")
                        f.write("LOOKUP_TABLE default\n")
                        for val in arr_np:
                            f.write(f"{float(val):.6e}\n")
                    elif arr_np.ndim == 2 and arr_np.shape[1] in (2, 3):
                        f.write(f"VECTORS {name} float\n")
                        for vec in arr_np:
                            vz = float(vec[2]) if len(vec) > 2 else 0.0
                            f.write(f"{float(vec[0]):.6e} {float(vec[1]):.6e} {vz:.6e}\n")

            # POINT_DATA
            if nodal_fields:
                f.write(f"POINT_DATA {n_pts}\n")
                for name, arr in nodal_fields.items():
                    arr_np = np.asarray(arr)
                    if arr_np.ndim == 1:
                        f.write(f"SCALARS {name} float 1\n")
                        f.write("LOOKUP_TABLE default\n")
                        for val in arr_np:
                            f.write(f"{float(val):.6e}\n")
                    elif arr_np.ndim == 2 and arr_np.shape[1] in (2, 3):
                        f.write(f"VECTORS {name} float\n")
                        for vec in arr_np:
                            vz = float(vec[2]) if len(vec) > 2 else 0.0
                            f.write(f"{float(vec[0]):.6e} {float(vec[1]):.6e} {vz:.6e}\n")

        return out_path

    @staticmethod
    def verify_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        fix: bool = True,
    ) -> Dict[str, Any]:
        """Compute topological and volumetric quality metrics using trimesh.

        Parameters
        ----------
        vertices : np.ndarray of shape (V, 3)
            Vertex coordinates.
        faces : np.ndarray of shape (F, 3)
            Triangle node indices.
        fix : bool, default True
            If True, automatically repairs face winding and outward normal orientation.

        Returns
        -------
        dict
            Dictionary containing:
            - is_watertight : bool
            - volume : float (positive solid volume)
            - surface_area : float
            - euler_number : int (V - E + F)
            - n_vertices : int
            - n_faces : int
            - is_winding_consistent : bool
            - repaired_vertices : np.ndarray
            - repaired_faces : np.ndarray
        """
        verts = np.asarray(vertices, dtype=np.float64)
        fcs = np.asarray(faces, dtype=np.int64)

        if len(verts) == 0 or len(fcs) == 0:
            return {
                "is_watertight": False,
                "volume": 0.0,
                "surface_area": 0.0,
                "euler_number": 0,
                "n_vertices": 0,
                "n_faces": 0,
                "is_winding_consistent": False,
                "repaired_vertices": verts,
                "repaired_faces": fcs,
            }

        tm = trimesh.Trimesh(vertices=verts, faces=fcs, process=False)

        if fix:
            try:
                trimesh.repair.fix_normals(tm)
                trimesh.repair.fix_winding(tm)
            except Exception:
                pass

        is_watertight = bool(tm.is_watertight)
        # For watertight solids, tm.volume is exact; for non-watertight, fall back to divergence sum
        if is_watertight and tm.volume > 0.0:
            vol = float(tm.volume)
        else:
            # Signed volume
            v0 = tm.vertices[tm.faces[:, 0]]
            v1 = tm.vertices[tm.faces[:, 1]]
            v2 = tm.vertices[tm.faces[:, 2]]
            vol = abs(float(np.sum(v0 * np.cross(v1, v2)) / 6.0))

        area = float(tm.area)
        euler = int(tm.euler_number)
        winding_consistent = bool(tm.is_winding_consistent)

        return {
            "is_watertight": is_watertight,
            "volume": vol,
            "surface_area": area,
            "euler_number": euler,
            "n_vertices": len(tm.vertices),
            "n_faces": len(tm.faces),
            "is_winding_consistent": winding_consistent,
            "repaired_vertices": np.asarray(tm.vertices),
            "repaired_faces": np.asarray(tm.faces),
        }

    @staticmethod
    def export_via_kratos_io(
        kratos_model_part: Any,
        output_stl_path: Union[str, Path],
    ) -> bool:
        """Export Kratos ModelPart conditions to STL using TopologyOptimizationApplication.

        Parameters
        ----------
        kratos_model_part : Kratos.ModelPart
            ModelPart instance containing surface triangle conditions.
        output_stl_path : str or Path
            Destination path for the output STL file.

        Returns
        -------
        bool
            True if export succeeded, False otherwise.
        """
        out_path = Path(output_stl_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import KratosMultiphysics.TopologyOptimizationApplication as KTO
            io_util = KTO.IOUtilities()
            io_util.WriteSurfaceAsSTLFile(str(out_path), kratos_model_part)
            return out_path.is_file() and out_path.stat().st_size > 0
        except Exception as e:
            # Fallback: extract condition nodes & elements directly to binary STL
            try:
                nodes_map = {}
                for node in kratos_model_part.Nodes:
                    nodes_map[node.Id] = [node.X, node.Y, node.Z]

                coords = []
                faces = []
                node_id_to_idx = {}
                for idx, (nid, xyz) in enumerate(nodes_map.items()):
                    coords.append(xyz)
                    node_id_to_idx[nid] = idx

                for cond in kratos_model_part.Conditions:
                    c_nodes = [node_id_to_idx[n.Id] for n in cond.GetNodes()]
                    if len(c_nodes) == 3:
                        faces.append(c_nodes)

                if faces:
                    STLExporter.export_binary_stl(np.array(coords), np.array(faces), out_path)
                    return True
            except Exception:
                pass
            return False
