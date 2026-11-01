"""Kratos standalone worker script executed in the Kratos Python environment.

Usage: python _kratos_worker.py <in.npz> <out.npz>
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import numpy as np


_ELEMENT_NAME_MAP = {
    "hex8": "SmallDisplacementElement3D8N",
    "tet4": "SmallDisplacementElement3D4N",
    "tet10": "SmallDisplacementElement3D10N",
    "hex20": "SmallDisplacementElement3D20N",
    "hex8_bbar": "SmallDisplacementBbarElement3D8N",
    "quad4": "SmallDisplacementElement2D4N",
    "tri3": "SmallDisplacementElement2D3N",
    "shell_thin": "ShellThinElementCorotational3D3N",
    "shell_thick": "ShellThickElementCorotational3D4N",
    "simp_hex8": "SmallDisplacementSIMPElement3D8N",
    "simp_tet4": "SmallDisplacementSIMPElement3D4N",
}


def _ensure_dlls() -> None:
    """Ensure Windows DLL directories are registered before importing Kratos extension modules."""
    import os
    from pathlib import Path

    kratos_install_root = Path(r"E:\kratos-build\install")
    if kratos_install_root.is_dir() and str(kratos_install_root) not in sys.path:
        sys.path.insert(0, str(kratos_install_root))

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        candidates = [
            Path(r"E:\kratos-build\install\libs"),
            Path(r"D:\PythonCodeStudy\kratos-deps\condadeps\Library\bin"),
            Path(r"D:\PythonCodeStudy\kratos-deps\mmg-5.8-novtk\bin"),
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.prefix) / "Scripts",
        ]
        for c in candidates:
            if c.is_dir():
                try:
                    os.add_dll_directory(str(c))
                except Exception:
                    pass


def run_worker(in_path: str, out_path: str) -> None:
    """Run Kratos FEA or TopOpt evaluation from serialized NPZ input."""
    _ensure_dlls()
    import KratosMultiphysics as KM
    import KratosMultiphysics.StructuralMechanicsApplication as SMA
    import KratosMultiphysics.LinearSolversApplication  # noqa: F401
    from KratosMultiphysics import python_linear_solver_factory

    try:
        import KratosMultiphysics.TopologyOptimizationApplication as TOA
        has_toa = True
    except Exception:
        TOA = None
        has_toa = False

    data = np.load(in_path, allow_pickle=False)
    nodes = np.asarray(data["nodes"], dtype=float)
    elems = np.asarray(data["elems"], dtype=np.int64)
    etype = str(data["etype"]).lower()
    method = str(data.get("solver", "direct")).lower()
    nu = float(data.get("nu", 0.3))

    youngs = np.asarray(data["youngs"], dtype=float) if "youngs" in data else np.ones(len(elems))
    fixed = np.asarray(data["fixed_dofs"], dtype=np.int64) if "fixed_dofs" in data else np.zeros(0, dtype=np.int64)
    loads = np.asarray(data["loads"], dtype=float)

    use_simp = bool(data.get("use_simp_elements", False)) and has_toa
    penal = float(data.get("penal", 3.0))
    e0 = float(data.get("e0", 1.0))
    emin = float(data.get("emin", 1e-6))
    x_phys = np.asarray(data["x_phys"], dtype=float) if "x_phys" in data else np.ones(len(elems))

    n_nodes = nodes.shape[0]
    dim = 2 if ("2d" in etype or nodes.shape[1] == 2 or "quad4" in etype or "tri3" in etype) else 3
    dof_per_node = dim

    # Reshape loads to (n_cases, n_nodes * dim)
    if loads.ndim == 1:
        loads = loads.reshape(1, -1)
    n_cases = loads.shape[0]

    # Resolve element type
    if use_simp:
        element_name = "SmallDisplacementSIMPElement3D8N" if "hex" in etype else "SmallDisplacementSIMPElement3D4N"
    else:
        element_name = _ELEMENT_NAME_MAP.get(etype, "SmallDisplacementElement3D8N")

    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.SetBufferSize(1)

    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.AddNodalSolutionStepVariable(KM.REACTION)
    mp.AddNodalSolutionStepVariable(SMA.POINT_LOAD)

    # Create nodes
    for i in range(n_nodes):
        node_id = i + 1
        x = float(nodes[i, 0])
        y = float(nodes[i, 1])
        z = float(nodes[i, 2]) if nodes.shape[1] >= 3 else 0.0
        mp.CreateNewNode(node_id, x, y, z)

    # Add DOFs
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)
    if dim == 3:
        KM.VariableUtils().AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z, mp)

    if use_simp and TOA is not None:
        prop = mp.CreateNewProperties(1)
        prop.SetValue(KM.YOUNG_MODULUS, e0)
        prop.SetValue(KM.POISSON_RATIO, nu)
        prop.SetValue(KM.DENSITY, 1.0)
        prop.SetValue(TOA.YOUNGS_MODULUS_0, e0)
        prop.SetValue(TOA.YOUNGS_MODULUS_MIN, emin)
        prop.SetValue(TOA.PENAL, penal)
        prop.SetValue(TOA.MAT_INTERP, "simp_modified")
        prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.LinearElastic3DLaw())

        for e_idx, conn in enumerate(elems):
            elem_id = e_idx + 1
            node_ids = [int(n) + 1 for n in conn]
            el = mp.CreateNewElement(element_name, elem_id, node_ids, prop)
            el.SetValue(TOA.X_PHYS, float(x_phys[e_idx]))
            el.SetValue(TOA.PENAL, penal)
            el.SetValue(TOA.MAT_INTERP, "simp_modified")
            el.SetValue(KM.YOUNG_MODULUS, e0)
    else:
        # Material bucketing or unique modulus
        uniq, inverse = np.unique(youngs, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        for pid, E in enumerate(uniq, 1):
            prop = mp.CreateNewProperties(pid)
            prop.SetValue(KM.YOUNG_MODULUS, float(E))
            prop.SetValue(KM.POISSON_RATIO, nu)
            prop.SetValue(KM.DENSITY, 1.0)
            law = SMA.LinearElastic2DPlaneStressLaw() if dim == 2 else SMA.LinearElastic3DLaw()
            prop.SetValue(KM.CONSTITUTIVE_LAW, law)

        for e_idx, conn in enumerate(elems):
            elem_id = e_idx + 1
            node_ids = [int(n) + 1 for n in conn]
            prop_id = int(inverse[e_idx]) + 1
            mp.CreateNewElement(element_name, elem_id, node_ids, mp.GetProperties()[prop_id])

    # Apply Dirichlet boundary conditions
    comps = [KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y]
    if dim == 3:
        comps.append(KM.DISPLACEMENT_Z)

    for d in np.unique(fixed):
        node_idx = int(d) // dof_per_node
        comp_idx = int(d) % dof_per_node
        mp.GetNode(node_idx + 1).Fix(comps[comp_idx])

    # Add point load conditions
    load_props = mp.CreateNewProperties(999999)
    cond_name = "PointLoadCondition2D1N" if dim == 2 else "PointLoadCondition3D1N"
    active_load_nodes = sorted({int(d) // dof_per_node for lc in loads for d in np.flatnonzero(lc)})
    for cid, n_idx in enumerate(active_load_nodes, 1):
        mp.CreateNewCondition(cond_name, cid, [n_idx + 1], load_props)

    # Configure linear solver with PARDISO direct acceleration
    has_mkl = False
    try:
        import KratosMultiphysics.LinearSolversApplication as LSA
        has_mkl = bool(LSA.HasMKL())
    except Exception:
        has_mkl = False

    if method in ("direct", "sparse_lu", "pardiso", "pardiso_lu"):
        solver_type = "LinearSolversApplication.pardiso_lu" if has_mkl else "LinearSolversApplication.sparse_lu"
    else:
        solver_type = "amgcl"

    params = {"solver_type": solver_type}
    if method in ("cg", "amgcl") or solver_type == "amgcl":
        params.update({"krylov_type": "cg", "tolerance": 1e-10, "max_iteration": 2000, "verbosity": 0})
    linear_solver = python_linear_solver_factory.ConstructSolver(KM.Parameters(json.dumps(params)))

    if use_simp and TOA is not None:
        scheme = TOA.ResidualBasedIncrementalUpdateStaticSIMPScheme()
    else:
        scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()

    builder = KM.ResidualBasedEliminationBuilderAndSolver(linear_solver) if solver_type == "LinearSolversApplication.pardiso_lu" else KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
    strategy.SetEchoLevel(0)

    u_out = np.zeros((n_cases, n_nodes * dof_per_node), dtype=float)
    dcdx_out = np.zeros(len(elems), dtype=float) if use_simp else None

    for c in range(n_cases):
        load_vec = loads[c]
        for node in mp.Nodes:
            i = node.Id - 1
            node.SetSolutionStepValue(KM.DISPLACEMENT, [0.0] * 3)
            fx = float(load_vec[dim * i])
            fy = float(load_vec[dim * i + 1])
            fz = float(load_vec[dim * i + 2]) if dim == 3 else 0.0
            node.SetSolutionStepValue(SMA.POINT_LOAD, [fx, fy, fz])

        if c == 0:
            strategy.Initialize()
        strategy.InitializeSolutionStep()
        strategy.Predict()
        strategy.SolveSolutionStep()
        strategy.FinalizeSolutionStep()

        for node in mp.Nodes:
            i = node.Id - 1
            disp = node.GetSolutionStepValue(KM.DISPLACEMENT)
            u_out[c, dim * i] = disp[0]
            u_out[c, dim * i + 1] = disp[1]
            if dim == 3:
                u_out[c, dim * i + 2] = disp[2]

    if use_simp and TOA is not None:
        # Compute and extract DCDX sensitivities
        sens_strategy = TOA.StructureAdjointSensitivityStrategy(mp, linear_solver, dim)
        sens_strategy.ComputeStrainEnergySensitivities()
        for e_idx, el in enumerate(mp.Elements):
            try:
                dcdx_out[e_idx] = el.GetValue(TOA.DCDX)
            except Exception:
                pass

    out_dict: Dict[str, Any] = {"u": u_out}
    if dcdx_out is not None:
        out_dict["dcdx"] = dcdx_out
    out_dict["info"] = json.dumps({"status": "ok", "solver": solver_type, "elements": len(elems)})

    np.savez(out_path, **out_dict)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python _kratos_worker.py <in.npz> <out.npz>")
        sys.exit(1)
    run_worker(sys.argv[1], sys.argv[2])
