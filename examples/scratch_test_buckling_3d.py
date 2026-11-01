"""Scratch test for 3D column buckling FEM and eigensolver timing."""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.boundary import BoundaryConditionManager

print("Initializing 3D mesh (16x16x80)...")
t0 = time.time()
mesh = StructuredMesh3D(nx=16, ny=16, nz=80, lx=20.0, ly=20.0, lz=100.0)
print(f"Mesh created in {time.time()-t0:.2f}s: {mesh.n_elem} elems, {mesh.n_nodes} nodes, {mesh.ndof} DOFs")

t0 = time.time()
fem = FEMEngine(mesh, nu=0.3, formulation="FI")
print(f"FEMEngine initialized in {time.time()-t0:.2f}s, is_uniform={fem.is_uniform}, use_numba={fem.use_numba}")

t0 = time.time()
bcm = BoundaryConditionManager.compressed_column_3d(mesh, F_mag=1.0)
print(f"BCM created in {time.time()-t0:.2f}s, fixed DOFs={len(bcm.fixed_dofs)}, free DOFs={len(bcm.free_dofs)}")

rho = np.ones(mesh.n_elem) * 0.45
t0 = time.time()
K = fem.assemble_stiffness_matrix(rho, penal=3.0, e0=10.0, emin=1e-6)
print(f"K assembled in {time.time()-t0:.2f}s, nnz={K.nnz}")

t0 = time.time()
free = bcm.free_dofs
K_ff = K[free, :][:, free].tocsc()
f_vec = bcm.F[:, 0]
f_free = f_vec[free]
K_lu = spla.splu(K_ff)
u_free = K_lu.solve(f_free)
u = np.zeros(mesh.ndof)
u[free] = u_free
print(f"Prebuckling solve in {time.time()-t0:.2f}s, max displacement={np.max(np.abs(u)):.6e}")

t0 = time.time()
Eg = (rho ** 3.0) * 10.0
KG = fem.assemble_KG(Eg, u)
print(f"KG assembled in {time.time()-t0:.2f}s, nnz={KG.nnz}")

t0 = time.time()
KG_ff = KG[free, :][:, free].tocsc()
A_ff = (-KG_ff).tocsc()
Minv = spla.LinearOperator((len(free), len(free)), matvec=K_lu.solve)

raw_vals, raw_vecs = spla.eigsh(
    A=A_ff,
    k=4,
    M=K_ff,
    which="LA",
    Minv=Minv,
    tol=1e-6,
    maxiter=1000,
)
sort_idx = np.argsort(raw_vals)[::-1]
vals = raw_vals[sort_idx]
print(f"Eigenvalues solved in {time.time()-t0:.2f}s!")
print(f"Eigenvalues mu: {vals}")
blfs = 1.0 / vals
print(f"Buckling load factors lambda_1..4: {blfs}")
