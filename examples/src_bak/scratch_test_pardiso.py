"""Test PyPardiso solve time on 3D column K_ff."""
import time
import numpy as np
import scipy.sparse as sp
import pypardiso

from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.boundary import BoundaryConditionManager

mesh = StructuredMesh3D(nx=16, ny=16, nz=80, lx=20.0, ly=20.0, lz=100.0)
fem = FEMEngine(mesh, nu=0.3, formulation="FI")
bcm = BoundaryConditionManager.compressed_column_3d(mesh, F_mag=1.0)

rho = np.ones(mesh.n_elem) * 0.45
K = fem.assemble_stiffness_matrix(rho, penal=3.0, e0=10.0, emin=1e-6)
free = bcm.free_dofs
K_ff = K[free, :][:, free].tocsr()
f_vec = bcm.F[:, 0]
f_free = f_vec[free]

t0 = time.time()
ps = pypardiso.PyPardisoSolver()
u_free = ps.solve(K_ff, f_free)
t_solve = time.time() - t0
print(f"PyPardiso solved 70k DOFs in: {t_solve:.4f} seconds! Max disp: {np.max(np.abs(u_free)):.6e}")

# Test multiple solves with same factorization (e.g. for ARPACK or repeated iterations)
t0 = time.time()
u2 = ps.solve(K_ff, f_free)
print(f"Subsequent solve in: {time.time()-t0:.4f} seconds!")
