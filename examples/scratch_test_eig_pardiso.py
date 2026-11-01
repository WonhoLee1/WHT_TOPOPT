"""Test spla.eigsh with PyPardiso Minv on 3D column buckling."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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

ps = pypardiso.PyPardisoSolver()
u_free = ps.solve(K_ff, f_free)
u = np.zeros(mesh.ndof)
u[free] = u_free

Eg = (rho ** 3.0) * 10.0
KG = fem.assemble_KG(Eg, u)
KG_ff = KG[free, :][:, free].tocsr()
A_ff = (-KG_ff).tocsr()

t0 = time.time()
# Factorize K_ff in PyPardiso once
ps.factorize(K_ff)
Minv = spla.LinearOperator((len(free), len(free)), matvec=lambda v: ps.solve(K_ff, v))

# Run ARPACK with PyPardiso Minv
raw_vals, raw_vecs = spla.eigsh(
    A=A_ff,
    k=4,
    M=K_ff,
    which="LA",
    Minv=Minv,
    tol=1e-5,
    maxiter=500,
)
sort_idx = np.argsort(raw_vals)[::-1]
vals = raw_vals[sort_idx]
blfs = 1.0 / vals
t_eig = time.time() - t0
print(f"Eigensolver with PyPardiso took: {t_eig:.2f} seconds!")
print(f"BLFs lambda_1..4: {blfs}")
