"""Unit and integration tests for Unified TopOpt Algorithm Adapters & Registry."""
import numpy as np
import pytest

from wht_topopt.algorithms import (
    FormulationType,
    OptimizerType,
    TopOptPairing,
    AlgorithmRegistry,
    AlgorithmCapabilities,
    OptimizationStepInput,
    OptimizationStepOutput,
    TopOptAlgorithmAdapter,
    MMAAdapter,
    GCMMAAdapter,
    OCAdapter,
    BESOAdapter,
    SEMDOTAdapter,
    LevelSetAdapter,
)


def test_algorithm_registry_listing_and_aliases():
    """Verify registry correctly lists all standard algorithms and resolves aliases."""
    available = AlgorithmRegistry.list_available()
    assert "mma" in available
    assert "gcmma" in available
    assert "oc" in available
    assert "beso" in available
    assert "semdot" in available
    assert "level_set" in available

    # Test alias resolution
    assert AlgorithmRegistry.resolve_name("simp_mma") == "mma"
    assert AlgorithmRegistry.resolve_name("simp") == "mma"
    assert AlgorithmRegistry.resolve_name("simp_oc") == "oc"
    assert AlgorithmRegistry.resolve_name("evolutionary") == "beso"
    assert AlgorithmRegistry.resolve_name("smooth_edged") == "semdot"
    assert AlgorithmRegistry.resolve_name("reaction_diffusion") == "level_set"

    # Default algorithm must be MMA
    default_opt = AlgorithmRegistry.get_default()
    assert isinstance(default_opt, MMAAdapter)
    assert default_opt.name == "mma"


def test_algorithm_capabilities():
    """Verify capabilities metadata for different algorithms."""
    mma_caps = AlgorithmRegistry.get_capabilities("mma")
    assert mma_caps.supports_multiple_constraints is True
    assert mma_caps.supports_stress_constraint is True
    assert mma_caps.supports_frequency_constraint is True
    assert mma_caps.supports_com_constraint is True
    assert mma_caps.variable_type == "continuous"

    gcmma_caps = AlgorithmRegistry.get_capabilities("gcmma")
    assert gcmma_caps.convergence_guarantee == "strictly_globally_convergent"
    assert gcmma_caps.supports_multiple_constraints is True

    oc_caps = AlgorithmRegistry.get_capabilities("oc")
    assert oc_caps.supports_multiple_constraints is False
    assert oc_caps.supports_stress_constraint is False

    beso_caps = AlgorithmRegistry.get_capabilities("beso")
    assert beso_caps.variable_type == "binary"


@pytest.mark.parametrize("algo_name", ["mma", "gcmma", "oc", "beso", "semdot", "level_set"])
def test_all_adapters_step_contract(algo_name: str):
    """Verify every registered adapter satisfies the OptimizationStepInput/Output contract."""
    n_elem = 20
    target_vol = 0.4
    rho_curr = np.full(n_elem, 0.5, dtype=np.float64)

    # Passive masks: element 0 solid, element 19 void
    passive_solid = np.zeros(n_elem, dtype=bool)
    passive_solid[0] = True
    passive_void = np.zeros(n_elem, dtype=bool)
    passive_void[-1] = True

    df_drho = -np.ones(n_elem, dtype=np.float64)
    g_vals = np.array([np.mean(rho_curr) / target_vol - 1.0])
    dg_drho = (1.0 / (n_elem * target_vol)) * np.ones((1, n_elem), dtype=np.float64)
    elem_vols = np.ones(n_elem, dtype=np.float64)

    adapter = AlgorithmRegistry.get(algo_name)
    assert isinstance(adapter, TopOptAlgorithmAdapter)
    adapter.initialize(n_elem, target_vol)

    step_input = OptimizationStepInput(
        iteration=1,
        rho=rho_curr,
        df_drho=df_drho,
        g_vals=g_vals,
        dg_drho=dg_drho,
        elem_volumes=elem_vols,
        passive_solid=passive_solid,
        passive_void=passive_void,
        f0_val=10.0,
    )

    out = adapter.step(step_input)
    assert isinstance(out, OptimizationStepOutput)
    assert out.rho_next.shape == (n_elem,)
    assert 0.0 <= np.min(out.rho_next) <= 1.0
    assert 0.0 <= np.max(out.rho_next) <= 1.0
    assert out.change >= 0.0
    assert out.subproblem_time_ms >= 0.0
    assert out.subproblem_converged is True

    # Passive regions strictly respected
    assert out.rho_next[0] == pytest.approx(1.0, rel=1e-5)
    assert out.rho_next[-1] == pytest.approx(1e-3, rel=1e-4)


def test_backward_compatible_update_call():
    """Verify adapter update() is drop-in compatible with legacy Optimizer.update()."""
    adapter = AlgorithmRegistry.get("mma")
    n_elem = 10
    rho = 0.5 * np.ones(n_elem)
    df = -np.ones(n_elem)
    g = np.array([0.0])
    dg = 0.1 * np.ones((1, n_elem))

    rho_new = adapter.update(rho, df, g, dg)
    assert isinstance(rho_new, np.ndarray)
    assert rho_new.shape == (n_elem,)
    assert np.all(rho_new >= 1e-3)
    assert np.all(rho_new <= 1.0)


def test_decoupled_create_system():
    """Verify decoupled Formulation and Optimizer creation via create_system()."""
    # 1. SIMP + MMA (Default pairing)
    sys1 = AlgorithmRegistry.create_system(formulation="simp", optimizer="mma")
    assert isinstance(sys1, MMAAdapter)
    assert sys1.formulation == FormulationType.SIMP
    assert sys1.optimizer_type == OptimizerType.MMA

    # 2. SIMP + GCMMA
    sys2 = AlgorithmRegistry.create_system(formulation="simp", optimizer="gcmma")
    assert isinstance(sys2, GCMMAAdapter)
    assert sys2.formulation == FormulationType.SIMP
    assert sys2.optimizer_type == OptimizerType.GCMMA

    # 3. SIMP + OC
    sys3 = AlgorithmRegistry.create_system(formulation="simp", optimizer="oc")
    assert isinstance(sys3, OCAdapter)
    assert sys3.formulation == FormulationType.SIMP
    assert sys3.optimizer_type == OptimizerType.OC

    # 4. BESO (Formulation with internal discrete ranking rule)
    sys4 = AlgorithmRegistry.create_system(formulation="beso")
    assert isinstance(sys4, BESOAdapter)
    assert sys4.formulation == FormulationType.BESO
    assert sys4.optimizer_type == OptimizerType.BESO_RANKING

    # 5. SEMDOT + MMA
    sys5 = AlgorithmRegistry.create_system(formulation="semdot", optimizer="mma")
    assert isinstance(sys5, SEMDOTAdapter)
    assert sys5.formulation == FormulationType.SEMDOT
    assert sys5.optimizer_type == OptimizerType.MMA

    # 6. Compound legacy name resolution
    sys6 = AlgorithmRegistry.create_system(formulation="simp_mma")
    assert isinstance(sys6, MMAAdapter)
    assert sys6.formulation == FormulationType.SIMP
    assert sys6.optimizer_type == OptimizerType.MMA

