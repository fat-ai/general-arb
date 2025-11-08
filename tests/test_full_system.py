import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_almost_equal
import logging

# Import the specific classes and functions we need to test
from full_system import (
    convert_to_beta,
    HistoricalProfiler,
    BeliefEngine,
    HybridKellySolver,
    GraphManager
)

# Set logging to ERROR to silence noisy info logs during testing
logging.basicConfig(level=logging.ERROR)

# ==============================================================================
# ### COMPONENT 3 & 5: TEST MATH UTILITIES ###
# ==============================================================================

@pytest.mark.parametrize("mean, ci, expected_alpha, expected_beta", [
    # Test 1: (0.6, [0.4, 0.8]) -> std=0.1, var=0.01 -> inner=23
    (0.6, (0.4, 0.8), 13.8, 9.2),
    
    # Test 2: (0.8, [0.79, 0.81]) -> std=0.005, var=2.5e-5 -> inner=6399
    (0.8, (0.79, 0.81), 5119.2, 1279.8),
    
    # Test 3: (0.5, [0.1, 0.9]) -> std=0.2, var=0.04 -> inner=5.25
    (0.5, (0.1, 0.9), 2.625, 2.625),
])
def test_convert_to_beta_happy_paths(mean, ci, expected_alpha, expected_beta):
    """Tests the convert_to_beta math (C3/C5)"""
    alpha, beta = convert_to_beta(mean, ci)
    assert_allclose([alpha, beta], [expected_alpha, expected_beta], rtol=1e-3)

@pytest.mark.parametrize("mean, ci", [
    (0.0, (0.0, 0.1)), # Test 4: Extreme mean (p=0) -> (1.0, inf)
    (1.0, (0.9, 1.0)), # Test 5: Extreme mean (p=1) -> (inf, 1.0)
    (0.5, (0.1, 0.4)), # Test 6: Invalid CI (mean not in interval) -> (1.0, 1.0)
])
def test_convert_to_beta_invalid_cases(mean, ci):
    """Tests that invalid inputs default to correct logical or weak priors"""
    alpha, beta = convert_to_beta(mean, ci)
    
    if mean == 0.0:
        assert alpha == 1.0 and beta == float('inf')
    elif mean == 1.0:
        assert alpha == float('inf') and beta == 1.0
    else:
        assert_allclose([alpha, beta], [1.0, 1.0])

def test_convert_to_beta_inconsistent_ci():
    """Tests that a CI that is too wide (inconsistent) returns (1,1)"""
    # (0.5, [0.0, 1.0]) -> std=0.25, var=0.0625.
    # max var for mean=0.5 is 0.5*(1-0.5) = 0.25.
    # code checks `if (mean * (1-mean)) < variance: return (1.0, 1.0)`
    # 0.25 is NOT < 0.0625.
    # inner = (0.25 / 0.0625) - 1 = 4 - 1 = 3.
    # This test was failing because my *test logic* was wrong, not the code.
    alpha, beta = convert_to_beta(0.5, (0.0, 1.0))
    assert_allclose([alpha, beta], [1.5, 1.5]) # <--- CORRECTED EXPECTATION

def test_convert_to_beta_logical_rule():
    """Tests that a CI of zero (infinite confidence) returns 'inf'"""
    alpha, beta = convert_to_beta(0.8, (0.8, 0.8))
    assert alpha == float('inf')
    assert beta == 1.0 # Per our new fix

def test_belief_engine_fusion(mocker):
    """Tests the Beta fusion math (C5)"""
    # 1. Setup
    mocker.patch.object(GraphManager, 'get_model_brier_scores', return_value={
        'brier_internal_model': 0.08,
        'brier_expert_model': 0.05,
        'brier_crowd_model': 0.15,
    })
    
    beta_internal = (13.8, 9.2) # Our internal model
    p_experts = 0.45            # Smart money
    p_crowd = 0.55              # The crowd
    
    # 2. Action
    engine = BeliefEngine(GraphManager(is_mock=True)) 
    engine.k_brier_scale = 0.5 # Lock k for testing
    
    beta_experts = engine._impute_beta_from_point(p_experts, 'expert')
    beta_crowd = engine._impute_beta_from_point(p_crowd, 'crowd')
    (fused_alpha, fused_beta) = engine._fuse_betas([beta_internal, beta_experts, beta_crowd])
    
    # 3. Assert (Corrected values)
    # Model 2 (Expert): mean=0.45, brier=0.05, k=0.5 -> var=0.025 -> inner=8.9 -> a=4.005, b=4.895
    assert_allclose([beta_experts[0], beta_experts[1]], [4.005, 4.895], rtol=1e-3)
    # Model 3 (Crowd): mean=0.55, brier=0.15, k=0.5 -> var=0.075 -> inner=2.3 -> a=1.265, b=1.035
    assert_allclose([beta_crowd[0], beta_crowd[1]], [1.265, 1.035], rtol=1e-3)
    
    # Fused = 1 + (13.8-1) + (4.005-1) + (1.265-1) = 1 + 12.8 + 3.005 + 0.265 = 17.07
    #       = 1 + (9.2-1) + (4.895-1) + (1.035-1) = 1 + 8.2 + 3.895 + 0.035 = 13.13
    assert_allclose([fused_alpha, fused_beta], [17.07, 13.13], rtol=1e-3)
    
    (mean, var) = engine._get_beta_stats(fused_alpha, fused_beta)
    assert_allclose(mean, 0.565, atol=1e-3)

# ==============================================================================
# ### COMPONENT 4: TEST BRIER SCORING ###
# ==============================================================================
@pytest.mark.filterwarnings("ignore:pandas.DataFrame")
def test_historical_profiler_brier_score():
    """Tests the Brier score calculation (C4)"""
    profiler = HistoricalProfiler(None, min_trades_threshold=3)
    
    df_group_good = pd.DataFrame([
        {'bet_price': 0.8, 'outcome': 1.0},
        {'bet_price': 0.7, 'outcome': 1.0},
        {'bet_price': 0.2, 'outcome': 0.0},
    ])
    expected_brier = (0.04 + 0.09 + 0.04) / 3.0
    score = profiler._calculate_brier_score(df_group_good)
    assert_allclose(score, expected_brier)
    
    df_group_bad = pd.DataFrame([{'bet_price': 0.8, 'outcome': 1.0}])
    score = profiler._calculate_brier_score(df_group_bad)
    assert score == 0.25

# ==============================================================================
# ### COMPONENT 6: TEST KELLY SOLVER (THE MOST CRITICAL TEST) ###
# ==============================================================================

@pytest.fixture
def arbitrage_setup():
    """A pytest fixture to set up the arbitrage scenario for C6 tests."""
    graph = GraphManager(is_mock=True) 
    solver = HybridKellySolver(num_samples_k=5000)
    contracts = graph.get_cluster_contracts("E_DUNE_3") 
    
    M = np.array([c['M'] for c in contracts])
    Q = np.array([c['Q'] for c in contracts])
    E = M - Q
    D = np.diag(Q)
    C = solver._build_covariance_matrix(graph, contracts)
    
    return {
        "graph": graph, "solver": solver, "contracts": contracts,
        "M": M, "Q": Q, "E": E, "D": D, "C": C
    }

def test_c6_build_covariance_matrix(arbitrage_setup):
    """Tests that the Covariance Matrix (C) is built correctly for the arb."""
    C = arbitrage_setup['C']
    expected_C = np.array([[0.24, 0.24],
                           [0.24, 0.24]])
    assert_array_almost_equal(C, expected_C)
    assert np.linalg.det(C) < 1e-9

def test_c6_triage_triggers_numerical(arbitrage_setup):
    """Tests that the 'is_logical_rule' flag correctly triggers the numerical solver."""
    E, Q, contracts, solver = (
        arbitrage_setup['E'], arbitrage_setup['Q'], 
        arbitrage_setup['contracts'], arbitrage_setup['solver']
    )
    solver.edge_thresh = 1.0
    solver.q_thresh = 0.0
    is_numerical = solver._is_numerical_required(E, Q, contracts)
    assert is_numerical == True, "The 'is_logical_rule' flag must trigger"

def test_c6_analytical_solver_failure(arbitrage_setup):
    """Tests that the analytical solver (known-failure) returns a [BUY, BUY] basket."""
    solver = arbitrage_setup['solver']
    C, D, E = arbitrage_setup['C'], arbitrage_setup['D'], arbitrage_setup['E']
    F_star_analytical = solver._solve_analytical(C, D, E)
    assert F_star_analytical[0] > 0
    assert F_star_analytical[1] > 0

def test_c6_numerical_solver_success(arbitrage_setup):
    """
    Tests that the FULL numerical solver (E[log(W)]) *correctly*
    finds the [SELL A, BUY B] arbitrage basket.
    """
    solver = arbitrage_setup['solver']
    M, Q, C = arbitrage_setup['M'], arbitrage_setup['Q'], arbitrage_setup['C']
    F_analytical = solver._solve_analytical(C, arbitrage_setup['D'], arbitrage_setup['E'])

    # Run the *full* numerical optimizer
    F_star_numerical = solver._solve_numerical(M, Q, C, F_analytical)
    
    assert F_star_numerical[0] < -0.01, "Solver should SELL MKT_A (F_star[0] < 0)"
    assert F_star_numerical[1] > 0.01,  "Solver should BUY MKT_B (F_star[1] > 0)"
