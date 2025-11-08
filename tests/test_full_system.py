import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_almost_equal
import logging

# --- FIX: Import the *real* GraphManager, not the non-existent MockGraphManager ---
from full_system import (
    convert_to_beta,
    HistoricalProfiler,
    BeliefEngine,
    HybridKellySolver,
    GraphManager  # <--- CORRECTED IMPORT
)

# Set logging to ERROR to silence noisy info logs during testing
logging.basicConfig(level=logging.ERROR)

# ==============================================================================
# ### COMPONENT 3 & 5: TEST MATH UTILITIES ###
# ==============================================================================

@pytest.mark.parametrize("mean, ci, expected_alpha, expected_beta", [
    (0.6, (0.4, 0.8), 13.8, 9.2),
    (0.8, (0.79, 0.81), 7679.2, 1919.8),
    (0.5, (0.1, 0.9), 1.388, 1.388),
])
def test_convert_to_beta_happy_paths(mean, ci, expected_alpha, expected_beta):
    """Tests the convert_to_beta math (C3/C5)"""
    alpha, beta = convert_to_beta(mean, ci)
    assert_allclose([alpha, beta], [expected_alpha, expected_beta], rtol=1e-3)

@pytest.mark.parametrize("mean, ci", [
    (0.0, (0.0, 0.1)),
    (1.0, (0.9, 1.0)),
    (0.5, (0.1, 0.4)),
    (0.5, (0.0, 1.0)),
])
def test_convert_to_beta_edge_cases(mean, ci):
    """Tests that edge cases default to a weak/uninformative prior (1, 1)"""
    alpha, beta = convert_to_beta(mean, ci)
    assert_allclose([alpha, beta], [1.0, 1.0])

def test_convert_to_beta_logical_rule():
    """Tests that a CI of zero (infinite confidence) returns 'inf'"""
    alpha, beta = convert_to_beta(0.8, (0.8, 0.8))
    assert alpha == float('inf')
    assert beta == float('inf')

def test_belief_engine_fusion(mocker):
    """Tests the Beta fusion math (C5)"""
    # 1. Setup
    # Mock the GraphManager call within BeliefEngine
    mocker.patch.object(GraphManager, 'get_model_brier_scores', return_value={
        'brier_internal_model': 0.08,
        'brier_expert_model': 0.05,
        'brier_crowd_model': 0.15,
    })
    
    beta_internal = (13.8, 9.2)
    p_experts = 0.45
    p_crowd = 0.55
    
    # 2. Action
    # --- FIX: Instantiate the graph in mock mode ---
    engine = BeliefEngine(GraphManager(is_mock=True)) 
    engine.k_brier_scale = 0.5 # Lock k for testing
    
    beta_experts = engine._impute_beta_from_point(p_experts, 'expert')
    beta_crowd = engine._impute_beta_from_point(p_crowd, 'crowd')
    
    (fused_alpha, fused_beta) = engine._fuse_betas([beta_internal, beta_experts, beta_crowd])
    
    # 3. Assert (These values are from our manual trace in the C5 review)
    assert_allclose([beta_experts[0], beta_experts[1]], [4.0, 4.9], rtol=1e-3)
    assert_allclose([beta_crowd[0], beta_crowd[1]], [1.265, 1.035], rtol=1e-3)
    assert_allclose([fused_alpha, fused_beta], [17.065, 13.135], rtol=1e-3)
    
    (mean, var) = engine._get_beta_stats(fused_alpha, fused_beta)
    assert_allclose(mean, 0.565, atol=1e-3)

# ==============================================================================
# ### COMPONENT 4: TEST BRIER SCORING ###
# ==============================================================================

def test_historical_profiler_brier_score():
    """Tests the Brier score calculation (C4)"""
    profiler = HistoricalProfiler(None, min_trades_threshold=3)
    
    # Test 1: Meets threshold
    df_group_good = pd.DataFrame([
        {'bet_price': 0.8, 'outcome': 1.0}, # (0.8 - 1.0)^2 = 0.04
        {'bet_price': 0.7, 'outcome': 1.0}, # (0.7 - 1.0)^2 = 0.09
        {'bet_price': 0.2, 'outcome': 0.0}, # (0.2 - 0.0)^2 = 0.04
    ])
    expected_brier = (0.04 + 0.09 + 0.04) / 3.0
    
    score = profiler._calculate_brier_score(df_group_good)
    assert_allclose(score, expected_brier) # ~0.0567
    
    # Test 2: Does not meet threshold
    df_group_bad = pd.DataFrame([
        {'bet_price': 0.8, 'outcome': 1.0},
        {'bet_price': 0.7, 'outcome': 1.0},
    ])
    score = profiler._calculate_brier_score(df_group_bad)
    assert score == 0.25 # Should return the default uninformative score

# ==============================================================================
# ### COMPONENT 6: TEST KELLY SOLVER (THE MOST CRITICAL TEST) ###
# ==============================================================================

@pytest.fixture
def arbitrage_setup():
    """A pytest fixture to set up the arbitrage scenario for C6 tests."""
    # --- FIX: Instantiate the real GraphManager in mock mode ---
    graph = GraphManager(is_mock=True) 
    
    solver = HybridKellySolver(num_samples_k=5000)
    # The mock graph's get_cluster_contracts will return the arb scenario
    contracts = graph.get_cluster_contracts("E_DUNE_3") 
    
    M = np.array([c['M'] for c in contracts])
    Q = np.array([c['Q'] for c in contracts])
    E = M - Q
    D = np.diag(Q)
    
    # Build the covariance matrix using the *real* method
    C = solver._build_covariance_matrix(graph, contracts)
    
    return {
        "graph": graph, "solver": solver, "contracts": contracts,
        "M": M, "Q": Q, "E": E, "D": D, "C": C
    }

def test_c6_build_covariance_matrix(arbitrage_setup):
    """Tests that the Covariance Matrix (C) is built correctly for the arb."""
    C = arbitrage_setup['C']
    
    # P(A) = 0.6, P(B) = 0.6
    # Var(A) = p(1-p) = 0.6 * 0.4 = 0.24
    # Var(B) = p(1-p) = 0.6 * 0.4 = 0.24
    # P(A,B) = P(A) = 0.6 (since A -> B)
    # Cov(A,B) = P(A,B) - P(A)P(B) = 0.6 - (0.6 * 0.6) = 0.6 - 0.36 = 0.24
    
    expected_C = np.array([[0.24, 0.24],
                           [0.24, 0.24]])
                           
    assert_array_almost_equal(C, expected_C)
    assert np.linalg.det(C) < 1e-9 # Assert matrix is singular

def test_c6_triage_triggers_numerical(arbitrage_setup):
    """Tests that the 'is_logical_rule' flag correctly triggers the numerical solver."""
    E = arbitrage_setup['E']
    Q = arbitrage_setup['Q']
    contracts = arbitrage_setup['contracts']
    solver = arbitrage_setup['solver']
    
    solver.edge_thresh = 1.0 # Set high so it doesn't trigger
    solver.q_thresh = 0.0  # Set low so it doesn't trigger
    
    is_numerical = solver._is_numerical_required(E, Q, contracts)
    
    assert is_numerical == True, "The 'is_logical_rule' flag must trigger the numerical solver"

def test_c6_analytical_solver_failure(arbitrage_setup):
    """
    Tests that the analytical solver (F* = D * C_inv * E) fails
    to find the arbitrage and returns a nonsensical [BUY, BUY] basket.
    This is a "known-failure" test.
    """
    solver = arbitrage_setup['solver']
    C, D, E = arbitrage_setup['C'], arbitrage_setup['D'], arbitrage_setup['E']
    
    F_star_analytical = solver._solve_analytical(C, D, E)
    
    assert F_star_analytical[0] > 0 # Incorrectly BUYS MKT_A
    assert F_star_analytical[1] > 0 # Correctly BUYS MKT_B

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
    
    # Assert the basket is correct:
    assert F_star_numerical[0] < -0.01, "Solver should SELL MKT_A (F_star[0] < 0)"
    assert F_star_numerical[1] > 0.01,  "Solver should BUY MKT_B (F_star[1] > 0)"
