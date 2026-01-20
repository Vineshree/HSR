import numpy as np
import pytest
from src.physics.dynamics import HSRSolver

def test_desitter_limit():
    """Test that derivatives are zero in a pure de Sitter background (eps=0, sigma=0)."""
    solver = HSRSolver()
    # State: [eps, sigma, lambda2, ..., lambda10]
    state = np.zeros(10) 
    derivs = solver.get_derivatives(0, state)
    
    assert np.all(derivs == 0), "Physics error: de Sitter background should not evolve."

def test_stewart_lyth_constants():
    """Verify the Stewart-Lyth constant C is calculated correctly."""
    solver = HSRSolver()
    # C should be approx 0.08145 based on your provided value
    assert np.isclose(solver.c_const, 0.0814514, atol=1e-5)

def test_first_order_ns_approximation():
    """
    Test that at very small epsilon, n_s - 1 is dominated by sigma.
    (n_s - 1 approx sigma)
    """
    solver = HSRSolver()
    eps = 1e-9
    sigma = -0.04
    xi = 0.0
    
    ns, r = solver.compute_observables(eps, sigma, xi)
    
    # Check if (ns - 1) is close to sigma
    assert np.isclose(ns - 1, sigma, atol=1e-3), "Second-order ns deviates too far from first-order limit."

def test_r_positivity():
    """The tensor-to-scalar ratio r must always be positive for physical inflation."""
    solver = HSRSolver()
    ns, r = solver.compute_observables(eps=0.01, sigma=-0.02, xi=0.001)
    assert r > 0, "Non-physical result: r cannot be negative."
