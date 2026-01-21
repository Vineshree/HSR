import sys
import os
import numpy as np

# Adds the parent directory to the python path
sys.path.append(os.path.abspath('../'))

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
    # Create the state vector y = [epsilon, sigma, lambda2]
    y = [1e-9, -0.04, 0.0]
    
    ns, r = solver.compute_observables(y)
    
    # ns should be approx 1 + (-0.04) = 0.96
    assert ns == pytest.approx(0.96, rel=1e-3)

def test_r_positivity():
    """The tensor-to-scalar ratio r must always be positive for physical inflation."""
    solver = HSRSolver()
    y = [0.01, -0.02, 0.001]
    
    ns, r = solver.compute_observables(y)
    assert r > 0
