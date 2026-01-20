from scipy.integrate import solve_ivp
from .dynamics import HSRSolver  # <--- Internal dependency
import numpy as np

class MonteCarloRunner:
    def __init__(self, solver: HSRSolver):
        self.solver = solver

    def run_single(self, y_initial: list, n_obs: float):
        """Integrates a single realization."""
        sol = solve_ivp(
            self.solver.get_derivatives, 
            (0, 200), 
            y_initial, 
            rtol=1e-8, 
            atol=1e-8
        )
        # Add logic to find N_obs_point and return (ns, r)
        # ... (logic from your original code)
        return ns, r