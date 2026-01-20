import numpy as np
from scipy.integrate import solve_ivp
from .dynamics import HSRSolver

class MonteCarloRunner:
    def __init__(self, solver: HSRSolver):
        """
        Initializes the runner with a specific physics solver.
        """
        self.solver = solver

    def _end_of_inflation(self, N, y):
        """Event function to stop integration when epsilon = 1."""
        return y[0] - 1.0
    
    # Setting event properties for scipy.integrate
    _end_of_inflation.terminal = True
    _end_of_inflation.direction = 1

    def run_single(self, y_initial: np.ndarray, n_obs: float = 55.0, n_critical: float = 200.0):
        """
        Integrates a single realization and returns observables at N_obs.
        """
        try:
            sol = solve_ivp(
                self.solver.get_derivatives, 
                (0, n_critical), 
                y_initial, 
                events=self._end_of_inflation,
                dense_output=True,
                rtol=1e-8, atol=1e-8
            )
            
            # 1. Identify when inflation ended
            if sol.status == 1 and len(sol.t_events[0]) > 0:
                n_end = sol.t_events[0][0]
            else:
                n_end = n_critical

            # 2. Look back N_obs e-folds to find the horizon crossing point
            n_obs_point = n_end - n_obs

            # 3. Ensure the point is within the integration range
            if sol.t[0] < n_obs_point < n_end:
                y_obs = sol.sol(n_obs_point)
                
                # Check if still in slow-roll (epsilon < 1) at that point
                if y_obs[0] < 1.0:
                    # Unpack relevant values for ns and r
                    eps, sigma, xi = y_obs[0], y_obs[1], y_obs[2]
                    return self.solver.compute_observables(eps, sigma, xi)
            
            return None, None # Return None if realization is non-physical

        except Exception:
            return None, None