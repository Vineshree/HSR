import numpy as np
from tqdm import tqdm # Make sure to pip install tqdm
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

    def generate_random_priors(self, l_max=10, epsilon_max=0.8):
        """
        Generates a random initial state vector y_initial.
        Values are typically drawn from a uniform distribution (Priors).
        """
        # epsilon must be between 0 and 1 for inflation to happen
        eps = np.random.uniform(0.0001, epsilon_max)
        
        # sigma and higher lambdas are usually small
        sigma = np.random.uniform(-0.1, 0.1)
        
        # Higher order terms (lambda_2 to lambda_l)
        higher_lambdas = np.random.uniform(-0.005, 0.005, size=l_max-2)
        
        # Combine into the y_initial vector
        return np.concatenate(([eps, sigma], higher_lambdas))
    def run_batch(self, n_runs: int, l_max: int = 10, n_obs: float = 55.0):
        """Runs a large batch of simulations and returns valid results."""
        ns_results = []
        r_results = []

        # The progress bar is a life-saver for 125,000 runs
        for _ in tqdm(range(n_runs), desc="Simulating HSR Flow"):
            # 1. Generate new priors for every run
            y_init = self.generate_random_priors(l_max=l_max)
            
            # 2. Run the physics engine
            ns, r = self.run_single(y_init, n_obs=n_obs)
            
            # 3. Only save if the simulation was physically valid
            if ns is not None and r is not None:
                ns_results.append(ns)
                r_results.append(r)
                
        return np.array(ns_results), np.array(r_results)