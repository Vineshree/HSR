import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

class MonteCarloRunner:
    def __init__(self, solver):
        self.solver = solver

    def generate_random_priors(self, m_order):
        """Kinney (2002) scaling: ranges shrink by factor of 10 for each higher order."""
        y = np.zeros(m_order + 1)
        y[0] = np.random.uniform(0, 0.8)   # epsilon
        y[1] = np.random.uniform(-0.5, 0.5) # sigma
        
        current_range = 0.05
        for l in range(2, m_order + 1):
            y[l] = np.random.uniform(-current_range, current_range)
            current_range /= 10.0
        return y

    def run_single_forward(self, m_order, n_obs=55):
        """
        Notebook 1 Logic: Simple forward integration.
        Integrates for exactly n_obs e-folds and takes the result.
        """
        y0 = self.generate_random_priors(m_order)
        # We integrate forward from N=0 to N=n_obs
        sol = solve_ivp(self.solver.get_derivatives, (0, n_obs), y0, rtol=1e-6)
        
        y_final = sol.y[:, -1]
        # Ensure we haven't already ended inflation (epsilon must be < 1)
        if y_final[0] < 1.0:
            return self.solver.compute_observables(y_final)
        return None, None

    def run_single_backwards(self, m_order, n_obs_range=(40, 70)):
        """
        Notebook 2 Logic: Forward-then-Backward.
        Finds the end of inflation first, then rewinds to N_obs.
        """
        y0 = self.generate_random_priors(m_order)
        
        def event_end(t, y): return y[0] - 1.0
        event_end.terminal = True
        
        # Forward search for epsilon=1
        sol_fwd = solve_ivp(self.solver.get_derivatives, (0, -1000), y0, 
                            events=event_end, rtol=1e-6)
        
        if sol_fwd.t_events[0].size > 0:
            y_end = sol_fwd.y[:, -1]
            n_obs = np.random.uniform(*n_obs_range)
            # Rewind
            sol_bwd = solve_ivp(self.solver.get_derivatives, (0, n_obs), y_end, rtol=1e-6)
            return self.solver.compute_observables(sol_bwd.y[:, -1])
        return None, None

    def run_batch(self, n_sims, m_order, method='backwards'):
        """Main interface for all notebooks."""
        results = []
        for _ in tqdm(range(n_sims), desc=f"Running {method} sim"):
            if method == 'backwards':
                ns, r = self.run_single_backwards(m_order)
            else:
                ns, r = self.run_single_forward(m_order)
            
            if ns is not None:
                results.append([ns, r])
        return np.array(results)