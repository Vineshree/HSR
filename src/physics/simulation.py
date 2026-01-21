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
    
    def run_stochastic_search(self, m_order, n_min=40, n_max=70):
        """
        Reflects Notebook 1: The exploratory 'Kinney-style' search.
        Seeds at N=0 and integrates forward to check for Graceful Exit.
        """
        y0 = self.generate_random_priors(m_order)
        
        def event_exit(t, y): return y[0] - 1.0
        event_exit.terminal = True
        
        sol = solve_ivp(self.solver.get_derivatives, (0, n_max), y0, 
                        events=event_exit, rtol=1e-6, atol=1e-8)
        
        # If it hit epsilon=1 within the 40-70 e-fold window
        if sol.t_events[0].size > 0:
            n_total = sol.t_events[0][0]
            if n_min <= n_total <= n_max:
                return self.solver.compute_observables(y0)
        
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

    def run_batch(self, n_sims, m_order, method='stochastic'):
        """
        The Master Batch Controller.
        method='stochastic' -> The Efstathiou/Kinney Landscape (Forward Search)
        method='backwards'  -> The Chen Modern Reconstruction (Backward Integration)
        """
        results = []
        for _ in tqdm(range(n_sims), desc=f"Running {method} batch"):
            # Route to the specific physics logic
            if method == 'stochastic':
                ns, r = self.run_stochastic_search(m_order)
            elif method == 'backwards':
                ns, r = self.run_single_backwards(m_order)
            
            # Only save the data if the simulation was "Physical"
            if ns is not None:
                results.append([ns, r])
                
        return np.array(results)
    
    def run_trajectory(self, y0, method='acm', n_max=75):
        """Runs a single trajectory and returns the full solver object (sol)."""
        func = self.solver.get_derivatives_acm if method == 'acm' else self.solver.get_derivatives
        
        def event_end(t, y): return y[0] - 1.0
        event_end.terminal = True
        
        sol = solve_ivp(func, (0, n_max), y0, t_eval=np.linspace(0, n_max, 500),
                        events=event_end, method='Radau', rtol=1e-8, atol=1e-10)
        return sol
    
    def run_batch_acm(self, n_sims, n_obs=55):
        """
        Notebook 4 Logic: Guided ACM simulations.
        Returns ns and r for models where inflation lasts > n_obs.
        """
        results = []
        for _ in tqdm(range(n_sims), desc="Running Guided ACM"):
            # Random starting conditions for epsilon, sigma, lambda_2
            y0 = [np.random.uniform(0.0001, 0.1), 
                  np.random.uniform(-0.2, 0.2), 
                  np.random.uniform(-0.01, 0.01)]
            
            sol = self.run_trajectory(y0, method='acm', n_max=200)
            
            # Check if inflation lasted long enough (e.g., 55 e-folds)
            if sol.t[-1] > n_obs:
                # Find the index corresponding to n_obs before the end
                target_t = sol.t[-1] - n_obs
                idx = np.argmin(np.abs(sol.t - target_t))
                
                y_obs = sol.y[:, idx]
                ns, r = self.solver.compute_observables(y_obs)
                results.append([ns, r])
                
        return np.array(results)