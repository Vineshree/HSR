import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import scipy.linalg as la
from tqdm import tqdm

class MonteCarloRunner:
    def __init__(self, solver):
        self.solver = solver

    def run_stability_analysis(self):
        """
        Computes the fixed point and eigenvalues of the ACM system.
        This explains the 'vertical' nature of the chaos collapse.
        """
        # Small guess near the slow-roll attractor
        guess = [1e-6, -2e-6, 1e-12]
        
        # Levenberg-Marquardt handles the center manifold (λ=0) better than fsolve
        sol = root(lambda y: self.solver.get_derivatives_acm(0, y), guess, method='lm')
        f_point = sol.x
        
        # Compute Jacobian using the solver's method
        J = self.solver.get_jacobian_acm(f_point)
        eigvals = la.eigvals(J)
        
        return f_point, eigvals

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
    
    def run_trajectory(self, y0, method='acm', n_max=100):
        """Runs a single trajectory using the specified dynamics engine."""
        func = self.solver.get_derivatives_acm if method == 'acm' else self.solver.get_derivatives
        
        def event_end(t, y): return y[0] - 1.0
        event_end.terminal = True
        
        # Using Radau for ACM as it handles potential stiffness near epsilon=1
        sol = solve_ivp(func, (0, n_max), y0, t_eval=np.linspace(0, n_max, 250),
                        events=event_end, method='Radau', rtol=1e-8, atol=1e-10)
        return sol
    
    def run_batch_acm(self, n_sims, n_obs=55):
        """
        Simulates the Chaos Collapse.
        Evolves trajectories and calculates observables at N_obs before end.
        """
        results = []
        for _ in tqdm(range(n_sims), desc="Running Guided ACM"):
            # Sample initial conditions for the 3rd order system
            # Note: Ranges are chosen to explore the vertical manifold
            y0 = [np.random.uniform(0, 0.8),  # epsilon_0
                  np.random.uniform(-0.05, 0.05), # sigma_0
                  np.random.uniform(-0.001, 0.001)] # lambda_2_0
            
            sol = self.run_trajectory(y0, method='acm', n_max=250)
            
            # Logic: If inflation ends at N_end, we want data from N_end - 55
            if sol.t_events[0].size > 0:
                n_end = sol.t_events[0][0]
                if n_end > n_obs:
                    target_t = n_end - n_obs
                    # Interpolate or find closest index to N_obs
                    idx = np.argmin(np.abs(sol.t - target_t))
                    
                    y_at_obs = sol.y[:, idx]
                    ns, r = self.solver.compute_observables(y_at_obs)
                    results.append([ns, r])
                
        return np.array(results)