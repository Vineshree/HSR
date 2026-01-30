import numpy as np
from scipy.integrate import solve_ivp


class HSRSolver:
    def __init__(self, c_const=0.0814514):
        self.c_const = c_const
        # ACM coefficients
        self.alpha = (3 - c_const)
        self.beta = -(5 - 3 * c_const)
        self.delta = -0.25 * (3 - 5 * c_const)

    def get_derivatives_acm(self, N, y):
        """3-D ACM Flow Equations with Algebraic Closure."""
        epsilon, sigma, lambda_2 = y
        
        d_eps_dN = epsilon * (sigma + 2 * epsilon)
        d_sigma_dN = -epsilon * (5 * sigma + 12 * epsilon) + 2 * lambda_2

        # The ACM Closure Logic
        R_y = sigma + self.beta * epsilon**2 + self.delta * sigma * epsilon + self.alpha * lambda_2
        G_y = (d_sigma_dN + 2 * self.beta * epsilon * d_eps_dN + 
               self.delta * (sigma * d_eps_dN + epsilon * d_sigma_dN) + 
               self.alpha * 0.5 * sigma * lambda_2)
        
        lambda_3_closed = (1.0 / self.alpha) * (0.5 * R_y**2 - G_y)
        d_lambda2_dN = 0.5 * sigma * lambda_2 + lambda_3_closed
        
        return [d_eps_dN, d_sigma_dN, d_lambda2_dN]

    def get_jacobian_acm(self, y, step=1e-9):
        """
        Computes the Jacobian matrix for the ACM system at state y.
        Useful for stability analysis and eigenvalue decomposition.
        """
        n = len(y)
        jac = np.zeros((n, n))
        # N is typically 0 for local stability analysis
        f0 = np.array(self.get_derivatives_acm(0, y))
        
        for j in range(n):
            dy = np.zeros(n)
            dy[j] = step
            f1 = np.array(self.get_derivatives_acm(0, y + dy))
            jac[:, j] = (f1 - f0) / step
            
        return jac
    
    def find_attractor_slope(solver, eps_val):
        # Construct a state that sits on the theoretical attractor
        y_attractor = np.array([eps_val, -2*eps_val, eps_val**2])
    
        # Get the Jacobian at this point
        J = solver.get_jacobian_acm(y_attractor)
    
        # Get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(J)
    
        # The eigenvector corresponding to the eigenvalue closest to 0 
        # is the direction of your blue line (the attractor manifold).
        idx = np.argmin(np.abs(eigvals))
        slope_vector = eigvecs[:, idx]
    
        return eigvals, slope_vector

    def compute_observables(self, y):
        eps = y[0]
        sig = y[1]
        lam2 = y[2] if len(y) > 2 else 0
        
        # Higher-order r and ns calculation from Efstathiou
        r = 16 * eps * (1 - self.c_const * (sig + 2 * eps)) 
        #r = 16 * eps

        delta_ns = sig - (5 - 3 * self.c_const) * (eps**2) \
                   - 0.25 * (3 - 5 * self.c_const) * sig * eps \
                   + 0.5 * (3 - self.c_const) * lam2
        
        return 1 + delta_ns, r

    def get_derivatives(self, t, y):
        epsilon = y[0]
        sigma = y[1]
        lambdas = y[2:]
        
        dydN = np.zeros_like(y)
        dydN[0] = epsilon * (sigma + 2 * epsilon)
        
        # The updated d(sigma)/dN from your notebook
        lambda2 = lambdas[0] if len(lambdas) > 0 else 0
        dydN[1] = -5 * epsilon * sigma - 12 * (epsilon**2) + 2 * lambda2
        
        # Hierarchy for l_lambda
        for i, l_val in enumerate(lambdas):
            l = i + 2
            term_next = lambdas[i+1] if (i + 1 < len(lambdas)) else 0.0
            prefactor = ((l - 1) / 2.0) * sigma + (l - 2) * epsilon
            dydN[i + 2] = prefactor * l_val + term_next
            
        return dydN
    def get_observables(self, sol, n_lookback=55):
        """Calculates n_s and r at the lookback point."""
        if sol.t[-1] < n_lookback:
            return None, None
        
        # Measure at N_end - 55
        n_obs = sol.t[-1] - n_lookback
        idx = (np.abs(sol.t - n_obs)).argmin()
        
        eps_obs = sol.y[0, idx]
        sig_obs = sol.y[1, idx]
        
        ns = 1 + sig_obs
        r = 16 * eps_obs
        return ns, r

    def run_stability_check(self, y0, n_max=1000):
        """Runs a trajectory and flags it as 'Exit' or 'Eternal'."""
        def exit_event(t, y): return y[0] - 1.0
        exit_event.terminal = True
            
        sol = solve_ivp(self.get_derivatives_acm, (0, n_max), y0, 
                            events=exit_event, method='Radau')
            
        status = "Exit" if sol.status == 1 else "Eternal"
        ns, r = self.get_observables(sol)
        return ns, r, status
        