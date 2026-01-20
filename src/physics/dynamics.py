import numpy as np

class HSRSolver:
    def __init__(self, c_const: float = -0.7296): # Default C_CONST
        self.c_const = c_const

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

    def compute_observables(self, y):
        eps = y[0]
        sig = y[1]
        lam2 = y[2] if len(y) > 2 else 0
        
        # Higher-order r and ns calculation
        r = 16 * eps * (1 - self.c_const * (sig + 2 * eps))
        delta_ns = sig - (5 - 3 * self.c_const) * (eps**2) \
                   - 0.25 * (3 - 5 * self.c_const) * sig * eps \
                   + 0.5 * (3 - self.c_const) * lam2
        
        return 1 + delta_ns, r