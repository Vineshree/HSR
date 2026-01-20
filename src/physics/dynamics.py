import numpy as np

class HSRSolver:
    """Handles the Hubble Slow-Roll (HSR) hierarchy physics."""
    def __init__(self, c_const: float = 0.0814514):
        self.c_const = c_const

    def get_derivatives(self, N: float, y: np.ndarray) -> np.ndarray:
        """HSR flow equations for l=10 truncation."""
        dy_dN = np.zeros_like(y)
        eps, sigma = y[0], y[1]
        lambdas = y[2:] 

        dy_dN[0] = eps * (sigma + 2 * eps)
        dy_dN[1] = -eps * (5 * sigma + 12 * eps) + 2 * lambdas[0]

        # Higher order dynamics
        for i in range(len(lambdas)):
            l = i + 2
            term1 = ((l - 1) / 2) * sigma + (l - 2) * eps
            l_next = lambdas[i + 1] if i + 1 < len(lambdas) else 0.0
            dy_dN[i + 2] = term1 * lambdas[i] + l_next
            
        return dy_dN

    def compute_observables(self, eps: float, sigma: float, xi: float):
        """Stewart-Lyth second-order relations."""
        r = 16 * eps * (1 - self.c_const * (sigma + 2 * eps))
        ns = (1 + sigma - (5 - 3 * self.c_const) * eps**2 
              - 0.25 * (3 - 5 * self.c_const) * sigma * eps 
              + 0.5 * (3 - self.c_const) * xi)
        return ns, r