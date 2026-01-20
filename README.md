# HSR
This repository implements a numerical framework for analyzing the phase-space dynamics of  the early universe. Specifically, it provides a consistent method for closing the  Hubble Slow-Roll (HSR) hierarchy by enforcing observational constraints from the  scalar spectral index ($n_s$) directly into the flow equations.

# Research Context
In standard HSR formalism, higher-order flow parameters (e.g., ${}^3\lambda_H$) are 
often arbitrarily truncated, leading to potential dynamical inconsistencies. This 
implementation utilizes a second-order observable manifold, $R(N)$, to analytically 
derive ${}^3\lambda_H$ at every point in the inflationary phase space.

## Technical Implementation
* **Hamilton-Jacobi Formalism**: The solver treats the Hubble parameter $H(\phi)$ as the 
    fundamental dynamical quantity, ensuring a potential-independent analysis of inflation.
* **Stiff ODE Integration**: Due to the coupling of the flow parameters, the resulting 
    system is numerically stiff. This implementation utilizes the **Backward 
    Differentiation Formula (BDF)** via `scipy.integrate.solve_ivp` to ensure 
    stability and convergence.
* **Modular Architecture**:
    * `physics.py`: Contains the algebraic derivation of the higher-order parameters.
    * `simulation.py`: The integration engine and parameter configuration.
    * `visualization.py`: Tools for generating phase-space portraits, attractors, 
        and nullclines.

## Key Features
- **Observationally-Driven**: Trajectories are pinned to the $n_s - 1 \approx -2/N$ 
  large-$N$ scaling characteristic of Starobinsky-like inflation.
- **Phase-Space Mapping**: Functions to visualize the evolution of $\epsilon$, $\sigma$, 
  and ${}^2\lambda_H$ to identify stable inflationary attractors.
