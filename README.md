# Hubble Slow-Roll (HSR) Flow Framework

A modular Python framework for simulating inflationary universes using the Hubble Slow-Roll hierarchy and Algebraic Closure Models (ACM). This repository replicates and extends results from Kinney (2002) ( 	
https://doi.org/10.48550/arXiv.astro-ph/0206032) and Chongchitnan (2005), et al. ( 	
https://doi.org/10.48550/arXiv.astro-ph/0508355), providing a robust engine for Monte Carlo exploration of the $n_s - r$ plane.

---

## 1. Project Architecture
The project is structured to separate physical laws from numerical execution, following professional software engineering standards:

* **`src/physics/dynamics.py`**: The "Physics Engine." Contains the `HSRSolver` class which defines the differential equations for the HSR hierarchy and the specialized math for Algebraic Closure (ACM).
* **`src/physics/simulation.py`**: The "Execution Layer." Contains the `MonteCarloRunner` class. It manages `solve_ivp` integration, including forward/backward integration and trajectory tracking.
* **`src/visualization/plots.py`**: The "Analysis Layer." Dedicated module for generating publication-quality visualizations, theoretical attractor lines, and observational target regions.



---

## 2. Physics & Methodology

### HSR Hierarchy
The framework evolves the slow-roll parameters $\epsilon, \sigma, \lambda_2, \dots, \lambda_l$ as functions of the number of e-folds $N$. The system is truncated at a specified order $M$, with initial conditions following the **Kinney (2002)** hierarchical scaling where parameter ranges shrink by a factor of 10 for each higher order.

### Integration Modes
1.  **Forward Integration**: Direct evolution from initial conditions to a fixed number of e-folds.
2.  **Backward Integration**: Evolving forward to find the end of inflation ($\epsilon=1$), then "rewinding" back $N_{obs}$ e-folds to calculate observables at the horizon crossing.
3.  **Algebraic Closure (ACM)**: Uses the physical constraint $dR/dN = R^2/2$ to close the hierarchy, forcing chaotic models to converge onto the $R^2$ attractor.

---

## 3. Installation

To set up the environment and install the package in "editable" mode (so changes to `src` are immediately reflected in your notebooks):

```bash
# Clone the repository
git clone [https://github.com/yourusername/HSR-Flow.git](https://github.com/yourusername/HSR-Flow.git)
cd HSR-Flow

# Install in editable mode
pip install -e .
