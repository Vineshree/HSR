import matplotlib.pyplot as plt
import numpy as np

def plot_hsr_results(ns_vals, r_vals, m_order, show_power_law=True):
    """
    Generates a professional n_s vs r scatter plot.
    """
    plt.figure(figsize=(10, 7))
    
    # 1. Filter outliers for a clean view
    mask = (r_vals < 10) & (ns_vals > 0.4) & (ns_vals < 1.4)
    
    # 2. Main Scatter Plot
    plt.scatter(ns_vals[mask], r_vals[mask], s=3, c='black', 
                alpha=0.6, label=f'Flow Models M={m_order}')
    
    # 3. Add the Red Power-Law Reference Line (from your notebook)
    if show_power_law:
        r_line = np.geomspace(1e-10, 0.4, 100)
        ns_line = 1 - 2 * r_line / (1 - r_line)
        plt.plot (ns_line, r_line, 'r-', lw=2, label='Power-Law Fixed Point')
    
    # 4. Professional Formatting
    plt.xlabel(r'Spectral Index $n_s$', fontsize=12)
    plt.ylabel(r'Tensor-to-Scalar Ratio $r$', fontsize=12)
    plt.yscale('log')
    plt.title(f'Inflationary Flow Reconstruction (M={m_order})', fontsize=14)
    plt.xlim(0.8, 1.1)  # Focused on the Planck region
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    return plt
def plot_acm_trajectories(solutions):
    """Creates a 3-panel plot showing parameter evolution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(solutions)))

    for i, sol in enumerate(solutions):
        axes[0].plot(sol.t, sol.y[0], color=colors[i], lw=1.5)
        axes[1].plot(sol.t, sol.y[1], color=colors[i], lw=1.5)
        axes[2].plot(sol.t, sol.y[2], color=colors[i], lw=1.5)

    axes[0].set_ylabel(r'$\epsilon(N)$')
    axes[1].set_ylabel(r'$\sigma(N)$')
    axes[2].set_ylabel(r'${}^2\lambda_H(N)$')
    
    for ax in axes:
        ax.set_xlabel('N (e-folds)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_acm_comparison(ns_vals, r_vals):
    plt.figure(figsize=(10, 7))
    
    # 1. Plot ACM results
    plt.scatter(ns_vals, r_vals, s=10, alpha=0.5, color='blue', 
                label=f'ACM Results ({len(ns_vals)} models)')

    # 2. Analytical Line: ns = 1 - 2/N, r = 12/N^2
    N_line = np.linspace(45, 75, 100)
    plt.plot(1 - 2/N_line, 12/(N_line**2), color='red', lw=2.5, 
             label=r'Theoretical $R^2$ Attractor')

    # 3. Planck 2018 Target Region
    plt.axvspan(0.961, 0.969, color='green', alpha=0.15, label='Planck $n_s$ 95% CL')

    plt.title("ACM Method: Chaos Collapsed to the Attractor", fontsize=14)
    plt.xlabel(r"$n_s$")
    plt.ylabel(r"$r$")
    plt.xlim(0.94, 1.0)
    plt.ylim(0, 0.02)
    plt.autoscale()
    plt.legend()
    plt.grid(True, alpha=0.2)
    return plt

def plot_acm_vs_attractor(results, title="ACM Chaos Collapse vs. Analytic Attractor"):
    """
    Plots Monte Carlo results and compares them to the analytic attractor line.
    
    Parameters:
    - results: np.array of [ns, r] values
    - title: Optional string for the plot title
    """
    if len(results) == 0:
        print("No results to plot.")
        return

    ns_vals = results[:, 0]
    r_vals = results[:, 1]
    
    plt.figure(figsize=(10, 7))
    
    # 1. Plot Monte Carlo Data
    plt.scatter(ns_vals, r_vals, s=10, color='blue', alpha=0.4, label=f'ACM Models ({len(results)})')
    
    # 2. Define and Plot the Analytic Attractor: n_s = 1 - r/8
    # Based on the derived locking: sigma = -2*eps and r = 16*eps
    # ns_plot_range = np.linspace(0.94, 1.0, 100)
    # r_range = np.linspace(0, 0.4, 100)
    # ns_theory = 1 - (r_range / 8) - (r_range**2 / 128) #correction terms to second order

    # plt.plot(ns_theory, r_range, color='red', linewidth=2, label='2nd-Order Attractor')
    
    # Use the actual coefficients from your physics.py
    C = 0.0814514
    r_fine = np.linspace(0, 0.4, 100)
    eps_fine = r_fine / 16.0

    # This replicates your compute_observables logic on the attractor (sig = -2*eps, lam2 = eps**2)
    ns_full_theory = 1 + (-2 * eps_fine) - (5 - 3*C)*(eps_fine**2) \
                 - 0.25*(3 - 5*C)*(-2*eps_fine)*eps_fine \
                 + 0.5*(3 - C)*(eps_fine**2)

    plt.plot(ns_full_theory, r_fine, color='red', label='Full Second-Order Theory')
    
    # 3. Observational Bounds
    plt.axvspan(0.9607, 0.9691, color='green', alpha=0.15, label='Planck $n_s$ 1-$\sigma$')
    
    # Formatting
    plt.xlabel(r'Spectral Index $n_s$', fontsize=14)
    plt.ylabel(r'Tensor-to-Scalar Ratio $r$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    
    # Ensure limits show the attractor clearly
    plt.xlim(min(ns_vals)-0.01, max(ns_vals)+0.01)
    plt.ylim(-0.02, max(r_vals)*1.1)
    plt.savefig('ns_r_plot.pdf', bbox_inches='tight', transparent=True)

    plt.show()

def plot_efstathiou_landscape(ns_vals, r_vals, m_order):
    """
    Visualizes the broad stochastic landscape with Efstathiou's 
    preferred observational window.
    """
    plt.figure(figsize=(10, 7))
    
    # 1. Scatter the stochastic results
    plt.scatter(ns_vals, r_vals, s=2, c='gray', alpha=0.3, label='Stochastic Landscape')
    
    # 2. Highlight the 'Efstathiou/Planck' Preferred Region (n_s ~ 0.96)
    plt.axvspan(0.958, 0.968, color='blue', alpha=0.1, label='Efstathiou/Bond Window')
    
    # 3. Add the Red Power-Law Line for context
    r_line = np.geomspace(1e-10, 0.4, 100)
    ns_line = 1 - 2 * r_line / (1 - r_line)
    plt.plot(ns_line, r_line, 'r--', lw=1, label='Power-Law Limit')

    plt.xlabel(r'Spectral Index $n_s$')
    plt.ylabel(r'Tensor-to-Scalar Ratio $r$')
    plt.yscale('log')
    plt.title(f'Inflationary Flow Landscape (M={m_order})')
    plt.xlim(0.85, 1.05)
    plt.ylim(1e-6, 1)
    plt.legend()
    plt.grid(True, alpha=0.2)
    return plt
def plot_chaos_collapse(results, title="3-D ACM Chaos Collapse"):
    """
    Plots the results of the ACM Monte Carlo simulation in the (ns, r) plane.
    """
    if len(results) == 0:
        print("No results to plot.")
        return

    ns_vals = results[:, 0]
    r_vals = results[:, 1]

    plt.figure(figsize=(8, 8))
    
    # 1. Plot the ACM generated points (the Vertical Manifold)
    plt.scatter(ns_vals, r_vals, s=15, color='blue', alpha=0.5, label='ACM Trajectories')

    # 2. Add Planck 2018 Observational Constraints (approximate)
    # n_s = 0.9649 ± 0.0042
    plt.axvspan(0.9607, 0.9691, color='green', alpha=0.1, label='Planck 2018 (1$\sigma$)')
    
    # 3. Reference line for n_s = 1 (Scale Invariant)
    plt.axvline(1.0, linestyle='--', color='black', alpha=0.3, label='Scale Invariant')

    # Formatting
    plt.xlabel(r'Spectral Index $n_s$', fontsize=14)
    plt.ylabel(r'Tensor-to-Scalar Ratio $r$', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Setting limits to highlight the collapse
    plt.xlim(0.93, 1.02)
    plt.ylim(-0.01, max(r_vals) * 1.2 if len(r_vals) > 0 else 0.1)
    
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_phase_space_flow(solver, eps_range=(0.001, 0.4), sig_range=(-1.0, 1.0)):
    # 1. MUST use linspace for streamplot to avoid ValueError
    # We increase the number of points to ensure resolution at the low end
    e_vec = np.linspace(eps_range[0], eps_range[1], 50) 
    s_vec = np.linspace(sig_range[0], sig_range[1], 50)
    E, S = np.meshgrid(e_vec, s_vec)
    
    U = np.zeros(E.shape) 
    V = np.zeros(E.shape) 
    
    for i in range(len(e_vec)):
        for j in range(len(s_vec)):
            # Fixed lam2 at attractor value for the slice
            y = [E[j,i], S[j,i], E[j,i]**2]
            dy = solver.get_derivatives_acm(0, y)
            U[j,i] = dy[0]
            V[j,i] = dy[1]

    plt.figure(figsize=(10, 8))
    magnitude = np.sqrt(U**2 + V**2)
    
    # Use density and color to show the 'drain' toward the attractor
    # Quick fix for your code block:
    strm = plt.streamplot(E, S, U, V, 
                      color=magnitude, 
                      cmap='plasma',      # High contrast colormap
                      density=1.2,        # Balanced density
                      linewidth=1.0)

    
    # Plot the analytic attractor line
    plt.plot(e_vec, -2*e_vec, 'red', linestyle='--', linewidth=3, 
             label=r'Analytic Attractor ($\sigma = -2\epsilon$)')
    
    # 2. Apply log scale AFTER plotting to see the capture behavior better
    plt.xscale('log')
    plt.xlim(eps_range) # Tighten limits
    plt.ylim(sig_range)
    
    plt.xlabel(r'Energy Scale $\epsilon$')
    plt.ylabel(r'Flow Parameter $\sigma$')
    plt.title('ACM Phase Space Flow: The Chaos Collapse Mechanism')
    plt.legend(facecolor='black', labelcolor='white')
    plt.colorbar(strm.lines, label='Flow Velocity')
    plt.gca().set_facecolor('#f0f0f0')       # Light gray background for visibility
    plt.savefig('stream_plot.pdf', bbox_inches='tight', transparent=True)
    plt.show()

def plot_relaxation_time(runner, n_samples=10):
    """
    Plots the deviation from the attractor vs time (N) for sample trajectories.
    """
    plt.figure(figsize=(10, 6))
    
    for _ in range(n_samples):
        # 1. Generate a random starting point based on your priors
        # These should match the priors used in your run_batch_acm
        y0 = [
            np.random.uniform(1e-4, 1e-2),  # epsilon
            np.random.uniform(-0.5, 0.5),   # sigma
            0.0                             # lambda2
        ]
        
        # 2. Pass y0 to the function
        path = runner.run_trajectory(y0) 
        
        if path.success:
            N = path.t
            eps = path.y[0]
            sig = path.y[1]
            
            # Calculate deviation from the attractor line: sigma = -2*epsilon
            # We want this to go to zero
            deviation = sig + 2 * eps
            
            # Plot vs "E-folds from End" to match your n_obs logic
            n_from_end = N[-1] - N
            plt.plot(n_from_end, deviation, alpha=0.6)

    plt.axvline(60, color='red', linestyle='--', label='Your n_obs (60)')
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5, label='Attractor')
    
    # We invert the axis to look backward from the end of inflation (N_end = 0)
    plt.gca().invert_xaxis()
    plt.xlabel('E-folds before end of inflation')
    plt.ylabel(r'Attractor Deviation ($\sigma + 2\epsilon$)')
    plt.title('Stability Analysis: Time Required for Chaos Collapse')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_stability_velocity(solver, eps_range=(1e-3, 0.4), sig_range=(-1.0, 1.0)):
    # Create grid (must be equally spaced for streamplot)
    e_vec = np.linspace(eps_range[0], eps_range[1], 50)
    s_vec = np.linspace(sig_range[0], sig_range[1], 50)
    E, S = np.meshgrid(e_vec, s_vec)
    
    U, V = np.zeros(E.shape), np.zeros(E.shape)
    
    for i in range(len(e_vec)):
        for j in range(len(s_vec)):
            y = [E[j,i], S[j,i], E[j,i]**2] # Fixed lam2 for slice
            dy = solver.get_derivatives_acm(0, y)
            U[j,i], V[j,i] = dy[0], dy[1]

    # Calculate Velocity (magnitude of the 'pull')
    velocity = np.sqrt(U**2 + V**2)

    plt.figure(figsize=(12, 8))
    
    # 1. Use a lighter background so dark lines are visible
    plt.gca().set_facecolor('#fdfdfd')
    
    # 2. Plot Streamlines with Velocity Coloring
    # 'Plasma' is great for showing 'hot' (fast) vs 'cold' (stagnant) zones
    strm = plt.streamplot(E, S, U, V, color=velocity, cmap='plasma', 
                          linewidth=1.5, density=1.2, arrowsize=1.5)
    
    # 3. Analytic Attractor
    plt.plot(e_vec, -2*e_vec, 'cyan', linestyle='--', linewidth=3, 
             label=r'Analytic Attractor ($\sigma = -2\epsilon$)')

    plt.xscale('log')
    plt.colorbar(strm.lines, label='Dynamical Velocity ($|dy/dN|$)')
    plt.xlabel(r'Energy Scale $\epsilon$')
    plt.ylabel(r'Flow Parameter $\sigma$')
    plt.title('Stability Analysis: Attractor Velocity vs. Chaos Zones')
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.legend(loc='upper left', facecolor='white', framealpha=1)
    plt.show()