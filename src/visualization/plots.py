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

    # 2. Analytical Starobinsky Line: ns = 1 - 2/N, r = 12/N^2
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
    plt.legend()
    plt.grid(True, alpha=0.2)
    return plt
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