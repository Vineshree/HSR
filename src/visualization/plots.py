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