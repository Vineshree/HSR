import matplotlib.pyplot as plt
import numpy as np

def set_publication_style():
    """Sets matplotlib parameters to CCM/Academic standards."""
    plt.rcParams.update({
        "text.usetex": True,      # Requires a LaTeX distribution
        "font.family": "serif",
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150
    })

def plot_ns_r_reconstruction(ns_results, r_results, valid_runs, save_path=None):
    """Generates the Figure 3 style scatter plot."""
    set_publication_style()
    
    plt.figure(figsize=(9, 7))
    plt.scatter(ns_results, r_results, s=2, alpha=0.5, color='#0077AA', 
                label=f'Simulated Models (n={valid_runs})')

    # Overplot the power-law inflation relation
    r_line = np.linspace(0.001, 16, 500)
    ns_line = 1 - (2 * r_line) / (16 - r_line)
    plt.plot(ns_line, r_line, color='red', linestyle='-', linewidth=1.5, label='Power-Law')

    plt.xlabel(r'$n_s$')
    plt.ylabel(r'$r$')
    plt.xlim(0.4, 1.1)
    plt.ylim(0, 4)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()