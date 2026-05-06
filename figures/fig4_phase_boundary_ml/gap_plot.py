"""
Figure 4 left panel

Plot the energy gap between ground state and first excited state from DMRG calculations.
Similar style to experiments/2d_xxz/dmrg_sampled_analysis/ipr_plot.py
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import dill as pickle
import logging

# Fix for unpickling objects from original /qaml/ repo
import qaml.diagonalisation
import qaml.diagonalisation.twod
import qaml.diagonalisation.twod.dmrg
sys.modules['diagonalisation'] = qaml.diagonalisation
sys.modules['diagonalisation.twod'] = qaml.diagonalisation.twod
sys.modules['diagonalisation.twod.dmrg'] = qaml.diagonalisation.twod.dmrg

# Configure matplotlib to use LaTeX for text rendering (manuscript style)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_pickle(path: Path):
    """Load pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)

def load_gap_data(dmrg_dir: Path, jz_range):
    """
    Load gap data (E1 - E0) from DMRG results.
    
    Args:
        dmrg_dir: Path to DMRG data directory
        jz_range: List of Jz values to load
    
    Returns:
        jz_values: List of Jz values
        gaps: List of gap values (E1 - E0)
    """
    jz_values = []
    gaps = []
    
    for jz in jz_range:
        dmrg_file = dmrg_dir / f"XXZ_2d_jz_{jz:.1f}.pkl"
        if not dmrg_file.exists():
            logging.warning(f"Missing DMRG data for Jz={jz:.1f}")
            continue
        
        try:
            dmrg_res = load_pickle(dmrg_file)
            e0 = float(dmrg_res["result"].e)
            e1 = float(dmrg_res["result"].e1)
            gap = e1 - e0
            
            jz_values.append(jz)
            gaps.append(gap)
            
        except Exception as ex:
            logging.warning(f"Failed to load DMRG for Jz={jz:.1f}: {ex}")
            continue
    
    return jz_values, gaps

if __name__ == "__main__":
    # Configuration
    base_data = Path("../../data")
    output_dir = Path("plots/chi_320/gap")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # n=57 configuration
    n57_dmrg_dir = base_data / "spins_57" / "dmrg" / "chi_max_320_trunc_1e-06"
    
    # n=115 configuration
    n115_dmrg_dir = base_data / "spins_115" / "dmrg" / "chi_max_320_trunc_1e-06"
    
    # Jz range (1.0 to 6.0 in steps of 0.1)
    jz_range = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]
    
    # Load gap data for both system sizes
    logging.info("Loading n=57 gap data...")
    jz_57, gap_57 = load_gap_data(n57_dmrg_dir, jz_range)
    
    logging.info("Loading n=115 gap data...")
    jz_115, gap_115 = load_gap_data(n115_dmrg_dir, jz_range)
    
    # Convert to arrays
    jz_57 = np.array(jz_57)
    gap_57 = np.array(gap_57)
    jz_115 = np.array(jz_115)
    gap_115 = np.array(gap_115)
    
    # Create figure with manuscript-optimized aspect ratio (same as ipr_plot.py)
    fig, ax = plt.subplots(figsize=(2.0, 3.4))
    
    # Color scheme matching energy_vs_gap_combined_n57_n115.py
    color_57 = '#2ca02c'  # Green for n=57
    color_115 = '#9467bd'  # Purple for n=115
    
    # Plot with different markers for each system size
    ax.plot(jz_57, gap_57, marker="v", color=color_57, linewidth=1.2, markersize=4,
            markerfacecolor=color_57, markeredgewidth=0.8, alpha=0.8, label=None)
    ax.plot(jz_115, gap_115, marker="s", color=color_115, linewidth=1.2, markersize=4,
            markerfacecolor=color_115, markeredgewidth=0.8, alpha=0.8, label=None)
    
    ax.set_xlabel(r"$J_z$")
    ax.set_ylabel(r"$E_1 - E_0$")
    ax.set_yscale("log")
    
    # Add vertical dashed line at jz=1.6 to indicate phase boundary region
    ax.axvline(x=1.6, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1)
    
    # Set x-axis ticks to ensure jz=1.0 is included (like in the original plot)
    ax.set_xticks([1.0, 2.0, 4, 6])
    
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5, which="both")
    
    # Create custom legend with non-colored markers (manuscript style)
    legend_elements = [
        Line2D([0], [0], marker="v", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=5, markeredgewidth=0.8, label=r"$n=57$"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=5, markeredgewidth=0.8, label=r"$n=115$"),
    ]
    
    ax.legend(
        handles=legend_elements,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        loc="best",
        fontsize=8,
        handletextpad=0.5,
    )
    
    plt.tight_layout(pad=0.2)
    
    # Save as both PDF and PNG
    pdf_path = output_dir / "gap_vs_jz.pdf"
    png_path = output_dir / "gap_vs_jz.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    
    logging.info(f"Saved gap plot: {pdf_path} and {png_path}")
    logging.info(f"n=57: {len(jz_57)} data points, n=115: {len(jz_115)} data points")
