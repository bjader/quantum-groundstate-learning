"""
Left panel in main text figure 2. You first need to run
data_generation/basis_optimization/create_unified_structure.py to get the n=57
and n=115 basis optimazion results in the same format.
"""

import logging
import sys
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np

# Fix for unpickling objects from original /qaml/ repo
# Import the modules first, then create aliases for old paths
import qaml.diagonalisation
import qaml.diagonalisation.twod
import qaml.diagonalisation.twod.dmrg
sys.modules['diagonalisation'] = qaml.diagonalisation
sys.modules['diagonalisation.twod'] = qaml.diagonalisation.twod
sys.modules['diagonalisation.twod.dmrg'] = qaml.diagonalisation.twod.dmrg

# Configure matplotlib to use LaTeX for text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def sorted_xy(d):
    xs = sorted(d.keys())
    return xs, [d[x] for x in xs]

def infer_jz_from_fname(path: Path):
    stem = path.stem
    if "jz_" not in stem:
        return None
    try:
        return float(stem.split("jz_")[1].split("_")[0])
    except Exception:
        return None

def latest_checkpoint_file(run_dir: Path):
    best = None
    best_i = None
    for p in run_dir.glob("*.pkl"):
        try:
            i = int(p.stem)
        except Exception:
            continue
        if best_i is None or i > best_i:
            best_i = i
            best = p
    return best

def choose_run_dir(base: Path, pattern: str):
    dirs = [p for p in base.glob(pattern) if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)

def load_vopt_stage2_data(n_spins, vopt_unified_path, dmrg_dir, timestamp, jz_range, postprocessing='random_flip'):
    """
    Load variational optimization stage2 results for a given system size.
    
    Args:
        n_spins: Number of spins (57 or 115)
        vopt_unified_path: Path to unified variational optimization results
        dmrg_dir: Path to DMRG reference data
        timestamp: Timestamp for the QPU run
        jz_range: Range of Jz values to process
        postprocessing: Postprocessing method ('random_flip' or 'post_select')
    
    Returns:
        diff_to_gap: Dictionary mapping Jz to |E - E_DMRG| / gap
        energies: Dictionary mapping Jz to energy values
        dmrg_ground: Dictionary mapping Jz to DMRG ground state energy
        dmrg_excited: Dictionary mapping Jz to DMRG excited state energy
    """
    diff_to_gap = {}
    energies = {}
    dmrg_ground = {}
    dmrg_excited = {}
    
    # Path to the n-specific subdirectory
    n_dir = vopt_unified_path / f"n_{n_spins}"
    if not n_dir.exists():
        logging.error(f"Unified directory not found: {n_dir}")
        return diff_to_gap
    
    # Both system sizes use the same energy key
    energy_key = "final_global_energy"
    
    for jz in jz_range:
        # Load DMRG reference
        dmrg_file = dmrg_dir / f"XXZ_2d_jz_{jz:.1f}.pkl"
        if not dmrg_file.exists():
            logging.warning(f"Missing DMRG data for n={n_spins}, Jz={jz:.1f}")
            continue
        
        try:
            dmrg_res = load_pickle(dmrg_file)
            dmrg_e = float(dmrg_res["result"].e)
            e1 = float(dmrg_res["result"].e1)
        except Exception as ex:
            logging.warning(f"Failed to load DMRG for n={n_spins}, Jz={jz:.1f}: {ex}")
            continue
        
        gap = e1 - dmrg_e
        if gap == 0:
            continue
        
        # For n=115, jz=3.3 uses a different timestamp (only for post_select)
        if n_spins == 115 and abs(jz - 3.3) < 0.01 and postprocessing == 'post_select':
            ts_to_use = 1773854302
        else:
            ts_to_use = timestamp
        
        # Find stage2 symlink in unified structure
        stage2_link = n_dir / f"ts_{ts_to_use}_jz_{jz:.1f}_stage2_{postprocessing}"
        
        if not stage2_link.exists():
            logging.warning(f"Missing vopt stage2 link for n={n_spins}, Jz={jz:.1f}, postprocessing={postprocessing}")
            continue
        
        # Determine which result file to use based on system size and postprocessing
        if n_spins == 57 and postprocessing == 'random_flip':
            # n=57 random_flip data uses res.pkl
            result_file = stage2_link / "res.pkl"
        else:
            # n=115 random_flip and all post_select data use final_res.pkl
            result_file = stage2_link / "final_res.pkl"
        
        if not result_file.exists():
            logging.warning(f"Missing {result_file.name} for n={n_spins}, Jz={jz:.1f}, postprocessing={postprocessing}")
            continue
        
        try:
            resq = load_pickle(result_file)
            if energy_key not in resq:
                logging.warning(f"Missing key '{energy_key}' for n={n_spins}, Jz={jz:.1f}, postprocessing={postprocessing}")
                continue
            e_final = resq[energy_key]
            diff_to_gap[jz] = abs((e_final - dmrg_e) / gap)
            energies[jz] = e_final
            dmrg_ground[jz] = dmrg_e
            dmrg_excited[jz] = e1
        except Exception as ex:
            logging.warning(f"Failed to load vopt results for n={n_spins}, Jz={jz:.1f}, postprocessing={postprocessing}: {ex}")
            continue
    
    return diff_to_gap, energies, dmrg_ground, dmrg_excited

if __name__ == "__main__":
    
    # Configuration
    base_data = Path("../../data")
    vopt_unified_path = Path("../../data_generation/basis_optimization/saved_opt_res_qpu_unified")
    
    # n=57 configuration
    n57_dmrg_dir = base_data / "spins_57" / "dmrg" / "chi_max_320_trunc_1e-06"
    n57_timestamp = 1773299045
    n57_jz_range = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]
    
    # n=115 configuration
    n115_dmrg_dir = base_data / "spins_115" / "dmrg" / "chi_max_320_trunc_1e-06"
    n115_timestamp = 1773150437
    n115_jz_range = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]
    
    # Load data for both system sizes (random_flip only)
    logging.info("Loading n=57 data (random_flip)...")
    n57_diff, n57_energies, n57_dmrg_ground, n57_dmrg_excited = load_vopt_stage2_data(
        57, vopt_unified_path, n57_dmrg_dir, n57_timestamp, n57_jz_range, postprocessing='random_flip'
    )

    logging.info("Loading n=115 data (random_flip)...")
    n115_diff, n115_energies, n115_dmrg_ground, n115_dmrg_excited = load_vopt_stage2_data(
        115, vopt_unified_path, n115_dmrg_dir, n115_timestamp, n115_jz_range, postprocessing='random_flip'
    )
    
    # Create output directory
    save_dir = ensure_dir(Path("results/combined_n57_n115"))
    
    # Define jz ranges for different plots
    jz_ranges = [
        ("all", None, None, "All $J_z$"),
        ("low", 1.1, 1.4, "$J_z$ = 1.1 to 1.4"),
        ("high", 1.6, 6.0, "$J_z$ = 1.6 to 6.0"),
    ]
    
    for range_name, jz_min, jz_max, title_suffix in jz_ranges:
        # Filter data for this range
        if jz_min is not None and jz_max is not None:
            n57_filtered = {k: v for k, v in n57_diff.items() if jz_min <= k <= jz_max}
            n115_filtered = {k: v for k, v in n115_diff.items() if jz_min <= k <= jz_max}
        else:
            n57_filtered = n57_diff
            n115_filtered = n115_diff
        
        if not n57_filtered and not n115_filtered:
            logging.warning(f"No data for range {range_name}")
            continue
        
        # Plot 1: Energy difference normalized by gap
        # Adjusted figure size for subfigure use (3.5 inches width, 2.8 inches height is typical)
        fig, ax = plt.subplots(figsize=(2.6, 3.5))
        
        # Plot n=57 random_flip data with green color
        if n57_filtered:
            xs, ys = sorted_xy(n57_filtered)
            ax.plot(xs, ys, 'v-', label=r'$n=57$', markersize=4, linewidth=1.2,
                   color='#2ca02c', markerfacecolor='#2ca02c', alpha=0.8)
        
        # Plot n=115 random_flip data with purple color
        if n115_filtered:
            xs, ys = sorted_xy(n115_filtered)
            ax.plot(xs, ys, 's-', label=r'$n=115$', markersize=4, linewidth=1.2,
                   color='#9467bd', markerfacecolor='#9467bd', alpha=0.8)
        
        # Add reference line at 1
        all_jzs = list(n57_filtered.keys()) + list(n115_filtered.keys())
        if all_jzs:
            ax.hlines(1, min(all_jzs) - 0.05, max(all_jzs) + 0.05,
                     linestyle='--', color='gray', alpha=0.6, linewidth=1.0, zorder=0)
        
        ax.set_yscale('log')
        ax.set_xlabel(r'$J_z$')
        ax.set_ylabel(r'$\dfrac{|E - E_0|}{E_1 - E_0}$')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        
        # Set x-axis limits for zoomed plots
        if jz_min is not None and jz_max is not None:
            ax.set_xlim(jz_min - 0.05, jz_max + 0.05)
        
        fig.tight_layout(pad=0.3)
        fig.savefig(save_dir / f"energy_diff_vs_gap_n57_n115_{range_name}.pdf",
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(save_dir / f"energy_diff_vs_gap_n57_n115_{range_name}.png",
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        
        logging.info(f"Saved {range_name} energy diff figure: n=57 random_flip ({len(n57_filtered)} pts), n=115 random_flip ({len(n115_filtered)} pts)")
        
        # Plot 2: Upper bounded norm distance (fidelity upper bound)
        # Filter energy data for this range
        if jz_min is not None and jz_max is not None:
            n57_energies_filt = {k: v for k, v in n57_energies.items() if jz_min <= k <= jz_max}
            n115_energies_filt = {k: v for k, v in n115_energies.items() if jz_min <= k <= jz_max}
            n57_dmrg_ground_filt = {k: v for k, v in n57_dmrg_ground.items() if jz_min <= k <= jz_max}
            n115_dmrg_ground_filt = {k: v for k, v in n115_dmrg_ground.items() if jz_min <= k <= jz_max}
            n57_dmrg_excited_filt = {k: v for k, v in n57_dmrg_excited.items() if jz_min <= k <= jz_max}
            n115_dmrg_excited_filt = {k: v for k, v in n115_dmrg_excited.items() if jz_min <= k <= jz_max}
        else:
            n57_energies_filt = n57_energies
            n115_energies_filt = n115_energies
            n57_dmrg_ground_filt = n57_dmrg_ground
            n115_dmrg_ground_filt = n115_dmrg_ground
            n57_dmrg_excited_filt = n57_dmrg_excited
            n115_dmrg_excited_filt = n115_dmrg_excited
        
        fig2, ax2 = plt.subplots(figsize=(3.8, 2.6))
        
        # Calculate upper bounded norm distance: 1 - sqrt(1 - (|E - E_DMRG| / gap)^2)
        # Clip values to [0, 1] to avoid numerical issues
        if n57_filtered:
            xs, ys = sorted_xy(n57_filtered)
            ys = np.clip(np.array(ys, dtype=float), 0.0, 1.0)
            ub = 1.0 - np.sqrt(1.0 - ys**2)
            ax2.plot(xs, ub, 'v-', label=r'$n=57$', markersize=4, linewidth=1.2,
                    color='#2ca02c', markerfacecolor='#2ca02c', alpha=0.8)
        
        if n115_filtered:
            xs, ys = sorted_xy(n115_filtered)
            ys = np.clip(np.array(ys, dtype=float), 0.0, 1.0)
            ub = 1.0 - np.sqrt(1.0 - ys**2)
            ax2.plot(xs, ub, 's-', label=r'$n=115$', markersize=4, linewidth=1.2,
                    color='#9467bd', markerfacecolor='#9467bd', alpha=0.8)
        
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$J_z$')
        ax2.set_ylabel(r'$||\,|\psi\rangle - |\psi_{\mathrm{DMRG}}\rangle\,||^2$')
        ax2.tick_params(which='both', direction='in', top=True, right=True)
        ax2.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        
        if jz_min is not None and jz_max is not None:
            ax2.set_xlim(jz_min - 0.05, jz_max + 0.05)
        
        fig2.tight_layout(pad=0.3)
        fig2.savefig(save_dir / f"upper_bound_norm_distance_n57_n115_{range_name}.pdf",
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig2.savefig(save_dir / f"upper_bound_norm_distance_n57_n115_{range_name}.png",
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig2)
        
        logging.info(f"Saved {range_name} upper bound norm distance figure")
        
        # Plot 3a: Raw energies vs DMRG ground state and excited state for n=57
        fig3a, ax3a = plt.subplots(figsize=(2.6, 3.5))
        
        # Plot n=57 DMRG ground state
        if n57_dmrg_ground_filt:
            xs, ys = sorted_xy(n57_dmrg_ground_filt)
            ax3a.plot(xs, ys, '--', label=r'DMRG $E_0$', linewidth=1.5,
                    color='#2ca02c', alpha=0.6)
        
        # Plot n=57 DMRG excited state
        if n57_dmrg_excited_filt:
            xs, ys = sorted_xy(n57_dmrg_excited_filt)
            ax3a.plot(xs, ys, ':', label=r'DMRG $E_1$', linewidth=1.5,
                    color='#2ca02c', alpha=0.6)
        
        # Plot n=57 SKQD energies
        if n57_energies_filt:
            xs, ys = sorted_xy(n57_energies_filt)
            ax3a.plot(xs, ys, 'v-', label=r'SKQD', markersize=4, linewidth=1.2,
                    color='#2ca02c', markerfacecolor='#2ca02c', alpha=0.8)
        
        ax3a.set_xlabel(r'$J_z$')
        ax3a.set_ylabel(r'$E$ ($n=57$)')
        ax3a.tick_params(which='both', direction='in', top=True, right=True)
        ax3a.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0, fontsize=9)
        ax3a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        
        if jz_min is not None and jz_max is not None:
            ax3a.set_xlim(jz_min - 0.05, jz_max + 0.05)
        
        fig3a.tight_layout(pad=0.3)
        fig3a.savefig(save_dir / f"raw_energies_vs_dmrg_n57_{range_name}.pdf",
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig3a.savefig(save_dir / f"raw_energies_vs_dmrg_n57_{range_name}.png",
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig3a)
        
        logging.info(f"Saved {range_name} raw energies figure for n=57")
        
        # Plot 3b: Raw energies vs DMRG ground state and excited state for n=115
        fig3b, ax3b = plt.subplots(figsize=(2.6, 3.5))
        
        # Plot n=115 DMRG ground state
        if n115_dmrg_ground_filt:
            xs, ys = sorted_xy(n115_dmrg_ground_filt)
            ax3b.plot(xs, ys, '--', label=r'DMRG $E_0$', linewidth=1.5,
                    color='#9467bd', alpha=0.6)
        
        # Plot n=115 DMRG excited state
        if n115_dmrg_excited_filt:
            xs, ys = sorted_xy(n115_dmrg_excited_filt)
            ax3b.plot(xs, ys, ':', label=r'DMRG $E_1$', linewidth=1.5,
                    color='#9467bd', alpha=0.6)
        
        # Plot n=115 SKQD energies
        if n115_energies_filt:
            xs, ys = sorted_xy(n115_energies_filt)
            ax3b.plot(xs, ys, 's-', label=r'SKQD', markersize=4, linewidth=1.2,
                    color='#9467bd', markerfacecolor='#9467bd', alpha=0.8)
        
        ax3b.set_xlabel(r'$J_z$')
        ax3b.set_ylabel(r'$E$ ($n=115$)')
        ax3b.tick_params(which='both', direction='in', top=True, right=True)
        ax3b.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0, fontsize=9)
        ax3b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        
        if jz_min is not None and jz_max is not None:
            ax3b.set_xlim(jz_min - 0.05, jz_max + 0.05)
        
        fig3b.tight_layout(pad=0.3)
        fig3b.savefig(save_dir / f"raw_energies_vs_dmrg_n115_{range_name}.pdf",
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig3b.savefig(save_dir / f"raw_energies_vs_dmrg_n115_{range_name}.png",
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig3b)
        
        logging.info(f"Saved {range_name} raw energies figure for n=115")
    
    logging.info(f"All figures saved to {save_dir}")
    logging.info(f"Total: n=57 random_flip: {len(n57_diff)} data points, n=115 random_flip: {len(n115_diff)} data points")
