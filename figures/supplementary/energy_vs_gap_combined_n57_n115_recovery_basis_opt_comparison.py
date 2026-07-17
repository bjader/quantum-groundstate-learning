"""
Makes supplementary fig energy_super_comparison

Combines:
1. Before basis optimization (post-select and random flip) for n=57 and n=115
2. After basis optimization (post-select and random flip) for n=57 and n=115
Total: 8 lines on the same plot

Legend is saved separately as a wide horizontal figure for manual placement in Keynote.

Marker scheme:
- Before optimization: circles (o) and diamonds (D)
- After optimization: triangles (v) and squares (s)
- Post-select: hollow markers
- Random flip: filled markers
- n=57: green color
- n=115: purple color
"""

import logging
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Reduced DMRG/SKQD data are plain dicts, so no qaml unpickling shims are needed.

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
    "legend.fontsize": 8,
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

def load_qpu_run_before_opt(qpu_path, dmrg_dir, dmrg_es, dmrg_e1s):
    """Load QPU recovery data before basis optimization and compute energy differences relative to gap."""
    qpu_path = Path(qpu_path)
    dmrg_dir = Path(dmrg_dir)
    es, diff = {}, {}
    
    for p in qpu_path.glob("XXZ_2d_jz_*.pkl"):
        jz = infer_jz_from_fname(p)
        if jz is None:
            continue
        
        # Try to get DMRG data from existing dicts or load it
        if jz in dmrg_es and jz in dmrg_e1s:
            dmrg_e = float(dmrg_es[jz])
            e1 = float(dmrg_e1s[jz])
        else:
            # Try to load DMRG data for this Jz value
            dmrg_file = dmrg_dir / f"XXZ_2d_jz_{jz:.1f}.pkl"
            if not dmrg_file.exists():
                logging.warning(f"Missing DMRG data for Jz={jz:.1f} (QPU file: {p.name})")
                continue
            try:
                dmrg_res = load_pickle(dmrg_file)
                dmrg_e = float(dmrg_res["ground_state_energy"])
                e1 = float(dmrg_res["e1"])
            except Exception as ex:
                logging.warning(f"Failed to load DMRG for Jz={jz:.1f}: {ex}")
                continue
        
        gap = e1 - dmrg_e
        if gap == 0:
            continue
            
        try:
            data = load_pickle(p)
        except Exception:
            continue
        if "ground_state_energy" not in data:
            continue
        e = float(data["ground_state_energy"])
        es[jz] = e
        diff[jz] = abs((e - dmrg_e) / gap)
    
    return es, diff

def load_vopt_stage2_data(n_spins, skqd_dir, dmrg_dir, jz_range, postprocessing='post_select'):
    """
    Load the basis-optimised (after-vopt) energy directly from the reduced SKQD data.

    The after-vopt energy is read from the ``final_global_energy`` key of each reduced
    SKQD file (script-faithful; identical to the top-level ``final_global_energy`` of the
    original stage2 basis-optimisation result). DMRG ground/excited energies come from
    the reduced DMRG files as plain dict keys.

    Args:
        n_spins: Number of spins (57 or 115)
        skqd_dir: Path to reduced SKQD data (recovery_{postprocessing})
        dmrg_dir: Path to reduced DMRG reference data
        jz_range: Range of Jz values to process
        postprocessing: Postprocessing method ('post_select' or 'random_flip')

    Returns:
        diff_to_gap: Dictionary mapping Jz to |E - E_DMRG| / gap
    """
    diff_to_gap = {}

    if not skqd_dir.exists():
        logging.error(f"SKQD directory not found: {skqd_dir}")
        return diff_to_gap

    # After-vopt energy key stored in the reduced SKQD files
    energy_key = "final_global_energy"

    for jz in jz_range:
        # Load DMRG reference (reduced format: plain dict keys)
        dmrg_file = dmrg_dir / f"XXZ_2d_jz_{jz:.1f}.pkl"
        if not dmrg_file.exists():
            logging.warning(f"Missing DMRG data for n={n_spins}, Jz={jz:.1f}")
            continue

        try:
            dmrg_res = load_pickle(dmrg_file)
            dmrg_e = float(dmrg_res["ground_state_energy"])
            e1 = float(dmrg_res["e1"])
        except Exception as ex:
            logging.warning(f"Failed to load DMRG for n={n_spins}, Jz={jz:.1f}: {ex}")
            continue

        gap = e1 - dmrg_e
        if gap == 0:
            continue

        # After-vopt energy comes from the reduced SKQD file for this (n, jz, postprocessing)
        skqd_file = skqd_dir / f"XXZ_2d_jz_{jz:.1f}.pkl"
        if not skqd_file.exists():
            logging.warning(f"Missing SKQD data for n={n_spins}, Jz={jz:.1f}, postprocessing={postprocessing}")
            continue

        try:
            resq = load_pickle(skqd_file)
            if energy_key not in resq:
                logging.warning(f"Missing key '{energy_key}' for n={n_spins}, Jz={jz:.1f}, postprocessing={postprocessing}")
                continue
            e_final = float(resq[energy_key])
            diff_to_gap[jz] = abs((e_final - dmrg_e) / gap)
        except Exception as ex:
            logging.warning(f"Failed to load SKQD results for n={n_spins}, Jz={jz:.1f}, postprocessing={postprocessing}: {ex}")
            continue
    
    return diff_to_gap

if __name__ == "__main__":
    
    # Configuration
    base_data = Path("../../data")
    cache_file = Path("results/super_comparison_n57_n115/cached_data.pkl")
    use_cache = False  # After-vopt energy now comes from the reduced SKQD files
    
    # NOTE: Only the reduced chi=320 DMRG is available locally, so the before-opt gap
    # (originally computed against chi=80 DMRG) now also uses the reduced chi=320 DMRG.
    # n=57 configuration (reduced SKQD + reduced DMRG)
    n57_dmrg_dir = base_data / "spins_57" / "dmrg" / "chi_max_320_trunc_1e-06_reduced"
    n57_dmrg_dir_chi80 = n57_dmrg_dir
    n57_jz_range = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]
    n57_post_select_path = base_data / "spins_57" / "skqd" / "recovery_post_select"
    n57_random_flip_path = base_data / "spins_57" / "skqd" / "recovery_random_flip"

    # n=115 configuration (reduced SKQD + reduced DMRG)
    n115_dmrg_dir = base_data / "spins_115" / "dmrg" / "chi_max_320_trunc_1e-06_reduced"
    n115_dmrg_dir_chi80 = n115_dmrg_dir
    n115_jz_range = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]
    n115_post_select_path = base_data / "spins_115" / "skqd" / "recovery_post_select"
    n115_random_flip_path = base_data / "spins_115" / "skqd" / "recovery_random_flip"
    
    # Try to load from cache first
    # Note: All data variables will be defined in either the cache or loading path
    if use_cache and cache_file.exists():
        logging.info(f"Loading data from cache: {cache_file}")
        try:
            cached_data = load_pickle(cache_file)
            n57_before_post_select = cached_data['n57_before_post_select']
            n57_before_random_flip = cached_data['n57_before_random_flip']
            n57_after_post_select = cached_data['n57_after_post_select']
            n57_after_random_flip = cached_data['n57_after_random_flip']
            n115_before_post_select = cached_data['n115_before_post_select']
            n115_before_random_flip = cached_data['n115_before_random_flip']
            n115_after_post_select = cached_data['n115_after_post_select']
            n115_after_random_flip = cached_data['n115_after_random_flip']
            logging.info("Successfully loaded data from cache")
        except Exception as ex:
            logging.warning(f"Failed to load cache: {ex}. Loading from source...")
            use_cache = False
    
    if not use_cache or not cache_file.exists():
        # Load DMRG data for both system sizes
        logging.info("Loading DMRG data for n=57...")
        n57_dmrg_es, n57_dmrg_e1s = {}, {}
        for jz in n57_jz_range:
            try:
                dmrg_res = load_pickle(n57_dmrg_dir_chi80 / f"XXZ_2d_jz_{jz:.1f}.pkl")
                n57_dmrg_es[jz] = float(dmrg_res["ground_state_energy"])
                n57_dmrg_e1s[jz] = float(dmrg_res["e1"])
            except Exception:
                logging.warning(f"Missing/corrupt DMRG for n=57, Jz={jz:.1f}")
        
        logging.info("Loading DMRG data for n=115...")
        n115_dmrg_es, n115_dmrg_e1s = {}, {}
        for jz in n115_jz_range:
            try:
                dmrg_res = load_pickle(n115_dmrg_dir_chi80 / f"XXZ_2d_jz_{jz:.1f}.pkl")
                n115_dmrg_es[jz] = float(dmrg_res["ground_state_energy"])
                n115_dmrg_e1s[jz] = float(dmrg_res["e1"])
            except Exception:
                logging.warning(f"Missing/corrupt DMRG for n=115, Jz={jz:.1f}")
        
        # Load BEFORE basis optimization data (from recovery directories)
        logging.info("Loading n=57 BEFORE optimization data (post-select)...")
        _, n57_before_post_select = load_qpu_run_before_opt(n57_post_select_path, n57_dmrg_dir_chi80, n57_dmrg_es, n57_dmrg_e1s)
        
        logging.info("Loading n=57 BEFORE optimization data (random flip)...")
        _, n57_before_random_flip = load_qpu_run_before_opt(n57_random_flip_path, n57_dmrg_dir_chi80, n57_dmrg_es, n57_dmrg_e1s)
        
        logging.info("Loading n=115 BEFORE optimization data (post-select)...")
        _, n115_before_post_select = load_qpu_run_before_opt(n115_post_select_path, n115_dmrg_dir_chi80, n115_dmrg_es, n115_dmrg_e1s)
        
        logging.info("Loading n=115 BEFORE optimization data (random flip)...")
        _, n115_before_random_flip = load_qpu_run_before_opt(n115_random_flip_path, n115_dmrg_dir_chi80, n115_dmrg_es, n115_dmrg_e1s)
        
        # Load AFTER basis optimization data (final_global_energy from reduced SKQD files)
        logging.info("Loading n=57 AFTER optimization data (post-select)...")
        n57_after_post_select = load_vopt_stage2_data(57, n57_post_select_path, n57_dmrg_dir, n57_jz_range, postprocessing='post_select')

        logging.info("Loading n=57 AFTER optimization data (random flip)...")
        n57_after_random_flip = load_vopt_stage2_data(57, n57_random_flip_path, n57_dmrg_dir, n57_jz_range, postprocessing='random_flip')

        logging.info("Loading n=115 AFTER optimization data (post-select)...")
        n115_after_post_select = load_vopt_stage2_data(115, n115_post_select_path, n115_dmrg_dir, n115_jz_range, postprocessing='post_select')

        logging.info("Loading n=115 AFTER optimization data (random flip)...")
        n115_after_random_flip = load_vopt_stage2_data(115, n115_random_flip_path, n115_dmrg_dir, n115_jz_range, postprocessing='random_flip')
        
        # Save to cache
        logging.info(f"Saving data to cache: {cache_file}")
        ensure_dir(cache_file.parent)
        cached_data = {
            'n57_before_post_select': n57_before_post_select,
            'n57_before_random_flip': n57_before_random_flip,
            'n57_after_post_select': n57_after_post_select,
            'n57_after_random_flip': n57_after_random_flip,
            'n115_before_post_select': n115_before_post_select,
            'n115_before_random_flip': n115_before_random_flip,
            'n115_after_post_select': n115_after_post_select,
            'n115_after_random_flip': n115_after_random_flip,
        }
        with cache_file.open('wb') as f:
            pickle.dump(cached_data, f)
        logging.info("Data cached successfully")
    
    # Create output directory
    save_dir = ensure_dir(Path("results/super_comparison_n57_n115"))
    
    # Define colors for n=57 (green) and n=115 (purple)
    color_57 = '#2ca02c'
    color_115 = '#9467bd'
    
    # Define jz ranges for different plots
    jz_ranges = [
        ("all", None, None, "All $J_z$"),
        ("low", 1.1, 1.4, "$J_z$ = 1.1 to 1.4"),
        ("high", 1.6, 6.0, "$J_z$ = 1.6 to 6.0"),
    ]
    
    for range_name, jz_min, jz_max, title_suffix in jz_ranges:
        # Filter data for this range
        if jz_min is not None and jz_max is not None:
            filter_fn = lambda d: {k: v for k, v in d.items() if jz_min <= k <= jz_max}
        else:
            filter_fn = lambda d: d
        
        n57_before_post_filt = filter_fn(n57_before_post_select)
        n57_before_rand_filt = filter_fn(n57_before_random_flip)
        n57_after_post_filt = filter_fn(n57_after_post_select)
        n57_after_rand_filt = filter_fn(n57_after_random_flip)
        
        n115_before_post_filt = filter_fn(n115_before_post_select)
        n115_before_rand_filt = filter_fn(n115_before_random_flip)
        n115_after_post_filt = filter_fn(n115_after_post_select)
        n115_after_rand_filt = filter_fn(n115_after_random_flip)
        
        # Check if we have any data
        all_data = [n57_before_post_filt, n57_before_rand_filt, n57_after_post_filt, n57_after_rand_filt,
                    n115_before_post_filt, n115_before_rand_filt, n115_after_post_filt, n115_after_rand_filt]
        if not any(all_data):
            logging.warning(f"No data for range {range_name}")
            continue
        
        # Plot: Energy difference normalized by gap - Super comparison with 8 lines
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        
        # Define colors for n=57 (green) and n=115 (purple)
        color_57 = '#2ca02c'
        color_115 = '#9467bd'
        
        # Marker scheme:
        # Before: circles (o) and diamonds (D)
        # After: triangles (v) and squares (s)
        # Post-select: hollow (white fill)
        # Random flip: filled (color fill)
        
        # Plot in order: before lines first, then n=115 after, then n=57 after (on top)
        
        # Plot n=57 BEFORE data (2 lines)
        if n57_before_post_filt:
            xs, ys = sorted_xy(n57_before_post_filt)
            ax.plot(xs, ys, 'o-', markersize=3.5, linewidth=1.0,
                   color=color_57, markerfacecolor='white', markeredgewidth=0.8, alpha=0.7)
        
        if n57_before_rand_filt:
            xs, ys = sorted_xy(n57_before_rand_filt)
            ax.plot(xs, ys, 'o-', markersize=3.5, linewidth=1.0,
                   color=color_57, markerfacecolor=color_57, alpha=0.7)
        
        # Plot n=115 BEFORE data (2 lines)
        if n115_before_post_filt:
            xs, ys = sorted_xy(n115_before_post_filt)
            ax.plot(xs, ys, 'D-', markersize=3, linewidth=1.0,
                   color=color_115, markerfacecolor='white', markeredgewidth=0.8, alpha=0.7)
        
        if n115_before_rand_filt:
            xs, ys = sorted_xy(n115_before_rand_filt)
            ax.plot(xs, ys, 'D-', markersize=3, linewidth=1.0,
                   color=color_115, markerfacecolor=color_115, alpha=0.7)
        
        # Plot n=115 AFTER data (2 lines)
        if n115_after_post_filt:
            xs, ys = sorted_xy(n115_after_post_filt)
            ax.plot(xs, ys, 's-', markersize=3.5, linewidth=1.2,
                   color=color_115, markerfacecolor='white', markeredgewidth=0.8, alpha=0.9)
        
        if n115_after_rand_filt:
            xs, ys = sorted_xy(n115_after_rand_filt)
            ax.plot(xs, ys, 's-', markersize=3.5, linewidth=1.2,
                   color=color_115, markerfacecolor=color_115, alpha=0.9)
        
        # Plot n=57 AFTER data (2 lines) - plotted last so they're on top
        if n57_after_post_filt:
            xs, ys = sorted_xy(n57_after_post_filt)
            ax.plot(xs, ys, 'v-', markersize=4, linewidth=1.2,
                   color=color_57, markerfacecolor='white', markeredgewidth=0.8, alpha=0.9)
        
        if n57_after_rand_filt:
            xs, ys = sorted_xy(n57_after_rand_filt)
            ax.plot(xs, ys, 'v-', markersize=4, linewidth=1.2,
                   color=color_57, markerfacecolor=color_57, alpha=0.9)
        
        # Add reference line at 1
        all_jzs = []
        for d in all_data:
            all_jzs.extend(d.keys())
        if all_jzs:
            ax.hlines(1, min(all_jzs) - 0.05, max(all_jzs) + 0.05,
                     linestyle='--', color='gray', alpha=0.6, linewidth=1.0, zorder=0)
        
        ax.set_yscale('log')
        ax.set_xlabel(r'$J_z$')
        ax.set_ylabel(r'$\dfrac{|E - E_0|}{E_1 - E_0}$')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        # No legend on the main plot
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        
        # Set x-axis limits for zoomed plots
        if jz_min is not None and jz_max is not None:
            ax.set_xlim(jz_min - 0.05, jz_max + 0.05)
        
        fig.tight_layout(pad=0.3)
        fig.savefig(save_dir / f"energy_diff_vs_gap_super_comparison_{range_name}.pdf",
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(save_dir / f"energy_diff_vs_gap_super_comparison_{range_name}.png",
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        
        logging.info(f"Saved {range_name} figure with 8 lines (no legend)")
    
    # Create separate legend figure with vertical layout (2 columns, 4 rows)
    logging.info("Creating separate legend figure...")
    fig_legend, ax_legend = plt.subplots(figsize=(6, 4))
    ax_legend.axis('off')
    
    # Create dummy plots for legend entries - explicit labels for all 8 lines
    # Ordered so n=57 entries are on top (first 4 rows), n=115 below (last 4 rows)
    legend_elements = [
        Line2D([0], [0], marker='o', color=color_57, linewidth=1.5, markersize=6,
                   markerfacecolor='white', markeredgewidth=1.5, label=r'$n=57$ post-sel.'),
        Line2D([0], [0], marker='o', color=color_57, linewidth=1.5, markersize=6,
                   markerfacecolor=color_57, label=r'$n=57$ rand. flip'),
        Line2D([0], [0], marker='v', color=color_57, linewidth=1.5, markersize=7,
                   markerfacecolor='white', markeredgewidth=1.5, label=r'$n=57$ post-sel. (basis opt.)'),
        Line2D([0], [0], marker='v', color=color_57, linewidth=1.5, markersize=7,
                   markerfacecolor=color_57, label=r'$n=57$ rand. flip (basis opt.)'),
        Line2D([0], [0], marker='D', color=color_115, linewidth=1.5, markersize=5.5,
                   markerfacecolor='white', markeredgewidth=1.5, label=r'$n=115$ post-sel.'),
        Line2D([0], [0], marker='D', color=color_115, linewidth=1.5, markersize=5.5,
                   markerfacecolor=color_115, label=r'$n=115$ rand. flip'),
        Line2D([0], [0], marker='s', color=color_115, linewidth=1.5, markersize=6.5,
                   markerfacecolor='white', markeredgewidth=1.5, label=r'$n=115$ post-sel. (basis opt.)'),
        Line2D([0], [0], marker='s', color=color_115, linewidth=1.5, markersize=6.5,
                   markerfacecolor=color_115, label=r'$n=115$ rand. flip (basis opt.)'),
    ]
    
    # Create legend with all 8 entries in 4 rows (2 columns per row)
    ax_legend.legend(handles=legend_elements, loc='center', ncol=1, frameon=True,
                    fancybox=False, edgecolor='black', framealpha=1.0, fontsize=10)
    
    fig_legend.tight_layout(pad=0.1)
    fig_legend.savefig(save_dir / "legend_separate.pdf", dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig_legend.savefig(save_dir / "legend_separate.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_legend)
    
    logging.info(f"Saved separate legend to {save_dir / 'legend_separate.pdf'}")
    
    logging.info(f"All figures saved to {save_dir}")
    logging.info("\nData point counts:")
    logging.info(f"n=57 before post-select: {len(n57_before_post_select)}")
    logging.info(f"n=57 before random flip: {len(n57_before_random_flip)}")
    logging.info(f"n=57 after post-select: {len(n57_after_post_select)}")
    logging.info(f"n=57 after random flip: {len(n57_after_random_flip)}")
    logging.info(f"n=115 before post-select: {len(n115_before_post_select)}")
    logging.info(f"n=115 before random flip: {len(n115_before_random_flip)}")
    logging.info(f"n=115 after post-select: {len(n115_after_post_select)}")
    logging.info(f"n=115 after random flip: {len(n115_after_random_flip)}")
    
    # Calculate improvement factors for n=57 after basis optimization
    logging.info("\n" + "=" * 60)
    logging.info("IMPROVEMENT ANALYSIS: n=57 after basis optimization")
    logging.info("=" * 60)
    
    # Find common Jz values between post-select and random flip
    common_jz_57_after = set(n57_after_post_select.keys()) & set(n57_after_random_flip.keys())
    
    if common_jz_57_after:
        ratios = []
        for jz in sorted(common_jz_57_after):
            post_val = n57_after_post_select[jz]
            rand_val = n57_after_random_flip[jz]
            ratio = post_val / rand_val  # How many times worse post-select is
            ratios.append(ratio)
            logging.info(f"Jz={jz:.1f}: post-select/random-flip = {ratio:.3f} (random flip is {1/ratio:.3f}x better)")
        
        ratios = np.array(ratios)
        logging.info(f"\nSummary statistics (post-select / random-flip ratio):")
        logging.info(f"  Median: {np.median(ratios):.3f}")
        logging.info(f"  Mean: {np.mean(ratios):.3f}")
        logging.info(f"  Min: {np.min(ratios):.3f}")
        logging.info(f"  Max: {np.max(ratios):.3f}")
        logging.info(f"\nMedian improvement factor (random flip vs post-select): {np.median(ratios):.3f}x")
        logging.info(f"Random flip achieves {np.median(ratios):.2f}x lower energy error (median)")
    else:
        logging.warning("No common Jz values found between n=57 post-select and random flip after optimization")
    
    # Also calculate for n=115 for comparison
    logging.info("\n" + "=" * 60)
    logging.info("IMPROVEMENT ANALYSIS: n=115 after basis optimization")
    logging.info("=" * 60)
    
    common_jz_115_after = set(n115_after_post_select.keys()) & set(n115_after_random_flip.keys())
    
    if common_jz_115_after:
        ratios_115 = []
        for jz in sorted(common_jz_115_after):
            post_val = n115_after_post_select[jz]
            rand_val = n115_after_random_flip[jz]
            ratio = post_val / rand_val
            ratios_115.append(ratio)
        
        ratios_115 = np.array(ratios_115)
        logging.info(f"Summary statistics (post-select / random-flip ratio):")
        logging.info(f"  Median: {np.median(ratios_115):.3f}")
        logging.info(f"  Mean: {np.mean(ratios_115):.3f}")
        logging.info(f"  Min: {np.min(ratios_115):.3f}")
        logging.info(f"  Max: {np.max(ratios_115):.3f}")
        logging.info(f"\nMedian improvement factor (random flip vs post-select): {np.median(ratios_115):.3f}x")
        logging.info(f"Random flip achieves {np.median(ratios_115):.2f}x lower energy error (median)")
    else:
        logging.warning("No common Jz values found between n=115 post-select and random flip after optimization")
