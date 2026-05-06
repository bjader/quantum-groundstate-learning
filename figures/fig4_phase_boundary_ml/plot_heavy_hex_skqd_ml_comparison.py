"""
Figure 4 right panel
Generate 3 heavy hex plots comparing SKQD and ML predictions:
1. SKQD heavy hex plot for jz=1.6
2. SKQD heavy hex plot for jz=1.1
3. ML heavy hex plot for jz=1.1

This script now:
- Loads and evaluates ML models directly (no saved predictions needed)
- Caches the final plot data for faster re-plotting
- Follows patterns from test_z_boundary_staggered_mag_error_ml_vs_skqd_combined.py
  and test_combined_observables_boundary.py
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import dill as pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader

import sys
from pathlib import Path as PathLib
# Add parent directories to path for imports
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent))

from qaml.analysis.analysis_utils import filter_to_less_neighbours
from qaml.observables.single_site import szs_from_state_dict
from qaml.visualisation.heavy_hex import plot_corr_heavy_hex, plot_heavy_hex_differences
from qaml.graph.graph_utils import read_edges_txt
from qaml.ml.dataset import HeisenbergDataset
from utils.model_utils import (
    get_most_recent_model_dir,
    load_model_and_settings,
    evaluate_model,
    extract_model_dimensions,
)
from utils.data_utils import resolve_data_root

# Configuration
n = 115
nn = 2
nn_to_plot = 2

# Customizable vmax for difference plot (set to None for auto-scaling from data)
DIFF_VMAX = 0.1  # e.g., 0.3 to set a fixed maximum, or None for auto

# SKQD data directory
dir_skqd = "ts_1_kd_11_shots_100k_ibm_boston_1773150437_1773854302_mixed/recovery_random_flip"
root_dir_skqd = f"../../data/spins_{n}/skqd/{dir_skqd}"

# ML model base directories (will find most recent)
ml_zz_model_base = "../../ml_models/model_boundary2/spins_115/skqd/2_nn_corr_funcs_zz_basis_opt"
ml_z_model_base = "../../ml_models/model_boundary2/spins_115/skqd/z_basis_opt"

# Output and cache directories
fig_dir = "plots/heavy_hex_skqd_ml_comparison/"
cache_dir = "plots/heavy_hex_skqd_ml_comparison/cache/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)
Path(cache_dir).mkdir(parents=True, exist_ok=True)

# Load graph edges for difference plot
edges_path = "../../cmaps/heavy_hex_edges_d_7.txt"
edges, _ = read_edges_txt(edges_path)

# JZ values to plot
jz_values = [1.1]


def get_cache_path(jz: float, obs_type: str) -> str:
    """Get path for cached ML predictions."""
    return os.path.join(cache_dir, f"ml_predictions_{obs_type}_jz_{jz:.1f}.pkl")


def load_cached_predictions(cache_path: str) -> Optional[Any]:
    """Load cached predictions if they exist."""
    if os.path.exists(cache_path):
        print(f"  Loading cached predictions from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def save_cached_predictions(predictions: Any, cache_path: str):
    """Save predictions to cache."""
    with open(cache_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"  Saved predictions to cache: {cache_path}")


def get_boundary_files(jz: float) -> list:
    """Get list of files for a specific Jz value."""
    return [f"XXZ_2d_jz_{jz:.1f}.pkl"]


def load_and_evaluate_model_for_jz(
    model_base: str,
    jz: float,
    device: str = "mps",
) -> Optional[np.ndarray]:
    """
    Load a model and evaluate it for a specific Jz value.
    
    Args:
        model_base: Base path for model directory
        jz: Jz value to evaluate
        device: Device to run model on
        
    Returns:
        Predictions array or None if evaluation fails
    """
    model_dir = get_most_recent_model_dir(model_base)
    
    if not os.path.exists(model_dir):
        print(f"  Warning: Model directory not found: {model_dir}")
        return None
    
    print(f"  Loading model from {model_dir}")
    model, settings = load_model_and_settings(model_dir, device)
    
    data_root = resolve_data_root(settings["dataset_params"]["data_root"], __file__)
    observable_key = settings["dataset_params"]["observable_key"]
    
    dims = extract_model_dimensions(settings, model_dir)
    n_edges = dims["n_edges"]
    
    print(f"    Observable: {observable_key}")
    print(f"    Data root: {data_root}")
    
    # Evaluate on the specific Jz value
    boundary_files = get_boundary_files(jz)
    
    print(f"    Evaluating on Jz={jz}...")
    dataset = HeisenbergDataset(
        root_dir=data_root,
        n_inputs=n_edges,
        observable_key=observable_key,
        jz_min=jz,
        files=boundary_files,
        preload=True,
    )
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions, avg_loss = evaluate_model(model, data_loader, device)
    
    print(f"      Loss: {avg_loss:.6f}")
    
    if jz not in predictions:
        print(f"  Warning: No predictions for Jz={jz}")
        return None
    
    try:
        preds = predictions[jz]["preds"]
    except KeyError:
        preds = predictions[jz]["predictions"]
    
    return preds


def get_ml_predictions(jz: float, device: str = "mps", use_cache: bool = True) -> Dict[str, np.ndarray]:
    """
    Get ML predictions for both Z and ZZ observables at a specific Jz.
    
    Args:
        jz: Jz value
        device: Device to run models on
        use_cache: Whether to use cached predictions
        
    Returns:
        Dictionary with 'z' and 'zz' predictions
    """
    results = {}
    
    # Get Z predictions
    z_cache_path = get_cache_path(jz, "z")
    if use_cache:
        z_preds = load_cached_predictions(z_cache_path)
        if z_preds is not None:
            results["z"] = z_preds
            print(f"  Using cached Z predictions for Jz={jz}")
    
    if "z" not in results:
        print(f"\n  Evaluating Z model for Jz={jz}...")
        z_preds = load_and_evaluate_model_for_jz(ml_z_model_base, jz, device)
        if z_preds is not None:
            results["z"] = z_preds
            save_cached_predictions(z_preds, z_cache_path)
    
    # Get ZZ predictions
    zz_cache_path = get_cache_path(jz, "zz")
    if use_cache:
        zz_preds = load_cached_predictions(zz_cache_path)
        if zz_preds is not None:
            results["zz"] = zz_preds
            print(f"  Using cached ZZ predictions for Jz={jz}")
    
    if "zz" not in results:
        print(f"\n  Evaluating ZZ model for Jz={jz}...")
        zz_preds = load_and_evaluate_model_for_jz(ml_zz_model_base, jz, device)
        if zz_preds is not None:
            results["zz"] = zz_preds
            save_cached_predictions(zz_preds, zz_cache_path)
    
    return results


def main():
    """Main execution function."""
    device = "mps"  # Change to "cuda" or "cpu" if needed
    use_cache = True  # Set to False to force re-evaluation
    
    print("=" * 80)
    print("Generating heavy hex plots for SKQD vs ML comparison")
    print(f"System size: n={n}")
    print(f"Output directory: {fig_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Use cache: {use_cache}")
    print("=" * 80)
    
    # Plot 1 & 2: SKQD plots for jz values
    for jz in jz_values:
        print(f"\nProcessing SKQD Jz={jz}")
        
        try:
            with open(os.path.join(root_dir_skqd, f"XXZ_2d_jz_{jz:.1f}.pkl"), "rb") as f:
                data_skqd = pickle.load(f)
            
            corr_skqd = data_skqd[f"{nn:d}_nn_corr_funcs_zz_basis_opt"]
            pairs_skqd = data_skqd[f"{nn:d}_nn_corr_pairs_zz_basis_opt"]
            
            try:
                zs_skqd = data_skqd["z_basis_opt"]
            except KeyError:
                zs_skqd = szs_from_state_dict(data_skqd["ground_state_dict"])
                zs_skqd = np.array(zs_skqd[::-1]) * 2
            
            # Filter to desired nearest neighbors
            pairs_filtered, corr_filtered = filter_to_less_neighbours(
                pairs_skqd, corr_skqd, nn_to_plot
            )
            
            # Plot
            plot = plot_corr_heavy_hex(pairs_filtered, n, corr_filtered, zs_skqd, plt.cm.seismic)
            plot.savefig(fig_dir + f"skqd_jz_{jz:.1f}.pdf")
            plot.close()
            print(f"  Saved: {fig_dir}skqd_jz_{jz:.1f}.pdf")
            
        except Exception as e:
            print(f"  Error loading SKQD data for Jz={jz}: {e}")
    
    # Plot 3: ML plot for jz=1.1
    jz_target = 1.1
    print(f"\n{'=' * 80}")
    print(f"Processing ML predictions for Jz={jz_target}")
    print(f"{'=' * 80}")
    
    try:
        # Get ML predictions (with caching)
        ml_predictions = get_ml_predictions(jz_target, device, use_cache)
        
        if "z" not in ml_predictions or "zz" not in ml_predictions:
            raise ValueError(f"Failed to get ML predictions for Jz={jz_target}")
        
        z_ml = ml_predictions["z"]
        zz_ml = ml_predictions["zz"]
        
        # Load SKQD data to get pairs and single-site magnetizations
        with open(os.path.join(root_dir_skqd, f"XXZ_2d_jz_{jz_target:.1f}.pkl"), "rb") as f:
            data_skqd = pickle.load(f)
        
        pairs_skqd = data_skqd[f"{nn:d}_nn_corr_pairs_zz_basis_opt"]
        
        try:
            zs_skqd = data_skqd["z_basis_opt"]
        except KeyError:
            zs_skqd = szs_from_state_dict(data_skqd["ground_state_dict"])
            zs_skqd = np.array(zs_skqd[::-1]) * 2
        
        # Filter to desired nearest neighbors for ML
        pairs_filtered_ml, zz_ml_filtered = filter_to_less_neighbours(
            pairs_skqd, zz_ml, nn_to_plot
        )
        
        # Plot ML with ML's predicted Z values
        print(f"\n  Creating ML heavy hex plot...")
        plot = plot_corr_heavy_hex(pairs_filtered_ml, n, zz_ml_filtered, z_ml, plt.cm.seismic)
        plot.savefig(fig_dir + f"ml_jz_{jz_target:.1f}.pdf")
        plot.close()
        print(f"  Saved: {fig_dir}ml_jz_{jz_target:.1f}.pdf")
        
        # Plot 4: Absolute difference between SKQD and ML for jz=1.1
        print(f"\n  Processing |SKQD - ML| difference for Jz={jz_target}")
        
        # Get SKQD correlations for jz=1.1
        zz_skqd = np.array(data_skqd[f"{nn:d}_nn_corr_funcs_zz_basis_opt"])
        
        # Compute absolute differences
        print(f"    Max Z (ML): {np.max(z_ml):.4f}, Max Z (SKQD): {np.max(zs_skqd):.4f}")
        z_abs_diff = np.abs(np.array(z_ml) - np.array(zs_skqd))
        zz_abs_diff = np.abs(np.array(zz_ml) - zz_skqd)
    
        # Determine color limits from data or use custom vmax
        if DIFF_VMAX is None:
            z_max = np.nanmax(z_abs_diff)
            zz_max = np.nanmax(zz_abs_diff)
            combined_max = max(z_max, zz_max)
        else:
            combined_max = DIFF_VMAX
    
        # Use the new plot_heavy_hex_differences function with Reds colormap
        # and auto-scaled limits (vmin=0, vmax=combined_max)
        fig = plot_heavy_hex_differences(
            edges=pairs_skqd,  # Use correlation pairs as edges
            n=n,
            z_diff=z_abs_diff,
            zz_diff=zz_abs_diff,
            corr_pairs=pairs_skqd,
            title=f"|SKQD - ML| (Jz={jz_target:.1f})",
            cmap=plt.cm.Reds,
            vmin_z=0,
            vmax_z=combined_max,
            vmin_zz=0,
            vmax_zz=combined_max
        )
        fig.savefig(fig_dir + f"abs_diff_skqd_ml_jz_{jz_target:.1f}.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_dir}abs_diff_skqd_ml_jz_{jz_target:.1f}.pdf")
    
        # Save a separate compact colorbar for figure composition
        fig_cbar, ax_cbar = plt.subplots(figsize=(0.5, 3))  # Narrower width
        norm = mpl.colors.Normalize(vmin=0, vmax=combined_max)
        sm = mpl.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax_cbar, ticks=[0, 0.05, 0.1])
        # cbar.set_label(r'$|\Delta\langle ZZ \rangle|$', fontsize=20, labelpad=10) # No label on this
        cbar.ax.tick_params(labelsize=18)
        # Move ticks to the left side
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        fig_cbar.savefig(fig_dir + f"colorbar_abs_diff_jz_{jz_target:.1f}.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig_cbar)
        print(f"  Saved colorbar: {fig_dir}colorbar_abs_diff_jz_{jz_target:.1f}.pdf")
    
        print(f"\n  Statistics:")
        print(f"    Mean absolute difference (Z): {np.mean(z_abs_diff):.4f}")
        print(f"    Max absolute difference (Z): {np.max(z_abs_diff):.4f}")
        print(f"    Mean absolute difference (ZZ): {np.mean(zz_abs_diff):.4f}")
        print(f"    Max absolute difference (ZZ): {np.max(zz_abs_diff):.4f}")
        
    except Exception as e:
        print(f"  Error processing ML data for Jz={jz_target}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print(f"Plotting complete! Saved to {fig_dir}")
    print(f"Cached predictions saved to {cache_dir}")
    print(f"To force re-evaluation, set use_cache=False in main()")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
