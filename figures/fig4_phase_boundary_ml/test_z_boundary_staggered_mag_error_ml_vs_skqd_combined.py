"""
Figure 4 top panel - Staggered Magnetization Error Analysis

WHAT THIS SCRIPT COMPUTES:
--------------------------
For each system size (n=57, n=115):
1. Loads trained ML model for Z observable
2. Evaluates model on boundary (Jz=1.1-1.4), train, and val splits
3. Gets ML predictions for single-site Z expectation values
4. Computes staggered magnetization: m_s = mean(sign(Z_i) * Z_i)
5. Compares ML staggered mag vs SKQD staggered mag
6. Calculates percentage error: |m_s_ML - m_s_SKQD| / |m_s_SKQD| * 100

WHAT IS CACHED:
--------------
Cache location: experiments/2d_xxz/ml/plots/combined_ml_test_boundary/cache/
Cache files: staggered_mag_results_n{57|115}_model_boundary2.pkl

Each cache file contains a dictionary:
{
    "jz": [list of Jz values],
    "error": [list of percentage errors for each Jz],
    "regime": [list of "boundary" or "training" labels for each Jz]
}

OUTPUT:
-------
Plot: experiments/2d_xxz/ml/plots/combined_ml_test_boundary/
      staggered_mag_error_ml_vs_skqd_boundary_combined_n57_n115.pdf

Shows percentage error vs Jz for both system sizes with:
- Boundary test points (Jz < 1.6) emphasized
- Training points (Jz >= 1.6) de-emphasized
- Shaded band for training regime
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

import sys
from pathlib import Path
# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qaml.ml.dataset import HeisenbergDataset
from utils.data_utils import load_skqd_data, resolve_data_root
from utils.model_utils import (
    get_most_recent_model_dir,
    load_model_and_settings,
    evaluate_model,
    extract_model_dimensions,
)

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

# System size configurations
SYSTEM_CONFIGS = {
    57: {
        "marker": "v",  # Triangle (down-pointing)
        "label": r"$n=57$",
        "color": "C0",
        "model_base": "../../ml_models/model_boundary2/spins_57/skqd/z_basis_opt",
    },
    115: {
        "marker": "s",  # Square
        "label": r"$n=115$",
        "color": "C1",
        "model_base": "../../ml_models/model_boundary2/spins_115/skqd/z_basis_opt",
    },
}


def get_model_type_from_base(model_base: str) -> str:
    """
    Extract model type from model_base path.
    Returns the model folder type to differentiate cache.
    Extracts the folder name after 'ml_models' in the path.
    """
    # Split path by slashes and find ml_models
    parts = model_base.split('/')

    # Find the index of 'ml_models' in the path
    try:
        ml_models_idx = parts.index('ml_models')
        # The next part after ml_models is the model type
        if ml_models_idx + 1 < len(parts):
            return parts[ml_models_idx + 1]
    except ValueError:
        pass

    return "unknown"

def detect_model_folder(config: Dict) -> str:
    """
    Detect which model folder is being used from SYSTEM_CONFIGS.
    
    Returns:
        Model folder type (e.g., 'model_boundary', 'model_boundary2')
    """
    # Check the first model path to determine folder
    for key, path in config.items():
        if key == "model_base":
            return get_model_type_from_base(path)
    # Default to model_boundary if not found
    return "model_boundary"

def get_cache_path(n_spins: int, model_type: str, cache_dir: str) -> str:
    """
    Get path for cached staggered magnetization results.
    
    Cache contains: {"jz": [...], "error": [...], "regime": [...]}
    """
    return os.path.join(cache_dir, f"staggered_mag_results_n{n_spins}_{model_type}.pkl")

def load_cached_results(cache_path: str) -> Optional[Any]:
    """Load cached results if they exist."""
    if os.path.exists(cache_path):
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def save_cached_results(results: Dict, cache_path: str):
    """
    Save staggered magnetization results to cache.
    
    Args:
        results: Dict with keys "jz", "error", "regime"
        cache_path: Path to save cache file
    """
    Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved results to cache: {cache_path}")

def compute_staggered_magnetization(z_values: np.ndarray) -> float:
    """
    Compute staggered magnetization from single-site Z expectation values.
    
    Uses the sign of Z values to determine sublattice structure:
    staggered_mag = mean(sign(Z_i) * Z_i)
    
    This automatically identifies the two sublattices based on the sign
    of the magnetization at each site.
    
    Args:
        z_values: Array of Z expectation values for each site
        
    Returns:
        Staggered magnetization value
    """
    # Determine sublattice pattern from sign of Z values
    eta = np.sign(z_values)
    # Handle zero values (treat as positive sublattice)
    eta = np.where(eta == 0, 1.0, eta)
    
    # Compute staggered magnetization
    staggered_mag = np.mean(eta * z_values)
    
    return float(staggered_mag)

def load_staggered_mag_from_predictions(
    predictions: Dict, jz: float
) -> Optional[float]:
    """
    Load ML predictions and compute staggered magnetization.
    
    Args:
        predictions: Dictionary of predictions by Jz value
        jz: Jz value to load
        
    Returns:
        Staggered magnetization value or None
    """
    if jz not in predictions:
        return None
    
    try:
        preds = predictions[jz]["preds"]
    except KeyError:
        preds = predictions[jz]["predictions"]
    
    staggered_mag = compute_staggered_magnetization(preds)
    return staggered_mag

def load_staggered_mag_from_skqd(
    jz: float, data_root: str, observable_key: str
) -> Optional[float]:
    """
    Load SKQD data and compute staggered magnetization.
    
    Args:
        jz: Jz value
        data_root: Root directory for SKQD data
        observable_key: Key for Z observable in SKQD data
        
    Returns:
        Staggered magnetization value or None
    """
    skqd_data = load_skqd_data(jz, data_root)
    if skqd_data is None or observable_key not in skqd_data:
        return None
    
    z_values = skqd_data[observable_key]
    staggered_mag = compute_staggered_magnetization(z_values)
    return staggered_mag

def compute_percentage_error(value1: float, value2: float) -> float:
    """
    Compute percentage error between two values.
    
    percentage_error = |value1 - value2| / |value2| * 100
    
    Args:
        value1: First value (e.g., ML prediction)
        value2: Second value (e.g., reference like SKQD)
        
    Returns:
        Percentage error
    """
    if value2 == 0:
        return np.nan
    
    return abs(value1 - value2) / abs(value2) * 100

def collect_staggered_mag_errors(
    all_predictions: Dict,
    data_root: str,
    observable_key: str,
    training_cutoff: float = 1.6,
) -> Dict:
    """
    Collect staggered magnetization percentage errors between ML and SKQD.
    
    Args:
        all_predictions: Dictionary of ML predictions by split
        data_root: Root directory for SKQD data
        observable_key: Key for Z observable
        training_cutoff: Jz value separating boundary from training regime
        
    Returns:
        Dictionary with structure:
        {
            "jz": [...],
            "error": [...],
            "regime": [...]
        }
    """
    # Collect all unique Jz values from predictions
    all_jz = set()
    for split_predictions in all_predictions.values():
        all_jz.update(split_predictions.keys())
    
    all_jz = sorted(all_jz)
    
    # Initialize data structure
    data = {"jz": [], "error": [], "regime": []}
    
    # Collect data for each Jz value
    for jz in all_jz:
        regime = "boundary" if jz < training_cutoff else "training"
        
        # Get ML prediction (from any split that has this Jz)
        ml_mag = None
        for split_predictions in all_predictions.values():
            if jz in split_predictions:
                ml_mag = load_staggered_mag_from_predictions(split_predictions, jz)
                break
        
        # Get SKQD data
        skqd_mag = load_staggered_mag_from_skqd(jz, data_root, observable_key)
        
        # Compute ML vs SKQD error
        if ml_mag is not None and skqd_mag is not None:
            error = compute_percentage_error(ml_mag, skqd_mag)
            if not np.isnan(error):
                data["jz"].append(jz)
                data["error"].append(error)
                data["regime"].append(regime)
    
    return data

def get_boundary_files(boundary_jz_min: float = 1.1, boundary_jz_max: float = 1.4) -> list:
    """Get list of boundary region files."""
    boundary_jz_values = [1.1, 1.2, 1.3, 1.4]
    boundary_files = []
    
    for jz in boundary_jz_values:
        if boundary_jz_min <= jz <= boundary_jz_max:
            fname = f"XXZ_2d_jz_{jz:.1f}.pkl"
            boundary_files.append(fname)
    
    return sorted(boundary_files)

def load_and_evaluate_model(
    model_dir: str,
    device: str = "mps",
    boundary_jz_min: float = 1.1,
    boundary_jz_max: float = 1.4,
) -> Tuple[Dict, Dict]:
    """
    Load a boundary-trained model and evaluate on boundary and training regions.
    
    Returns:
        Tuple of (all_predictions, settings)
    """
    print(f"  Loading model from {model_dir}")
    model, settings = load_model_and_settings(model_dir, device)
    
    data_root = resolve_data_root(settings["dataset_params"]["data_root"], __file__)
    observable_key = settings["dataset_params"]["observable_key"]
    jz_min = settings["dataset_params"]["jz_min"]
    
    dims = extract_model_dimensions(settings, model_dir)
    n_edges = dims["n_edges"]
    
    print(f"    Observable: {observable_key}")
    print(f"    Data root: {data_root}")
    
    all_predictions = {}
    
    # 1. Evaluate on boundary region (test set)
    boundary_files = get_boundary_files(boundary_jz_min, boundary_jz_max)
    
    if len(boundary_files) > 0:
        print(f"\n    Evaluating on BOUNDARY region (jz={boundary_jz_min}-{boundary_jz_max})...")
        dataset_boundary = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_edges,
            observable_key=observable_key,
            jz_min=boundary_jz_min,
            files=boundary_files,
            preload=True,
        )
        
        data_loader = DataLoader(dataset_boundary, batch_size=1, shuffle=False)
        predictions, avg_loss = evaluate_model(model, data_loader, device)
        
        print(f"      Boundary Loss: {avg_loss:.6f}, Jz values: {len(predictions)}")
        all_predictions["boundary"] = predictions
    
    # 2. Evaluate on training split
    train_files = settings["split"]["train_files"]
    if len(train_files) > 0:
        print(f"\n    Evaluating on TRAIN split...")
        dataset_train = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_edges,
            observable_key=observable_key,
            jz_min=jz_min,
            files=train_files,
            preload=True,
        )
        
        data_loader = DataLoader(dataset_train, batch_size=1, shuffle=False)
        predictions, avg_loss = evaluate_model(model, data_loader, device)
        
        print(f"      Train Loss: {avg_loss:.6f}, Jz values: {len(predictions)}")
        all_predictions["train"] = predictions
    
    # 3. Evaluate on validation split
    val_files = settings["split"]["val_files"]
    if len(val_files) > 0:
        print(f"\n    Evaluating on VAL split...")
        dataset_val = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_edges,
            observable_key=observable_key,
            jz_min=jz_min,
            files=val_files,
            preload=True,
        )
        
        data_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
        predictions, avg_loss = evaluate_model(model, data_loader, device)
        
        print(f"      Val Loss: {avg_loss:.6f}, Jz values: {len(predictions)}")
        all_predictions["val"] = predictions
    
    return all_predictions, settings

def load_system_data(
    n_spins: int,
    model_base: str,
    training_cutoff: float,
    cache_dir: str,
    device: str = "mps",
    use_cache: bool = True,
    boundary_jz_min: float = 1.1,
    boundary_jz_max: float = 1.4,
) -> Dict:
    """
    Load data for one system size with caching support.
    
    Args:
        n_spins: Number of spins
        model_base: Base path for model directory
        training_cutoff: Jz value separating boundary from training regime
        cache_dir: Directory for cache files
        device: Device to run model on
        use_cache: Whether to use cached results if available
        boundary_jz_min: Minimum Jz for boundary region
        boundary_jz_max: Maximum Jz for boundary region
        
    Returns:
        Dictionary with error data
    """
    # Determine model type for cache differentiation
    model_type = get_model_type_from_base(model_base)
    cache_path = get_cache_path(n_spins, model_type, cache_dir)
    
    # Try to load from cache first
    if use_cache:
        cached_data = load_cached_results(cache_path)
        if cached_data is not None:
            print(f"Using cached results for n={n_spins} ({model_type})")
            # Print summary
            if len(cached_data["jz"]) > 0:
                n_boundary = sum(1 for r in cached_data["regime"] if r == "boundary")
                n_training = sum(1 for r in cached_data["regime"] if r == "training")
                mean_error = np.mean(cached_data["error"])
                print(f"  n={n_spins}: {len(cached_data['jz'])} points "
                      f"({n_boundary} boundary, {n_training} training), "
                      f"mean error: {mean_error:.2f}%")
            return cached_data
    
    # If not using cache or cache miss, compute results
    print(f"Computing results for n={n_spins} ({model_type})")
    
    model_dir = get_most_recent_model_dir(model_base)
    
    if not os.path.exists(model_dir):
        print(f"Warning: Model directory not found for n={n_spins}: {model_dir}")
        return {"jz": [], "error": [], "regime": []}
    
    print(f"  Model directory: {model_dir}")
    
    # Load and evaluate model directly
    all_predictions, settings = load_and_evaluate_model(
        model_dir, device, boundary_jz_min, boundary_jz_max
    )
    
    if len(all_predictions) == 0:
        print(f"Warning: No predictions generated for n={n_spins}")
        return {"jz": [], "error": [], "regime": []}
    
    data_root = resolve_data_root(settings["dataset_params"]["data_root"], __file__)
    observable_key = settings["dataset_params"]["observable_key"]
    
    # Collect staggered magnetization errors
    data = collect_staggered_mag_errors(
        all_predictions=all_predictions,
        data_root=data_root,
        observable_key=observable_key,
        training_cutoff=training_cutoff,
    )
    
    # Save to cache
    if len(data["jz"]) > 0:
        save_cached_results(data, cache_path)
    
    # Print summary
    if len(data["jz"]) > 0:
        n_boundary = sum(1 for r in data["regime"] if r == "boundary")
        n_training = sum(1 for r in data["regime"] if r == "training")
        mean_error = np.mean(data["error"])
        print(f"  n={n_spins}: {len(data['jz'])} points "
              f"({n_boundary} boundary, {n_training} training), "
              f"mean error: {mean_error:.2f}%")
    
    return data

def plot_combined_staggered_mag_errors(
    results: Dict[int, Dict],
    output_dir: str,
    training_cutoff: float = 1.6,
):
    """
    Create manuscript-style plot of staggered magnetization percentage errors for both system sizes.
    
    Args:
        results: Dictionary with error data for each system size
        output_dir: Directory to save plot
        training_cutoff: Jz value separating boundary from training regime
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create figure with manuscript-optimized aspect ratio
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    
    # Find global max Jz for shaded region
    max_jz = 0
    for n_spins, data in results.items():
        if len(data["jz"]) > 0:
            max_jz = max(max_jz, max(data["jz"]))
    
    if max_jz == 0:
        print("Warning: No data to plot")
        return
    
    # Plot each system size
    for n_spins in [57, 115]:
        if n_spins not in results or len(results[n_spins]["jz"]) == 0:
            continue
        
        data = results[n_spins]
        config = SYSTEM_CONFIGS[n_spins]
        
        jz_vals = np.array(data["jz"])
        error_vals = np.array(data["error"])
        regimes = data["regime"]
        
        # Separate boundary and training points
        boundary_mask = np.array([r == "boundary" for r in regimes])
        training_mask = ~boundary_mask
        
        color = config["color"]
        marker = config["marker"]
        label = config["label"]
        
        # Plot training points (de-emphasized)
        if np.any(training_mask):
            ax.plot(
                jz_vals[training_mask],
                error_vals[training_mask],
                marker=marker,
                color=color,
                linestyle="",
                linewidth=0.8,
                markersize=3,
                markerfacecolor="none",
                markeredgewidth=0.6,
                alpha=0.4,  # De-emphasized
                label=None,  # No label for training points
            )

        # Plot boundary points (emphasized)
        if np.any(boundary_mask):
            ax.plot(
                jz_vals[boundary_mask],
                error_vals[boundary_mask],
                marker=marker,
                color=color,
                linestyle="",
                linewidth=1.2,
                markersize=5,
                markerfacecolor="none",
                markeredgewidth=1.0,
                alpha=1.0,  # Emphasized
                label=label,  # Label only on boundary points
            )
    
    # Add shaded band for training regime
    ax.axvspan(
        training_cutoff,
        max_jz,
        alpha=0.1,
        color="gray",
        zorder=0,
    )
    
    # Add vertical line at training cutoff
    ax.axvline(
        training_cutoff,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        zorder=0,
    )
    
    # Calculate the middle of the shaded region in data coordinates
    mid_jz = (training_cutoff + max_jz) / 2
    
    # Convert to axes coordinates for positioning
    xlim = ax.get_xlim()
    x_pos = (mid_jz - xlim[0]) / (xlim[1] - xlim[0])
    
    # Add text annotation for training regime (in the middle of shaded area)
    ax.text(
        x_pos, 0.98,
        "Training regime",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )
    
    ax.set_xlabel(r"$J_z$")
    ax.set_ylabel(r"$\Delta m_s$ (\%)")
    
    # Let y-axis limits be automatic based on data
    ax.set_ylim(-0.1, 2.5)
    
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5, which="both")
    
    # Create custom legend with non-colored markers
    from matplotlib.lines import Line2D
    legend_elements = []
    
    for n_spins in [57, 115]:
        if n_spins in results and len(results[n_spins]["jz"]) > 0:
            config = SYSTEM_CONFIGS[n_spins]
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker=config["marker"],
                    color="w",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=5,
                    markeredgewidth=0.8,
                    label=config["label"],
                )
            )
    
    # ax.legend(
    #     handles=legend_elements,
    #     frameon=True,
    #     fancybox=False,
    #     edgecolor="black",
    #     framealpha=1.0,
    #     loc="center right",
    #     fontsize=8,
    #     handletextpad=0.5,
    #     ncol=2,
    #     columnspacing=1.0,
    # )
    
    plt.tight_layout(pad=0.2)
    
    # Save plot
    filename = "staggered_mag_error_ml_vs_skqd_boundary_combined_n57_n115.pdf"
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.savefig(
        os.path.join(output_dir, filename.replace(".pdf", ".png")),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()
    
    print(f"Saved combined staggered magnetization error plot: {filename}")

def main():
    """Main execution function."""
    # Configuration variables
    device = "mps"  # Change to "cuda" or "cpu" if needed
    training_cutoff = 1.6
    base_output_dir = "plots/combined_ml_test_boundary"
    base_cache_dir = "plots/combined_ml_test_boundary"
    use_cache = True  # Set to False to force re-evaluation
    
    # Boundary region parameters
    boundary_jz_min = 1.1
    boundary_jz_max = 1.4
    
    # Detect which model folder is being used (check first system config)
    model_folder = detect_model_folder(SYSTEM_CONFIGS[57])
    print(f"Detected model folder: {model_folder}")
    
    # Create model-specific output and cache directories
    output_dir = os.path.join(base_output_dir, model_folder)
    cache_dir = os.path.join(base_cache_dir, "cache")
    
    print(f"\n{'=' * 60}")
    print(f"Processing combined staggered magnetization errors (ML vs SKQD)")
    print(f"Training cutoff: {training_cutoff}")
    print(f"Boundary region: jz={boundary_jz_min}-{boundary_jz_max}")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Use cache: {use_cache}")
    print(f"{'=' * 60}\n")
    
    # Load data for both system sizes
    results = {}
    for n_spins, config in SYSTEM_CONFIGS.items():
        print(f"\nProcessing n={n_spins}...")
        results[n_spins] = load_system_data(
            n_spins=n_spins,
            model_base=config["model_base"],
            training_cutoff=training_cutoff,
            cache_dir=cache_dir,
            device=device,
            use_cache=use_cache,
            boundary_jz_min=boundary_jz_min,
            boundary_jz_max=boundary_jz_max,
        )
    
    # Create combined plot
    print("\nGenerating combined staggered magnetization error plot...")
    plot_combined_staggered_mag_errors(
        results=results,
        output_dir=output_dir,
        training_cutoff=training_cutoff,
    )
    
    print(f"\n{'=' * 60}")
    print(f"Complete! Plot saved to {output_dir}")
    print(f"Cached results saved to {cache_dir}")
    print(f"To force re-evaluation, set use_cache=False in main()")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
