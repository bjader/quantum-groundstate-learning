"""
Script to generate plots in Figure 3.

Combined test script for Z, ZZ, and Z_loop observables in 2D XXZ models.

This script:
1. Loads trained models for z, zz, and z_loop observables for both n=57 and n=115
2. Evaluates on train/validation/test splits (or loads cached results)
3. Creates two combined plots (one for n=57, one for n=115)
4. Each plot shows percentage error for all three observables
5. Uses manuscript-style formatting optimized for stacking in single column

Each plot shows:
- Three observables (z, zz, z_loop) with different markers
- Three splits (train, val, test) with different colors
- Lines connecting the markers
- Percentage error between ML predictions and SKQD data
- Fixed y-axis limit of 5% for consistency
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

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
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# Observable configurations
OBSERVABLE_CONFIGS = {
    "z": {
        "key": "z_basis_opt",
        "label": r"$Z$",
        "marker": "o",
    },
    "zz": {
        "key": "2_nn_corr_funcs_zz_basis_opt",
        "label": r"$C_{ZZ}$",
        "marker": "s",
    },
    "z_loop": {
        "key": "z_loop",
        "label": r"$Z_{\mathrm{loop}}$",
        "marker": "^",
    },
}

# Split configurations
SPLIT_CONFIGS = {
    "train": {"color": "C0", "label": "Train"},
    "val": {"color": "C1", "label": "Val"},
    "test": {"color": "C2", "label": "Test"},
}

# System configurations
SYSTEM_CONFIGS = {
    57: {
        "z_model_base": "../../ml_models/model2/spins_57/skqd/z_basis_opt",
        "zz_model_base": "../../ml_models/model2/spins_57/skqd/2_nn_corr_funcs_zz_basis_opt",
        "z_loop_model_base": "../../ml_models/model2/spins_57/skqd/z_loop",
    },
    115: {
        "z_model_base": "../../ml_models/model2/spins_115/skqd/z_basis_opt",
        "zz_model_base": "../../ml_models/model2/spins_115/skqd/2_nn_corr_funcs_zz_basis_opt",
        "z_loop_model_base": "../../ml_models/model2/spins_115/skqd/z_loop",
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
    Detect which model folder is being used.
    
    Args:
        config: System configuration dictionary with model paths
        
    Returns:
        Model folder type (e.g., 'model', 'model2', 'model_boundary', 'model_boundary2')
    """
    # Check the first model path to determine folder
    for key, path in config.items():
        if key.endswith("_model_base"):
            return get_model_type_from_base(path)
    # Default to model if not found
    return "model"

def get_cache_path(n_spins: int, obs_name: str, cache_dir: str, model_type: str = "model") -> str:
    """Get path for cached results, differentiated by model type."""
    return os.path.join(cache_dir, f"results_n{n_spins}_{obs_name}_{model_type}.pkl")

def load_cached_results(cache_path: str) -> Dict:
    """Load cached results if they exist."""
    if os.path.exists(cache_path):
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def save_cached_results(results: Dict, cache_path: str):
    """Save results to cache."""
    Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved results to cache: {cache_path}")

def load_and_evaluate_model(
    model_dir: str, device: str = "mps"
) -> Tuple[Dict, Dict]:
    """
    Load a model and evaluate on all splits.

    Returns:
        Tuple of (all_predictions, settings)
    """
    print(f"Loading model from {model_dir}")
    model, settings = load_model_and_settings(model_dir, device)

    data_root = resolve_data_root(settings["dataset_params"]["data_root"], __file__)
    observable_key = settings["dataset_params"]["observable_key"]
    jz_min = settings["dataset_params"]["jz_min"]

    dims = extract_model_dimensions(settings, model_dir)
    n_edges = dims["n_edges"]

    print(f"  Observable: {observable_key}")
    print(f"  Data root: {data_root}")

    splits = ["train", "val", "test"]
    all_predictions = {}

    for split in splits:
        split_files = settings["split"][f"{split}_files"]
        if len(split_files) == 0:
            print(f"  No files in {split} split, skipping")
            continue

        dataset = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_edges,
            observable_key=observable_key,
            jz_min=jz_min,
            files=split_files,
            preload=True,
        )

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions, avg_loss = evaluate_model(model, data_loader, device)

        print(f"  {split.capitalize()} Loss: {avg_loss:.6f}, Jz values: {len(predictions)}")
        all_predictions[split] = predictions

    return all_predictions, settings

def compute_percentage_errors(
    predictions: Dict, data_root: str, observable_key: str
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute percentage errors between predictions and SKQD data.

    Returns:
        Tuple of (jz_values, mean_errors, std_errors)
    """
    jz_values = []
    mean_errors = []
    std_errors = []

    for jz in sorted(predictions.keys()):
        # Load SKQD data
        skqd_data = load_skqd_data(jz, data_root)
        if skqd_data is None:
            continue

        if observable_key not in skqd_data:
            print(f"  Warning: {observable_key} not found in SKQD data for Jz={jz}")
            continue

        # Get predictions and labels
        try:
            preds = predictions[jz]["preds"]
        except KeyError:
            preds = predictions[jz]["predictions"]

        labels = predictions[jz]["labels"]

        # Compute percentage error
        # Use labels (SKQD data) as reference
        denom = np.where(labels != 0, labels, np.nan)
        rel_error = np.abs((labels - preds) / denom) * 100

        # Filter out NaN values
        valid_errors = rel_error[~np.isnan(rel_error)]

        if len(valid_errors) > 0:
            jz_values.append(jz)
            mean_errors.append(float(np.mean(valid_errors)))
            std_errors.append(float(np.std(valid_errors)))

    return jz_values, mean_errors, std_errors

def plot_combined_observables(
    results: Dict[str, Dict], n_spins: int, output_dir: str
):
    """
    Create combined plot showing all three observables.

    Args:
        results: Dictionary mapping observable names to their results
                 Each result contains: {split: (jz_values, mean_errors, std_errors)}
        n_spins: Number of spins (57 or 115)
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Aspect ratio optimized for stacking in single column (slightly more vertical)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    # Track which observables have been added to legend
    legend_added = set()

    # Plot each observable
    for obs_name, obs_config in OBSERVABLE_CONFIGS.items():
        if obs_name not in results:
            continue

        obs_results = results[obs_name]
        marker = obs_config["marker"]
        label_prefix = obs_config["label"]

        # Plot each split
        for split_name, split_config in SPLIT_CONFIGS.items():
            if split_name not in obs_results:
                continue

            jz_values, mean_errors, std_errors = obs_results[split_name]

            if len(jz_values) == 0:
                continue

            color = split_config["color"]

            # Only add to legend once per observable (no color, no split label)
            if obs_name not in legend_added:
                label = label_prefix
                legend_added.add(obs_name)
            else:
                label = None

            # Plot with error bars, NO LINES between markers
            ax.errorbar(
                jz_values,
                mean_errors,
                yerr=std_errors,
                fmt=marker,  # No line, just marker
                color=color,
                label=label,
                markersize=4,
                linewidth=0.8,
                capsize=2,
                capthick=0.6,
                markerfacecolor="none",
                markeredgewidth=0.8,
                alpha=0.7,
            )

    ax.set_xlabel(r"$J_z$")
    ax.set_ylabel(r"$\Delta O (\%)$")
    ax.set_title(rf"$n={n_spins}$", fontsize=10)

    # Set fixed y-axis limit to 4%
    ax.set_ylim(-0.2, 4.0)

    ax.tick_params(which="both", direction="in", top=True, right=True)

    # Clever legend: just show observable symbols (no colors)
    # Create custom legend handles with black markers
    from matplotlib.lines import Line2D
    legend_elements = []
    for obs_name, obs_config in OBSERVABLE_CONFIGS.items():
        if obs_name in results:
            legend_elements.append(
                Line2D([0], [0], marker=obs_config["marker"], color='w',
                       markerfacecolor='none', markeredgecolor='black',
                       markersize=5, markeredgewidth=0.8, label=obs_config["label"])
            )

    ax.legend(
        handles=legend_elements,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        ncol=3,
        loc="upper right",
        fontsize=7,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5, which="both")

    plt.tight_layout(pad=0.2)

    # Save plot
    filename = f"combined_obs_error_n{n_spins}.pdf"
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

    print(f"Saved combined plot: {filename}")

def process_system(n_spins: int, device: str, base_output_dir: str, base_cache_dir: str,
                   use_cache: bool = True):
    """Process a single system (n=57 or n=115)."""
    print("=" * 80)
    print(f"Processing n={n_spins}")
    print("=" * 80)

    config = SYSTEM_CONFIGS[n_spins]
    
    # Detect which model folder is being used
    model_folder = detect_model_folder(config)
    print(f"Detected model folder: {model_folder}")
    
    # Create model-specific output and cache directories
    output_dir = os.path.join(base_output_dir, model_folder)
    cache_dir = os.path.join(base_cache_dir, model_folder, "cache")

    # Get model directories and determine model types
    z_model_base = config["z_model_base"]
    zz_model_base = config["zz_model_base"]
    z_loop_model_base = config["z_loop_model_base"]
    
    z_model_dir = get_most_recent_model_dir(z_model_base)
    zz_model_dir = get_most_recent_model_dir(zz_model_base)
    z_loop_model_dir = get_most_recent_model_dir(z_loop_model_base)
    
    # Get model types for cache differentiation
    z_model_type = get_model_type_from_base(z_model_base)
    zz_model_type = get_model_type_from_base(zz_model_base)
    z_loop_model_type = get_model_type_from_base(z_loop_model_base)

    # Dictionary to store results for all observables
    results = {}

    # Process each observable
    for obs_name, model_dir, model_type in [
        ("z", z_model_dir, z_model_type),
        ("zz", zz_model_dir, zz_model_type),
        ("z_loop", z_loop_model_dir, z_loop_model_type),
    ]:
        print(f"\n{'=' * 80}")
        print(f"Processing {obs_name.upper()} observable ({model_type})")
        print(f"{'=' * 80}")

        # Check cache first
        cache_path = get_cache_path(n_spins, obs_name, cache_dir, model_type)
        if use_cache:
            cached_results = load_cached_results(cache_path)
            if cached_results is not None:
                results[obs_name] = cached_results
                print(f"  Using cached results")
                continue

        if not os.path.exists(model_dir):
            print(f"Warning: Model directory not found: {model_dir}")
            print(f"Skipping {obs_name} observable")
            continue

        # Load and evaluate model
        all_predictions, settings = load_and_evaluate_model(model_dir, device)

        data_root = resolve_data_root(settings["dataset_params"]["data_root"], __file__)
        observable_key = OBSERVABLE_CONFIGS[obs_name]["key"]

        # Compute errors for each split
        obs_results = {}
        for split_name in ["train", "val", "test"]:
            if split_name not in all_predictions:
                continue

            print(f"\nComputing errors for {split_name} split...")
            jz_values, mean_errors, std_errors = compute_percentage_errors(
                all_predictions[split_name], data_root, observable_key
            )

            if len(jz_values) > 0:
                obs_results[split_name] = (jz_values, mean_errors, std_errors)
                print(f"  Jz range: {min(jz_values):.1f} - {max(jz_values):.1f}")
                print(f"  Mean error: {np.mean(mean_errors):.2f}%")

        results[obs_name] = obs_results

        # Save to cache
        save_cached_results(obs_results, cache_path)

    # Create combined plot
    print(f"\n{'=' * 80}")
    print(f"Generating combined plot for n={n_spins}")
    print(f"{'=' * 80}")

    plot_combined_observables(results, n_spins, output_dir)

def main():
    """Main execution function."""
    device = "mps"  # Change to "cuda" or "cpu" if needed
    base_output_dir = "plots/combined_ml_test/"
    base_cache_dir = "plots/combined_ml_test/"
    use_cache = True  # Set to False to force re-evaluation

    print("=" * 80)
    print("Combined Observable Testing for ML Models")
    print("Processing both n=57 and n=115")
    if use_cache:
        print("Using cached results if available")
    print("Plots will be saved to model-specific subdirectories (model/ or model2/)")
    print("=" * 80)

    # Process both systems
    for n_spins in [57, 115]:
        process_system(n_spins, device, base_output_dir, base_cache_dir, use_cache)
        print()

    print("=" * 80)
    print(f"All testing complete! Plots saved to {base_output_dir}/[model|model2]/")
    print("Generated plots:")
    print(f"  - combined_obs_error_n57.pdf")
    print(f"  - combined_obs_error_n115.pdf")
    print(f"\nCached results saved to {base_cache_dir}/[model|model2]/cache/")
    print("To force re-evaluation, set use_cache=False in main()")
    print("=" * 80)

if __name__ == "__main__":
    main()
