"""
Generates plots including bottom middle panel of Figure 4.

This script:
1. Loads boundary-trained models (trained on jz=1.6-6.0) for z, zz, and z_loop observables
2. Evaluates on boundary region (jz=1.1-1.4) and training region (jz=1.6-6.0)
3. Creates combined plots showing percentage error for all three observables
4. Uses caching to make tweaking plots faster
5. Highlights the boundary test regime vs training regime
6. Uses manuscript-style formatting matching test_combined_observables.py

Each plot shows:
- Three observables (z, zz, z_loop) with different markers
- Boundary test points (emphasized) vs training points (de-emphasized)
- Percentage error between ML predictions and SKQD data
- Shaded band for training regime
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from torch.utils.data import DataLoader

from qaml.ml.dataset import HeisenbergDataset
from utils.model_utils import (
    get_most_recent_model_dir,
    load_model_and_settings,
    evaluate_model,
    extract_model_dimensions,
)
from utils.data_utils import load_skqd_data, resolve_data_root

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

# System configurations
SYSTEM_CONFIGS = {
    57: {
        "z_model_base": "../../ml_models/model_boundary2/spins_57/skqd/z_basis_opt",
        "zz_model_base": "../../ml_models/model_boundary2/spins_57/skqd/2_nn_corr_funcs_zz_basis_opt",
        "z_loop_model_base": "../../ml_models/model_boundary2/spins_57/skqd/z_loop",
    },
    115: {
        "z_model_base": "../../ml_models/model_boundary2/spins_115/skqd/z_basis_opt",
        "zz_model_base": "../../ml_models/model_boundary2/spins_115/skqd/2_nn_corr_funcs_zz_basis_opt",
        "z_loop_model_base": "../../ml_models/model_boundary2/spins_115/skqd/z_loop",
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
        Model folder type (e.g., 'model_boundary', 'model_boundary2')
    """
    # Check the first model path to determine folder
    for key, path in config.items():
        if key.endswith("_model_base"):
            return get_model_type_from_base(path)
    # Default to model_boundary if not found
    return "model_boundary"

def get_cache_path(n_spins: int, obs_name: str, split_name: str, cache_dir: str, model_type: str = "model_boundary") -> str:
    """Get path for cached results, differentiated by model type."""
    return os.path.join(cache_dir, f"results_n{n_spins}_{obs_name}_{split_name}_{model_type}.pkl")

def load_cached_results(cache_path: str) -> Optional[Any]:
    """Load cached results if they exist."""
    if os.path.exists(cache_path):
        print(f"  Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def save_cached_results(results: Any, cache_path: str):
    """Save results to cache."""
    Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved results to cache: {cache_path}")

def get_boundary_files(boundary_jz_min: float = 1.1, boundary_jz_max: float = 1.4) -> List[str]:
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
    print(f"Loading model from {model_dir}")
    model, settings = load_model_and_settings(model_dir, device)

    data_root = resolve_data_root(settings["dataset_params"]["data_root"], __file__)
    observable_key = settings["dataset_params"]["observable_key"]
    jz_min = settings["dataset_params"]["jz_min"]

    dims = extract_model_dimensions(settings, model_dir)
    n_edges = dims["n_edges"]

    print(f"  Observable: {observable_key}")
    print(f"  Data root: {data_root}")

    all_predictions = {}

    # 1. Evaluate on boundary region (test set)
    boundary_files = get_boundary_files(boundary_jz_min, boundary_jz_max)

    if len(boundary_files) > 0:
        print(f"\n  Evaluating on BOUNDARY region (jz={boundary_jz_min}-{boundary_jz_max})...")
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

        print(f"    Boundary Loss: {avg_loss:.6f}, Jz values: {len(predictions)}")
        all_predictions["boundary"] = predictions

    # 2. Evaluate on training split
    train_files = settings["split"]["train_files"]
    if len(train_files) > 0:
        print(f"\n  Evaluating on TRAIN split...")
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

        print(f"    Train Loss: {avg_loss:.6f}, Jz values: {len(predictions)}")
        all_predictions["train"] = predictions

    # 3. Evaluate on validation split
    val_files = settings["split"]["val_files"]
    if len(val_files) > 0:
        print(f"\n  Evaluating on VAL split...")
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

        print(f"    Val Loss: {avg_loss:.6f}, Jz values: {len(predictions)}")
        all_predictions["val"] = predictions

    return all_predictions, settings

def compute_percentage_errors(
    predictions: Dict, data_root: str, observable_key: str, training_cutoff: float = 1.6
) -> Tuple[List[float], List[float], List[float], List[str]]:
    """
    Compute percentage errors between predictions and SKQD data.

    Returns:
        Tuple of (jz_values, mean_errors, std_errors, regimes)
    """
    jz_values = []
    mean_errors = []
    std_errors = []
    regimes = []

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
            regimes.append("boundary" if jz < training_cutoff else "training")

    return jz_values, mean_errors, std_errors, regimes

# Split configurations (matching test_combined_observables.py)
SPLIT_CONFIGS = {
    "train": {"color": "C0", "label": "Train"},
    "val": {"color": "C1", "label": "Val"},
    "boundary": {"color": "C2", "label": "Test"},  # boundary is the test set
}

def plot_combined_observables_boundary(
    results: Dict[str, Dict], n_spins: int, output_dir: str, training_cutoff: float = 1.6
):
    """
    Create combined plot showing all three observables.

    Args:
        results: Dictionary mapping observable names to their results
                 Each result contains: {split: (jz_values, mean_errors, std_errors, regimes)}
        n_spins: Number of spins (57 or 115)
        output_dir: Directory to save plots
        training_cutoff: Jz value separating boundary from training regime
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Aspect ratio optimized for stacking in single column
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

        # Plot each split with its own color
        for split_name, split_config in SPLIT_CONFIGS.items():
            if split_name not in obs_results:
                continue

            jz_values, mean_errors, std_errors, regimes = obs_results[split_name]

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

    # Create custom legend with black markers
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
    filename = f"combined_obs_error_boundary_n{n_spins}.pdf"
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

    print(f"Saved combined boundary plot: {filename}")

def plot_boundary_observable_comparison(
    all_results: Dict[int, Dict[str, Dict]],
    obs_name: str,
    output_dir: str,
    boundary_jz_min: float = 1.1,
    boundary_jz_max: float = 1.4,
):
    """Plot one observable for n=57 and n=115 on the same boundary-only axes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    obs_labels = {
        "zz": r"$\Delta C_{ZZ}$ (\%)",
        "z_loop": r"$\Delta Z_{\mathrm{loop}}$ (\%)",
    }
    filenames = {
        "zz": "czz_boundary_ml_vs_skqd_n57_n115.pdf",
        "z_loop": "zloop_boundary_ml_vs_skqd_n57_n115.pdf",
    }
    system_styles = {
        57: {"marker": "v", "color": "C0", "label": r"$n=57$"},
        115: {"marker": "s", "color": "C1", "label": r"$n=115$"},
    }

    fig, ax = plt.subplots(figsize=(1.7, 1.8))

    for n_spins in [57, 115]:
        if n_spins not in all_results or obs_name not in all_results[n_spins]:
            continue
        obs_results = all_results[n_spins][obs_name]
        if "boundary" not in obs_results:
            continue

        jz_values, mean_errors, std_errors, regimes = obs_results["boundary"]
        if len(jz_values) == 0:
            continue

        style = system_styles[n_spins]
        ax.errorbar(
            jz_values,
            mean_errors,
            yerr=std_errors,
            fmt=style["marker"],
            color=style["color"],
            label=style["label"],
            markersize=5,
            linewidth=1.2,
            capsize=2,
            capthick=0.8,
            markerfacecolor="none",
            markeredgewidth=1.0,
            alpha=1.0,
        )

    ax.set_xlabel(r"$J_z$")
    ax.set_ylabel(obs_labels[obs_name])
    ax.set_xlim(boundary_jz_min - 0.05, boundary_jz_max + 0.05)
    ax.set_ylim(-0.1, 7)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5, which="both")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0], [0],
            marker=system_styles[n]["marker"],
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=5,
            markeredgewidth=0.8,
            label=system_styles[n]["label"],
        )
        for n in [57, 115]
        if n in all_results
        and obs_name in all_results[n]
        and "boundary" in all_results[n][obs_name]
        and len(all_results[n][obs_name]["boundary"][0]) > 0
    ]

    # if legend_elements:
    #     ax.legend(
    #         handles=legend_elements,
    #         frameon=True,
    #         fancybox=False,
    #         edgecolor="black",
    #         framealpha=1.0,
    #         loc="upper right",
    #         fontsize=8,
    #         handletextpad=0.5,
    #         ncol=1,
    #     )

    plt.tight_layout(pad=0.2)

    filename = filenames[obs_name]
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

    print(f"Saved combined boundary comparison plot: {filename}")

def process_system(
    n_spins: int,
    device: str,
    output_dir: str,
    cache_dir: str,
    use_cache: bool = True,
    boundary_jz_min: float = 1.1,
    boundary_jz_max: float = 1.4,
    training_cutoff: float = 1.6,
) -> Dict[str, Dict]:
    """Process a single system (n=57 or n=115)."""
    print("=" * 80)
    print(f"Processing n={n_spins} (Boundary-trained models)")
    print(f"Boundary region: jz={boundary_jz_min}-{boundary_jz_max}")
    print(f"Training cutoff: jz={training_cutoff}")
    print("=" * 80)

    config = SYSTEM_CONFIGS[n_spins]

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

        if not os.path.exists(model_dir):
            print(f"Warning: Model directory not found: {model_dir}")
            print(f"Skipping {obs_name} observable")
            continue

        obs_results = {}

        # Process each split
        for split_name in ["boundary", "train", "val"]:
            # Check cache first
            cache_path = get_cache_path(n_spins, obs_name, split_name, cache_dir, model_type)
            if use_cache:
                cached_results = load_cached_results(cache_path)
                if cached_results is not None:
                    obs_results[split_name] = cached_results
                    print(f"  Using cached results for {split_name} split")
                    continue

            # Need to evaluate - but only do it once for all splits
            if split_name == "boundary" or (split_name in ["train", "val"] and "boundary" not in obs_results):
                # Load and evaluate model (gets all splits at once)
                all_predictions, settings = load_and_evaluate_model(
                    model_dir, device, boundary_jz_min, boundary_jz_max
                )

                data_root = resolve_data_root(settings["dataset_params"]["data_root"], __file__)
                observable_key = OBSERVABLE_CONFIGS[obs_name]["key"]

                # Compute errors for each split
                for split in ["boundary", "train", "val"]:
                    if split not in all_predictions:
                        continue

                    print(f"\n  Computing errors for {split} split...")
                    jz_values, mean_errors, std_errors, regimes = compute_percentage_errors(
                        all_predictions[split], data_root, observable_key, training_cutoff
                    )

                    if len(jz_values) > 0:
                        split_results = (jz_values, mean_errors, std_errors, regimes)
                        obs_results[split] = split_results

                        # Save to cache
                        split_cache_path = get_cache_path(n_spins, obs_name, split, cache_dir, model_type)
                        save_cached_results(split_results, split_cache_path)

                        print(f"    Jz range: {min(jz_values):.1f} - {max(jz_values):.1f}")
                        print(f"    Mean error: {np.mean(mean_errors):.2f}%")

                break  # Don't re-evaluate for other splits

        results[obs_name] = obs_results

    # Create combined plot
    print(f"\n{'=' * 80}")
    print(f"Generating combined boundary plot for n={n_spins}")
    print(f"{'=' * 80}")

    plot_combined_observables_boundary(results, n_spins, output_dir, training_cutoff)
    return results

def main():
    """Main execution function."""
    device = "mps"  # Change to "cuda" or "cpu" if needed
    base_output_dir = "plots/combined_ml_test_boundary"
    base_cache_dir = "plots/combined_ml_test_boundary"
    use_cache = True  # Set to False to force re-evaluation

    # Boundary region parameters
    boundary_jz_min = 1.1
    boundary_jz_max = 1.4
    training_cutoff = 1.6

    # Detect which model folder is being used (check first system config)
    model_folder = detect_model_folder(SYSTEM_CONFIGS[57])
    print(f"Detected model folder: {model_folder}")

    # Create model-specific output and cache directories
    output_dir = os.path.join(base_output_dir, model_folder)
    cache_dir = os.path.join(base_cache_dir, "cache")

    print("=" * 80)
    print("Combined Observable Testing for Boundary-Trained ML Models")
    print("Processing both n=57 and n=115")
    print(f"Boundary test region: jz={boundary_jz_min}-{boundary_jz_max}")
    print(f"Training region: jz>={training_cutoff}")
    if use_cache:
        print("Using cached results if available")
    print(f"Plots will be saved to {output_dir}")
    print("=" * 80)
    
    # Process both systems and retain results for combined boundary-only comparisons
    all_system_results = {}
    for n_spins in [57, 115]:
        all_system_results[n_spins] = process_system(
            n_spins,
            device,
            output_dir,
            cache_dir,
            use_cache,
            boundary_jz_min,
            boundary_jz_max,
            training_cutoff,
        )
        print()

    print("=" * 80)
    print("Generating boundary-only combined comparison plots")
    print("=" * 80)
    plot_boundary_observable_comparison(
        all_system_results, "zz", output_dir, boundary_jz_min, boundary_jz_max
    )
    plot_boundary_observable_comparison(
        all_system_results, "z_loop", output_dir, boundary_jz_min, boundary_jz_max
    )
    
    print("=" * 80)
    print(f"All testing complete! Plots saved to {output_dir}")
    print("Generated plots:")
    print(f"  - combined_obs_error_boundary_n57.pdf")
    print(f"  - combined_obs_error_boundary_n115.pdf")
    print(f"  - czz_boundary_ml_vs_skqd_n57_n115.pdf")
    print(f"  - zloop_boundary_ml_vs_skqd_n57_n115.pdf")
    print(f"\nCached results saved to {cache_dir}")
    print("To force re-evaluation, set use_cache=False in main()")
    print("=" * 80)

if __name__ == "__main__":
    main()
