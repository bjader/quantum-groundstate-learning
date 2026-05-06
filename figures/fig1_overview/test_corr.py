"""

Creates right panel of Figure 1.

Test script for correlation function observables in 2D XXZ models.

This script:
1. Loads trained models from checkpoints
2. Evaluates on train/validation/test splits
3. Saves raw predictions to disk
4. Generates comparison plots against SKQD training data
5. Generates comparison plots against DMRG reference data
6. Generates heavy hex lattice visualizations

Focuses specifically on correlation function (XX, ZZ, etc.) testing.
"""

import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader

from qaml.ml.dataset import HeisenbergDataset
from utils.data_utils import resolve_data_root
from utils.plotting_utils import (
    extract_nn_from_observable_key,
    plot_correlation_functions_vs_skqd_with_splits,
    plot_correlation_functions_vs_dmrg_with_splits,
    plot_heavy_hex_correlations_vs_skqd,
    plot_heavy_hex_correlations_vs_dmrg,
)
from utils.model_utils import get_most_recent_model_dir, load_model_and_settings, evaluate_model, \
    save_predictions, extract_model_dimensions

def main():
    parser = argparse.ArgumentParser(
        description="Test trained ML models on correlation function observables")
    parser.add_argument("--model_dir", type=str,
                        default=get_most_recent_model_dir(
                            "../../ml_models/model2/spins_115/skqd/2_nn_corr_funcs_zz_basis_opt"),
                        help="Path to model directory containing settings.json and checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on (cpu, cuda, mps)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--dmrg_root", type=str,
                        default=None,
                        help="Path to DMRG data root for comparison plots")

    args = parser.parse_args()

    # Load model and settings
    print(f"Loading model from {args.model_dir}")
    model, settings = load_model_and_settings(args.model_dir, args.device)

    # Extract settings
    data_root = settings["dataset_params"]["data_root"]
    observable_key = settings["dataset_params"]["observable_key"]
    jz_min = settings["dataset_params"]["jz_min"]
    
    # Resolve data_root to absolute path
    data_root = resolve_data_root(data_root, __file__)

    # Verify this is a correlation function model
    if "corr" not in observable_key.lower() or "loop" in observable_key.lower():
        print(
            f"Warning: This script is designed for correlation functions, but model uses '{observable_key}'")
        print("Continuing anyway...")

    # Extract nn from observable key
    nn = extract_nn_from_observable_key(observable_key)

    # Extract model dimensions using helper function
    dims = extract_model_dimensions(settings, args.model_dir)
    n_edges = dims["n_edges"]  # Number of edges (model input dimension)
    n_qubits = dims["n_qubits"]  # Number of qubits/spins in the system
    n_outputs = dims["n_outputs"]  # Number of observables being predicted (model output dimension)

    print(f"Observable: {observable_key}")
    print(f"Number of edges (model input): {n_edges}")
    print(f"Number of qubits (system nodes): {n_qubits}")
    print(f"Number of outputs (observables): {n_outputs}")
    print(f"Nearest neighbors (nn): {nn}")
    print(f"Data root: {data_root}")

    # Create save directory
    results_dir = os.path.join(args.model_dir, "test_results_corr")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load datasets for each split
    splits = ["train", "val", "test"]
    all_predictions = {}

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {split} split")
        print(f"{'=' * 60}")

        # Get files for this split
        split_files = settings["split"][f"{split}_files"]
        if len(split_files) == 0:
            print(f"No files in {split} split, skipping")
            continue

        # Create dataset (use n_edges for input dimensions)
        dataset = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_edges,
            observable_key=observable_key,
            jz_min=jz_min,
            files=split_files,
            preload=True
        )

        # Use batch_size=1 to ensure each Jz file is processed separately
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Evaluate
        predictions, avg_loss = evaluate_model(model, data_loader, args.device)
        print(f"{split.capitalize()} Loss: {avg_loss:.6f}")
        print(f"Number of Jz values: {len(predictions)}")

        # Save predictions
        pred_save_path = os.path.join(results_dir, f"predictions_{split}.pkl")
        save_predictions(predictions, pred_save_path)

        all_predictions[split] = predictions

    # Generate plots
    print(f"\n{'=' * 60}")
    print("Generating comparison plots")
    print(f"{'=' * 60}")

    # Plots vs SKQD (use n_outputs for correlation pairs)
    skqd_plot_dir = os.path.join(results_dir, "plots_vs_skqd")
    print("Generating correlation function plots vs SKQD (all splits)...")
    plot_correlation_functions_vs_skqd_with_splits(
        all_predictions=all_predictions,
        observable_key=observable_key,
        save_dir=skqd_plot_dir,
        split_config=None,  # Use default train/val/test configuration
        training_cutoff=None
    )

    # Generate heavy hex plots vs SKQD for correlation functions (test split only)
    if "_nn_corr_funcs_zz_basis_opt" in observable_key and "test" in all_predictions:
        print("\nGenerating heavy hex lattice visualizations vs SKQD (test split)...")
        plot_heavy_hex_correlations_vs_skqd(
            predictions=all_predictions["test"],
            observable_key=observable_key,
            n_spins=n_qubits,
            data_root=data_root,
            save_dir=skqd_plot_dir,
            nn=nn,
            split_name="test"
        )

    # Plots vs DMRG (use n_outputs for correlation pairs)
    if args.dmrg_root is not None and os.path.isdir(args.dmrg_root):
        print("\nGenerating DMRG comparison plots (all splits combined)...")
        dmrg_plot_dir = os.path.join(results_dir, "plots_vs_dmrg")

        plot_correlation_functions_vs_dmrg_with_splits(
            all_predictions=all_predictions,
            observable_key=observable_key,
            n_spins=n_outputs,
            dmrg_root=args.dmrg_root,
            data_root=data_root,
            save_dir=dmrg_plot_dir,
            nn=nn,
            split_config=None,  # Use default train/val/test configuration
            training_cutoff=None,
            include_skqd_comparison=False
        )

        # Generate heavy hex plots vs DMRG for correlation functions (test split only)
        if "_nn_corr_funcs_zz_basis_opt" in observable_key and "test" in all_predictions:
            print("\nGenerating heavy hex lattice visualizations vs DMRG (test split)...")
            plot_heavy_hex_correlations_vs_dmrg(
                predictions=all_predictions["test"],
                observable_key=observable_key,
                n_spins=n_qubits,
                data_root=data_root,
                dmrg_root=args.dmrg_root,
                save_dir=dmrg_plot_dir,
                nn=nn,
                split_name="test"
            )
    elif args.dmrg_root is not None:
        print(f"\nWarning: DMRG root directory not found: {args.dmrg_root}")

    print(f"\n{'=' * 60}")
    print(f"Testing complete! Results saved to {results_dir}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
