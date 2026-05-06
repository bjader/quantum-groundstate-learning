from itertools import product
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from qaml.ml.dataset import HeisenbergDataset
from qaml.ml.models.model import LightweightPerObsDNN
from qaml.ml.training_utils import set_seed, train_one_epoch, evaluate
from qaml.ml.stratified_split import stratified_kfold_split, make_stratified_test_split


def run_kfold_cv(
    all_files,
    jz_values,
    data_root,
    n_inputs,
    observable_key,
    jz_min,
    device,
    hyperparams,
    k=5,
    batch_size=32,
    seed=1
):
    folds = stratified_kfold_split(all_files, jz_values, k=k, seed=seed)
    fold_losses = []
    fold_best_epochs = []

    for fold_idx in range(k):
        print(f"  Fold {fold_idx + 1}/{k}")

        val_indices = folds[fold_idx]
        train_indices = [i for f in range(k) if f != fold_idx for i in folds[f]]

        train_files = [all_files[i] for i in train_indices]
        val_files = [all_files[i] for i in val_indices]

        # Print jz distribution for this fold
        train_jz = sorted([jz_values[f] for f in train_files])
        val_jz = sorted([jz_values[f] for f in val_files])
        print(f"    Train: {len(train_files)} files, jz: [{train_jz[0]:.2f}, {train_jz[-1]:.2f}]")
        print(f"    Val: {len(val_files)} files, jz: [{val_jz[0]:.2f}, {val_jz[-1]:.2f}]")

        ds_train = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_inputs,
            observable_key=observable_key,
            jz_min=jz_min,
            files=train_files
        )

        ds_val = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_inputs,
            observable_key=observable_key,
            jz_min=jz_min,
            files=val_files
        )

        train_loader = DataLoader(
            ds_train,
            batch_size=min(batch_size, len(ds_train)),
            shuffle=True
        )

        val_loader = DataLoader(
            ds_val,
            batch_size=min(batch_size, len(ds_val)),
            shuffle=False
        )

        model = LightweightPerObsDNN(
            n_outputs=ds_train.n_outputs,
            width=hyperparams["width"],
            depth=hyperparams["depth"],
            act_fun=hyperparams["act_fun"],
            dropout=hyperparams.get("dropout", 0.0),
            device=str(device)
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable parameters:", total_params)

        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        criterion = nn.MSELoss(reduction="mean")

        best_val = float("inf")
        best_epoch = -1

        for epoch in range(hyperparams["n_epochs"]):
            train_one_epoch(model, criterion, optimizer, train_loader, device)
            val_loss = evaluate(model, criterion, val_loader, device)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch

        fold_losses.append(best_val)
        fold_best_epochs.append(best_epoch)

    return {
        "fold_losses": fold_losses,
        "fold_best_epochs": fold_best_epochs,
        "mean_val_loss": float(np.mean(fold_losses)),
        "std_val_loss": float(np.std(fold_losses)),
        "mean_best_epoch": float(np.mean(fold_best_epochs)),
        "std_best_epoch": float(np.std(fold_best_epochs)),
    }


if __name__ == "__main__":

    set_seed(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    n_qubits = 115
    jz_min = 1.6
    k = 5
    batch_size = 32
    seed = 1

    DATA_ROOT = os.environ.get("DATA_ROOT", "../data")
    data_root = f"{DATA_ROOT}/spins_{n_qubits}/skqd/ts_1_kd_11_shots_100k_ibm_boston_1773299045/recovery_random_flip"

    observable_list = [
        "z_basis_opt",
        "2_nn_corr_funcs_zz_basis_opt",
        "z_loop"
    ]

    # Define hyperparameters to search over
    widths = [8, 32, 64, 128, 256]
    depths = [2, 3, 5]
    lrs = [1e-3, 1e-4]

    hyperparam_list = [
        {
            "width": w,
            "depth": d,
            "act_fun": "tanh",
            "dropout": 0.0,
            "lr": lr,
            "n_epochs": 200
        }
        for w, d, lr in product(widths, depths, lrs)
    ]

    for observable_key in observable_list:

        print("\n==============================================")
        print(f"Hyperparameter Search for {observable_key}")
        print("==============================================")

        dataset_all = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=1,
            observable_key=observable_key,
            jz_min=jz_min
        )

        all_files = dataset_all.filenames()

        # Create stratified test split matching train.py
        train_files, test_files = make_stratified_test_split(
            all_files,
            test_frac=0.2,
            seed=seed,
            jz_values=dataset_all.jz_values
        )

        # Print split information
        train_jz = sorted([dataset_all.jz_values[f] for f in train_files])
        test_jz = sorted([dataset_all.jz_values[f] for f in test_files])
        print(f"\nDataset split (matching train.py):")
        print(
            f"  Train (for k-fold CV): {len(train_files)} files, jz range: [{train_jz[0]:.2f}, {train_jz[-1]:.2f}]")
        print(
            f"  Test (held out): {len(test_files)} files, jz range: [{test_jz[0]:.2f}, {test_jz[-1]:.2f}]")
        print(f"  K-fold CV will use only the {len(train_files)} training files\n")

        results = []

        for config in hyperparam_list:
            print("\nTesting config:", config)

            cv_result = run_kfold_cv(
                all_files=train_files,  # Use only train_files, not all_files
                jz_values=dataset_all.jz_values,
                data_root=data_root,
                n_inputs=1,
                observable_key=observable_key,
                jz_min=jz_min,
                device=device,
                hyperparams=config,
                k=k,
                batch_size=batch_size,
                seed=seed
            )

            result_entry = {
                "hyperparams": config,
                "mean_val_loss": cv_result["mean_val_loss"],
                "std_val_loss": cv_result["std_val_loss"],
                "fold_losses": cv_result["fold_losses"],
                "mean_best_epoch": cv_result["mean_best_epoch"],
                "std_best_epoch": cv_result["std_best_epoch"],
                "fold_best_epochs": cv_result["fold_best_epochs"],
            }

            results.append(result_entry)

            print("  Mean:", cv_result["mean_val_loss"])
            print("  Std:", cv_result["std_val_loss"])

        best_result = min(results, key=lambda x: x["mean_val_loss"])

        print("\nBEST CONFIG FOR", observable_key)
        print(best_result["hyperparams"])
        print("Mean:", best_result["mean_val_loss"])
        print("Std:", best_result["std_val_loss"])

        # Create timestamped save directory
        timestamp = str(datetime.now()).replace(" ", "-")
        save_dir = Path("hyperparam_results") / observable_key / f"n{n_qubits}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        best_config = {
            # Model + training hyperparams
            "hyperparams": {
                **best_result["hyperparams"],
                "batch_size": batch_size,
            },
            # CV performance
            "mean_val_loss": best_result["mean_val_loss"],
            "std_val_loss": best_result["std_val_loss"],
            "fold_losses": best_result["fold_losses"],
            "mean_best_epoch": best_result["mean_best_epoch"],
            "std_best_epoch": best_result["std_best_epoch"],
            "fold_best_epochs": best_result["fold_best_epochs"],
            # Search provenance
            "search_meta": {
                "k_folds": k,
                "seed": seed,
                "n_configs_searched": len(hyperparam_list),
                "results_path": str(save_dir / "results.json"),
                "timestamp": timestamp,
            },
            # Data provenance
            "data_params": {
                "data_root": data_root,
                "n_inputs": 1,
                "observable_key": observable_key,
                "jz_min": jz_min,
            },
            # Test split info (matching train.py)
            "test_split": {
                "test_frac": 0.2,
                "n_train_files": len(train_files),
                "n_test_files": len(test_files),
                "train_jz_range": [float(min(train_jz)), float(max(train_jz))],
                "test_jz_range": [float(min(test_jz)), float(max(test_jz))],
                "note": "Test files held out from k-fold CV, matching train.py stratified split",
            },
        }

        with open(save_dir / "best_config.json", "w") as f:
            json.dump(best_config, f, indent=2)

        print(f"\nResults saved to: {save_dir}")
