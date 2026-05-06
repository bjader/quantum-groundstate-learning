from itertools import product
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from qaml.ml.dataset import HeisenbergDataset
from qaml.ml.models.geometry import HeavyHexGridMap
from qaml.ml.models.model2 import ModifiedCombinedFullDNNFast
from qaml.ml.training_utils import set_seed, train_one_epoch, evaluate
from qaml.ml.stratified_split import stratified_kfold_split, make_stratified_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def run_kfold_cv_modified(
    all_files,
    jz_values,
    data_root,
    n_inputs,
    observable_key,
    jz_min,
    device,
    hyperparams,
    geometry_parameters,
    n_terms,
    k=5,
    batch_size=32,
    seed=1,
):
    folds = stratified_kfold_split(all_files, jz_values, k=k, seed=seed)
    fold_losses = []
    fold_best_epochs = []

    for fold_idx in range(k):
        logger.info(f"Fold {fold_idx + 1}/{k}")

        val_indices = folds[fold_idx]
        train_indices = [i for f in range(k) if f != fold_idx for i in folds[f]]

        train_files = [all_files[i] for i in train_indices]
        val_files = [all_files[i] for i in val_indices]

        train_jz = sorted([jz_values[f] for f in train_files])
        val_jz = sorted([jz_values[f] for f in val_files])

        logger.info(f"  Train: {len(train_files)} files, jz: [{train_jz[0]:.2f}, {train_jz[-1]:.2f}]")
        logger.info(f"  Val:   {len(val_files)} files, jz: [{val_jz[0]:.2f}, {val_jz[-1]:.2f}]")

        ds_train = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_inputs,
            observable_key=observable_key,
            jz_min=jz_min,
            files=train_files,
        )

        ds_val = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_inputs,
            observable_key=observable_key,
            jz_min=jz_min,
            files=val_files,
        )

        train_loader = DataLoader(
            ds_train,
            batch_size=min(batch_size, len(ds_train)),
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            ds_val,
            batch_size=min(batch_size, len(ds_val)),
            shuffle=False,
            num_workers=0,
        )

        n_outputs = ds_train.n_outputs
        logger.info(f"  n_outputs={n_outputs}, n_terms={n_terms}")

        model = ModifiedCombinedFullDNNFast(
            n_terms=n_terms,
            n_outputs=n_outputs,
            geometry_parameters=geometry_parameters,
            local_parameters={
                "width": hyperparams["width"],
                "depth": hyperparams["depth"],
                "act_fun": hyperparams["act_fun"],
                "dropout": hyperparams.get("dropout", 0.0),
            },
            device=str(device),
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable parameters: {total_params}")

        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        criterion = nn.MSELoss(reduction="mean")

        best_val = float("inf")
        best_epoch = -1

        for epoch in range(hyperparams["n_epochs"]):
            train_one_epoch(model, criterion, optimizer, train_loader, device)
            val_loss = evaluate(model, criterion, val_loader, device)

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_epoch = epoch

        fold_losses.append(best_val)
        fold_best_epochs.append(best_epoch)

        del model, optimizer, criterion
        torch.cuda.empty_cache()

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    n_qubits = 57
    jz_min = 1.6
    k = 5
    batch_size = 32
    seed = 1

    # Heavy-hex topology settings
    heavy_hex_distance = 7
    graph_distance = 0

    geometry_parameters = {
        "distance": heavy_hex_distance,
        "delta1": graph_distance,
        "mode": "local",
        "use_qiskit": True,
        "topology": "heavy_hex",
    }

    # Build once to get the number of edges / input dimension
    hh_map = HeavyHexGridMap(
        distance=heavy_hex_distance,
        delta1=graph_distance,
        mode="local",
        use_qiskit=True,
    )
    n_terms = hh_map.m
    n_edges = n_terms
    n_inputs = n_edges

    DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
    data_root = f"{DATA_ROOT}/spins_{n_qubits}/skqd/ts_1_kd_11_shots_100k_ibm_boston_1773150437_1773854302_mixed/recovery_random_flip"

    observable_list = [
        "z_basis_opt",
        "2_nn_corr_funcs_zz_basis_opt",
        "z_loop",
    ]

    widths = [8, 16, 32, 64]
    depths = [2, 3, 5]
    lrs = [1e-3, 1e-4]

    hyperparam_list = [
        {
            "width": w,
            "depth": d,
            "act_fun": "tanh",
            "dropout": 0.0,
            "lr": lr,
            "n_epochs": 300,
        }
        for w, d, lr in product(widths, depths, lrs)
    ]

    for observable_key in observable_list:
        logger.info("=" * 50)
        logger.info(f"Hyperparameter search for {observable_key}")
        logger.info("=" * 50)

        dataset_all = HeisenbergDataset(
            root_dir=data_root,
            n_inputs=n_inputs,
            observable_key=observable_key,
            jz_min=jz_min,
        )

        all_files = dataset_all.filenames()

        train_files, test_files = make_stratified_test_split(
            all_files,
            test_frac=0.2,
            seed=seed,
            jz_values=dataset_all.jz_values,
        )

        train_jz = sorted([dataset_all.jz_values[f] for f in train_files])
        test_jz = sorted([dataset_all.jz_values[f] for f in test_files])

        logger.info(f"Train (k-fold CV): {len(train_files)} files, jz: [{train_jz[0]:.2f}, {train_jz[-1]:.2f}]")
        logger.info(f"Test  (held out):  {len(test_files)} files, jz: [{test_jz[0]:.2f}, {test_jz[-1]:.2f}]")

        results = []

        for config in hyperparam_list:
            logger.info(f"Testing config: {config}")

            cv_result = run_kfold_cv_modified(
                all_files=train_files,
                jz_values=dataset_all.jz_values,
                data_root=data_root,
                n_inputs=n_inputs,
                observable_key=observable_key,
                jz_min=jz_min,
                device=device,
                hyperparams=config,
                geometry_parameters=geometry_parameters,
                n_terms=n_terms,
                k=k,
                batch_size=batch_size,
                seed=seed,
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

            logger.info(f"  Mean val loss: {cv_result['mean_val_loss']:.6f} +/- {cv_result['std_val_loss']:.6f}")

        best_result = min(results, key=lambda x: x["mean_val_loss"])

        logger.info(f"Best config for {observable_key}: {best_result['hyperparams']}")
        logger.info(f"  Mean: {best_result['mean_val_loss']:.6f}  Std: {best_result['std_val_loss']:.6f}")

        timestamp = str(datetime.now()).replace(" ", "-")
        save_dir = Path("hyperparam_results_heavy_hex_eff") / observable_key / f"n{n_qubits}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "results2.json", "w") as f:
            json.dump(results, f, indent=2)

        best_config = {
            "hyperparams": {
                **best_result["hyperparams"],
                "batch_size": batch_size,
            },
            "mean_val_loss": best_result["mean_val_loss"],
            "std_val_loss": best_result["std_val_loss"],
            "fold_losses": best_result["fold_losses"],
            "mean_best_epoch": best_result["mean_best_epoch"],
            "std_best_epoch": best_result["std_best_epoch"],
            "fold_best_epochs": best_result["fold_best_epochs"],
            "search_meta": {
                "k_folds": k,
                "seed": seed,
                "n_configs_searched": len(hyperparam_list),
                "results_path": str(save_dir / "results2.json"),
                "timestamp": timestamp,
                "model": "model2.SimpleFullDNN (batched einsum)",
            },
            "data_params": {
                "data_root": data_root,
                "n_inputs": n_inputs,
                "observable_key": observable_key,
                "jz_min": jz_min,
                "heavy_hex_distance": heavy_hex_distance,
                "graph_distance": graph_distance,
                "n_qubits": n_qubits,
                "n_edges": n_edges,
            },
            "test_split": {
                "test_frac": 0.2,
                "n_train_files": len(train_files),
                "n_test_files": len(test_files),
                "train_jz_range": [float(min(train_jz)), float(max(train_jz))],
                "test_jz_range": [float(min(test_jz)), float(max(test_jz))],
                "note": "Test files held out from k-fold CV, matching train.py stratified split",
            },
            "geometry_parameters": geometry_parameters,
        }

        with open(save_dir / "best_config2.json", "w") as f:
            json.dump(best_config, f, indent=2)

        logger.info(f"Results saved to: {save_dir}")
