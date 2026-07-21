"""
Train ML models on full dataset (jz=1.6 to 6.0) for phase boundary prediction.

This version uses:
- ModifiedCombinedFullDNN
- HeavyHexGridMap

Boundary-training behavior:
- no test set
- 5% validation set
- train on jz in [1.6, 6.0]
- intended for extrapolation toward the phase boundary region (jz=1.1 to 1.4)
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from qaml.ml.dataset import HeisenbergDataset
from qaml.ml.hyperparameter_utils import initialize_hyperparams, find_most_recent_config
from qaml.ml.models.geometry import HeavyHexGridMap
from qaml.ml.models.model2 import ModifiedCombinedFullDNNFast
from qaml.ml.stratified_split import stratified_split_by_jz, print_split_info
from qaml.ml.training_utils import set_seed, train_one_epoch, evaluate


def mkdir_save_dir(n_spins, data_gen_method, observable_key):
    """Create directory for saving model artifacts."""
    save_dir = os.path.join(
        "ml_models",
        "model_boundary2",
        f"spins_{n_spins}",
        data_gen_method,
        observable_key,
        str(datetime.now()).replace(" ", "-"),
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def make_or_load_split(in_files, val_frac=0.05, seed=1, split_path=None, jz_values=None):
    """Create or load train/val split (no test set)."""
    rng = np.random.default_rng(seed)
    if split_path and os.path.exists(split_path):
        with open(split_path, "r") as f:
            split = json.load(f)
        return split

    train_idx, val_idx, _ = stratified_split_by_jz(
        in_files,
        jz_values,
        test_frac=0.0,
        val_frac=val_frac,
        rng=rng,
    )

    split = {
        "seed": seed,
        "test_frac": 0.0,
        "val_frac": val_frac,
        "train_files": [in_files[i] for i in train_idx],
        "val_files": [in_files[i] for i in val_idx],
        "test_files": [],
    }
    if split_path:
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
    return split


CONFIG = {
    "observables": ["z_basis_opt", "2_nn_corr_funcs_zz_basis_opt", "z_loop"],
    "custom_config_paths": None,  # loads most recent by default, or pass dict: {"z_loop": "path/to/config.json"}
    "patience": 50,
    "ckpt_every": 10,
    "seed": 1,
    "val_frac": 0.05,
    "jz_min": 1.6,
    "jz_max": 6.0,
}

for observable_key in CONFIG["observables"]:
    print("\n" + "=" * 80)
    print(f"Training model for observable: {observable_key}")
    print(f"Training range: jz={CONFIG['jz_min']} to {CONFIG['jz_max']}")
    print(f"Validation fraction: {CONFIG['val_frac']*100}%")
    print("=" * 80)

    set_seed(CONFIG["seed"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Heavy-hex geometry
    # ------------------------------------------------------------------
    n_qubits = 115
    heavy_hex_distance = 5 if n_qubits == 57 else 7
    graph_distance = 0

    hh_map = HeavyHexGridMap(
        distance=heavy_hex_distance,
        delta1=graph_distance,
        mode="local",
        use_qiskit=True,
    )
    n_terms = hh_map.m
    n_edges = n_terms

    print(f"Heavy-hex qubits : {n_qubits}")
    print(f"Heavy-hex edges  : {n_edges}")
    print(f"Graph distance   : {graph_distance}")
    print(f"Dataset input dim: {n_edges}")

    geometry_parameters = {
        "distance": heavy_hex_distance,
        "delta1": graph_distance,
        "mode": "local",
        "use_qiskit": True,
        "topology": "heavy_hex",
    }

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    # Automatically find the most recent hyperparameter config for the specific n_qubits
    # Use the heavy_hex_eff directory which contains configs organized by qubit count
    if CONFIG["custom_config_paths"] is None:
        auto_config_path = find_most_recent_config(
            observable_key,
            n_qubits,
            hyperparam_results_dir="hyperparam_results_heavy_hex_eff"
        )
        custom_config_paths = {observable_key: auto_config_path} if auto_config_path else None
    else:
        custom_config_paths = CONFIG["custom_config_paths"]
    
    hp, hyperparam_config_path = initialize_hyperparams(
        observable_key, n_qubits, custom_config_paths
    )

    if hyperparam_config_path is None:
        print("WARNING: Using default hyperparameters instead of CV-optimised hyperparameters!")

    width = hp["width"]
    depth = hp["depth"]
    act_fun = hp["act_fun"]
    dropout = hp["dropout"]
    lr = hp["lr"]
    n_epochs = hp["n_epochs"]
    batch_size = hp["batch_size"]

    nn_parameters = {
        "width": width,
        "depth": depth,
        "act_fun": act_fun,
        "dropout": dropout,
    }

    local_parameters = {
        "width": width,
        "depth": depth,
        "act_fun": act_fun,
        "dropout": dropout,
    }

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    if n_qubits == 57:
        ts, kd, shots, qpu, timestamp = 1, 11, "100k", "boston", "1773299045"
    elif n_qubits == 115:
        ts, kd, shots, qpu, timestamp = 1, 11, "100k", "boston", "1773150437_1773854302_mixed"
    data_gen_method = "skqd"
    data_dir = f"ts_{ts}_kd_{kd}_shots_{shots}_ibm_{qpu}_{timestamp}"
    data_path = f"../data/spins_{n_qubits}/{data_gen_method}/{data_dir}"
    data_root = os.path.join(data_path, "recovery_random_flip")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_all = HeisenbergDataset(
        root_dir=data_root,
        n_inputs=n_edges,
        observable_key=observable_key,
        jz_min=CONFIG["jz_min"],
    )

    all_files_unfiltered = dataset_all.filenames()
    all_in_files = [
        f for f in all_files_unfiltered
        if dataset_all.jz_values[f] <= CONFIG["jz_max"]
    ]
    filtered_jz_values = {f: dataset_all.jz_values[f] for f in all_in_files}
    n_outputs = dataset_all.n_outputs

    print(f"Total files in range [{CONFIG['jz_min']}, {CONFIG['jz_max']}]: {len(all_in_files)}")
    print(f"n_outputs (observables to predict): {n_outputs}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ModifiedCombinedFullDNNFast(
        n_terms="edges",
        n_outputs=n_outputs,
        geometry_parameters=geometry_parameters,
        local_parameters=local_parameters,
        device=str(device),
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", total_params)
    nn_parameters["trainable_parameters"] = total_params

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    save_dir = mkdir_save_dir(n_qubits, data_gen_method, observable_key)
    split_path = os.path.join(save_dir, "split.json")

    split = make_or_load_split(
        all_in_files,
        val_frac=CONFIG["val_frac"],
        seed=CONFIG["seed"],
        split_path=split_path,
        jz_values=filtered_jz_values,
    )

    print_split_info(split, filtered_jz_values)

    ds_train = HeisenbergDataset(
        root_dir=data_root,
        n_inputs=n_edges,
        observable_key=observable_key,
        jz_min=CONFIG["jz_min"],
        files=split["train_files"],
    )
    ds_val = HeisenbergDataset(
        root_dir=data_root,
        n_inputs=n_edges,
        observable_key=observable_key,
        jz_min=CONFIG["jz_min"],
        files=split["val_files"],
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

    print("Device:", device)
    print("Model param device:", next(model.parameters()).device)
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_model_params}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_dict = defaultdict(list)
    best_val = float("inf")
    best_epoch = -1
    epoch = -1

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss = evaluate(model, criterion, val_loader, device)

        loss_dict["train_loss"].append(train_loss)
        loss_dict["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                },
                os.path.join(save_dir, "best_model.pth"),
            )

        if epoch % CONFIG["ckpt_every"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

        if len(loss_dict["val_loss"]) >= CONFIG["patience"] + 1:
            recent = loss_dict["val_loss"][-CONFIG["patience"] - 1:]
            if np.nanmin(recent[1:]) >= recent[0] - 1e-8:
                print(f"Early stopping at epoch {epoch}.")
                break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(save_dir, "last_model.pth"),
    )

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    with open(os.path.join(save_dir, "settings.json"), "w") as f:
        json.dump(
            {
                "dataset_params": {
                    "data_root": data_root,
                    "observable_key": observable_key,
                    "jz_min": CONFIG["jz_min"],
                    "jz_max": CONFIG["jz_max"],
                },
                "split": split,
                "model_hyper_params": {
                    "model_name": model.__class__.__name__,
                    "geometry_parameters": geometry_parameters,
                    "nn_parameters": nn_parameters,
                    "n_terms": n_terms,
                    "n_outputs": n_outputs,
                    "n_edges": n_edges,
                    "n_qubits": n_qubits,
                    "model_param_count": total_params,
                },
                "training_hyper_params": {
                    "lr": lr,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "patience": CONFIG["patience"],
                    "ckpt_every": CONFIG["ckpt_every"],
                    "device": str(device),
                },
                "artifacts": {
                    "best_model": "best_model.pth",
                    "last_model": "last_model.pth",
                    "split_path": "split.json",
                },
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "hyperparam_config_path": hyperparam_config_path,
            },
            f,
            indent=2,
        )

    with open(os.path.join(save_dir, "loss.pkl"), "wb") as f:
        pickle.dump(loss_dict, f)

    plt.figure()
    plt.plot(loss_dict["train_loss"], label="train")
    if not np.all(np.isnan(loss_dict["val_loss"])):
        plt.plot(loss_dict["val_loss"], label="val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.pdf"))
    plt.show()

print("\n" + "=" * 80)
print("Training complete for all observables!")
print("Models saved in model_boundary/ directory")
print("=" * 80)