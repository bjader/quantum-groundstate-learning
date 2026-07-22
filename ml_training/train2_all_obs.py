"""
Training script for 2D XXZ observables using ModifiedCombinedFullDNN with HeavyHexGridMap.

This script is similar to train2.py but updated to match train.py's approach:
- Loops over all observable keys automatically
- Uses the same data source as train.py (recovery_random_flip)
- Saves models to model2/ directory
- Uses default hyperparameter configs (custom_config_paths=None)

Key differences from train.py
------------------------------
* Model      : ModifiedCombinedFullDNN  (one SimpleFullDNN per observable output)
* Topology   : HeavyHexGridMap(distance=7, delta1=0, mode="local", use_qiskit=True)
* Input dim  : n_terms = number of edges in the heavy-hex graph (m edges, ~126 for d=7)
               Each sample is a vector of m coupling-strength values (one per edge).
* Sub-network input : a *local subset* of those m values, selected by the
               HeavyHexGridMap parameter_map (see overlap explanation below).

Input overlap between sub-networks (graph distance 0)
------------------------------------------------------
See the docstring at the bottom of this file for a detailed explanation.
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
from qaml.ml.models.geometry import HeavyHexGridMap
from qaml.ml.models.model2 import ModifiedCombinedFullDNNFast
from qaml.ml.training_utils import set_seed, train_one_epoch, evaluate
from qaml.ml.stratified_split import stratified_split_by_jz, print_split_info
from qaml.ml.hyperparameter_utils import initialize_hyperparams, find_most_recent_config

# Config
CONFIG = {
    "observables": ["z_basis_opt", "2_nn_corr_funcs_zz_basis_opt", "z_loop"],
    "custom_config_paths": None,  # or dict: {"z_loop": "path/to/config.json"}
    "patience": 50,
    "ckpt_every": 10,
    "seed": 1,
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def mkdir_save_dir(n_spins, data_gen_method, observable_key):
    save_dir = os.path.join("ml_models", "model2", f"spins_{n_spins}",
                            data_gen_method,
                            observable_key,
                            str(datetime.now()).replace(" ", "-"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir

def make_or_load_split(in_files, test_frac=0.2, val_frac=0.1, seed=1, split_path=None,
                       jz_values=None):
    rng = np.random.default_rng(seed)
    if split_path and os.path.exists(split_path):
        with open(split_path, "r") as f:
            split = json.load(f)
        return split

    train_idx, val_idx, test_idx = stratified_split_by_jz(in_files, jz_values, test_frac, val_frac,
                                                          rng)

    split = {
        "seed": seed,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "train_files": [in_files[i] for i in train_idx],
        "val_files": [in_files[i] for i in val_idx],
        "test_files": [in_files[i] for i in test_idx],
    }
    if split_path:
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
    return split

# ---------------------------------------------------------------------------
# Main training loop — one iteration per observable key
# ---------------------------------------------------------------------------

for observable_key in CONFIG["observables"]:
    print("\n" + "=" * 80)
    print(f"Training model for observable: {observable_key}")
    print("=" * 80)

    set_seed(CONFIG["seed"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    n_qubits = 115
    heavy_hex_distance = 7 if n_qubits == 115 else 5
    graph_distance = 0 # delta1: neighbourhood radius in graph hops (0 = only edge endpoints)

    # Build the HeavyHexGridMap once to inspect topology dimensions
    hh_map = HeavyHexGridMap(
        distance=heavy_hex_distance,
        delta1=graph_distance,
        mode="local",
        use_qiskit=True,
    )
    n_terms = hh_map.m              # number of edges = input feature dimension
    n_edges = n_terms               # alias for clarity: n_edges = number of edges (inputs to model/dataset)

    # The HeisenbergDataset fills the input vector with the scalar jz repeated
    # n_spins times.  ModifiedCombinedFullDNN expects an input of length n_edges
    # (one value per edge).  We therefore pass n_edges as the dataset's n_spins
    # so the input tensor has the correct width; every entry is still jz.

    print(f"Heavy-hex qubits : {n_qubits}")
    print(f"Heavy-hex edges  : {n_edges}  (= n_terms / input dim)")
    print(f"Graph distance   : {graph_distance}")
    print(f"Dataset input dim: {n_edges}  (= n_terms, filled with jz)")

    # ------------------------------------------------------------------
    # Geometry parameters stored in settings (mirrors train.py convention)
    # ------------------------------------------------------------------
    geometry_parameters = {
        "distance": heavy_hex_distance,
        "delta1": graph_distance,  # 0 = only edge endpoints, gives coordination number (2-3 edges)
        "mode": "local",
        "use_qiskit": True,
        "topology": "heavy_hex",
    }

    # ------------------------------------------------------------------
    # Neural-network hyper-parameters
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
    
    hp, hyperparam_config_path = initialize_hyperparams(observable_key, n_qubits,
                                                        custom_config_paths)
    width, depth, act_fun, dropout = hp["width"], hp["depth"], hp["act_fun"], hp["dropout"]
    lr, n_epochs, batch_size = hp["lr"], hp["n_epochs"], hp["batch_size"]

    nn_parameters = {"width": width, "depth": depth, "act_fun": act_fun, "dropout": dropout}

    local_parameters = {"width": width, "depth": depth, "act_fun": act_fun, "dropout": dropout}

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    patience = CONFIG["patience"]
    ckpt_every = CONFIG["ckpt_every"]

    # ------------------------------------------------------------------
    # Data paths  (same as train.py)
    # ------------------------------------------------------------------
    ts, kd, shots, qpu, timestamp = 1, 11, "100k", "boston", "1773150437_1773854302_mixed"
    # ts, kd, shots, qpu, timestamp = 1, 11, "100k", "boston", "1773299045"
    data_gen_method = "skqd"
    data_dir = f"ts_{ts}_kd_{kd}_shots_{shots}_ibm_{qpu}_{timestamp}"
    data_path = f"../data/spins_{n_qubits}/{data_gen_method}/{data_dir}"
    data_root = os.path.join(data_path, "recovery_random_flip")

    jz_min = 1.6

    # ------------------------------------------------------------------
    # Dataset — discover n_outputs from data
    # ------------------------------------------------------------------
    dataset_all = HeisenbergDataset(
        root_dir=data_root,
        n_inputs=n_edges,  # Input dimension = number of edges (Jz coupling values)
        observable_key=observable_key,
        jz_min=jz_min,
    )
    all_in_files = dataset_all.filenames()
    n_outputs = dataset_all.n_outputs

    print(f"n_outputs (observables to predict): {n_outputs}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ModifiedCombinedFullDNNFast(
        n_terms="edges",            # resolved to hh_map.m via get_n_terms
        n_outputs=n_outputs,
        geometry_parameters=geometry_parameters,
        local_parameters=local_parameters,
        device=str(device),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", total_params)
    nn_parameters["trainable_parameters"] = total_params

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------
    save_dir = mkdir_save_dir(n_qubits, data_gen_method, observable_key)
    split_path = os.path.join(save_dir, "split.json")

    split = make_or_load_split(
        all_in_files,
        test_frac=0.2,
        val_frac=0.05,
        seed=CONFIG["seed"],
        split_path=split_path,
        jz_values=dataset_all.jz_values
    )

    print_split_info(split, dataset_all.jz_values)

    ds_train = HeisenbergDataset(
        root_dir=data_root, n_inputs=n_edges, observable_key=observable_key,
        jz_min=jz_min, files=split["train_files"],
    )
    ds_val = (
        HeisenbergDataset(
            root_dir=data_root, n_inputs=n_edges, observable_key=observable_key,
            jz_min=jz_min, files=split["val_files"],
        )
        if len(split["val_files"]) > 0
        else None
    )

    train_loader = DataLoader(
        ds_train, batch_size=min(batch_size, len(ds_train)), shuffle=True, num_workers=0
    )
    val_loader = (
        DataLoader(ds_val, batch_size=min(batch_size, len(ds_val)), shuffle=False, num_workers=0)
        if ds_val is not None
        else None
    )

    print("Device:", device)
    print("Model param device:", next(model.parameters()).device)

    # ------------------------------------------------------------------
    # Optimiser & loss
    # ------------------------------------------------------------------
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_dict = defaultdict(list)
    best_val = float("inf")
    best_epoch = -1
    epoch = -1

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        if val_loader is not None:
            val_loss = evaluate(model, criterion, val_loader, device)
        else:
            val_loss = float("nan")
        loss_dict["train_loss"].append(train_loss)
        loss_dict["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loader is not None and val_loss < best_val - 1e-8:
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

        if epoch % ckpt_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

        if val_loader is not None:
            if len(loss_dict["val_loss"]) >= patience + 1:
                recent = loss_dict["val_loss"][-patience - 1:]
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
    # Persist settings
    # ------------------------------------------------------------------
    with open(os.path.join(save_dir, "settings.json"), "w") as f:
        json.dump(
            {
                "dataset_params": {
                    "data_root": data_root,
                    "observable_key": observable_key,
                    "jz_min": jz_min,
                },
                "split": split,
                "model_hyper_params": {
                    "model_name": model.__class__.__name__,
                    "geometry_parameters": geometry_parameters,
                    "nn_parameters": nn_parameters,
                    "n_terms": n_terms,
                    "n_outputs": n_outputs,
                    "n_edges": n_edges,  # Number of edges (inputs to model/dataset)
                    "n_qubits": n_qubits,  # Actual number of qubits in the system
                },
                "training_hyper_params": {
                    "lr": lr,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "patience": patience,
                    "ckpt_every": ckpt_every,
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
