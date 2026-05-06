import json
import os
from typing import Tuple, Dict

import dill as pickle
import torch
from torch import nn
from torch.utils.data import DataLoader

from qaml.ml.models.model import LightweightPerObsDNN, ModifiedCombinedFullDNN
from qaml.ml.models.model2 import ModifiedCombinedFullDNNFast


def get_most_recent_model_dir(base_path: str) -> str:
    """
    Find the most recent timestamped model directory in the given base path.

    Args:
        base_path: Path like "model/spins_115/skqd/z_basis_opt"

    Returns:
        Full path to most recent model directory, or base_path if no timestamps found
    """
    if not os.path.exists(base_path):
        return base_path

    # List all subdirectories
    subdirs = [d for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d))]

    # Filter for timestamp-like directories (YYYY-MM-DD-HH:MM:SS.microseconds)
    timestamp_dirs = []
    for d in subdirs:
        # Check if it looks like a timestamp (starts with year)
        if d.startswith('20') and len(d) > 10:
            timestamp_dirs.append(d)

    if not timestamp_dirs:
        return base_path

    # Sort by timestamp (lexicographic sort works for ISO format)
    most_recent = sorted(timestamp_dirs)[-1]
    return os.path.join(base_path, most_recent)


def load_model_and_settings(model_dir: str, device: str = "cpu") -> Tuple[nn.Module, Dict]:
    """Load trained model and settings from checkpoint directory."""
    settings_path = os.path.join(model_dir, "settings.json")
    with open(settings_path, "r") as f:
        settings = json.load(f)

    # Determine model class
    model_name = settings["model_hyper_params"]["model_name"]
    
    if model_name == "LightweightPerObsDNN":
        n_outputs = settings["model_hyper_params"]["n_outputs"]
        nn_params = settings["model_hyper_params"]["nn_parameters"]
        model = LightweightPerObsDNN(
            n_outputs=n_outputs,
            width=nn_params["width"],
            depth=nn_params["depth"],
            act_fun=nn_params["act_fun"],
            dropout=nn_params.get("dropout", 0.0),
            device=device
        )
    elif model_name in ["ModifiedCombinedFullDNN", "ModifiedCombinedFullDNNFast"]:
        n_outputs = settings["model_hyper_params"]["n_outputs"]
        n_terms = settings["model_hyper_params"]["n_terms"]
        geometry_parameters = settings["model_hyper_params"]["geometry_parameters"]
        nn_params = settings["model_hyper_params"]["nn_parameters"]
        local_parameters = {
            "width": nn_params["width"],
            "depth": nn_params["depth"],
            "act_fun": nn_params["act_fun"],
            "dropout": nn_params.get("dropout", 0.0)
        }
        
        # Select the appropriate model class
        if model_name == "ModifiedCombinedFullDNNFast":
            ModelClass = ModifiedCombinedFullDNNFast
        else:
            ModelClass = ModifiedCombinedFullDNN
        
        model = ModelClass(
            n_terms=n_terms,
            n_outputs=n_outputs,
            geometry_parameters=geometry_parameters,
            local_parameters=local_parameters,
            device=device
        )
    else:
        raise ValueError(f"Unknown model class: {model_name}")

    # Load best model weights — falls back to last_model when USE_VAL=False
    best_model_artifact = settings["artifacts"]["best_model"]
    if best_model_artifact is not None:
        best_model_path = os.path.join(model_dir, best_model_artifact)
    else:
        best_model_path = None

    if best_model_path is None or not os.path.exists(best_model_path):
        if best_model_path is not None:
            print("Warning: best_model.pth not found, falling back to last_model.pth")
        best_model_path = os.path.join(model_dir, settings["artifacts"]["last_model"])

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    # Use strict=False for both ModifiedCombinedFullDNN variants due to custom parameter structure
    strict = model_name not in ["ModifiedCombinedFullDNN", "ModifiedCombinedFullDNNFast"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.eval()

    return model, settings


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str) -> Tuple[Dict, float]:
    """Evaluate model on a dataset and return predictions and loss."""
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    predictions = {}

    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)

            preds, _ = model(inputs)
            loss = criterion(preds, labels)
            total_loss += loss.item()

            # Process each sample in the batch separately
            # Each sample corresponds to a different Jz value
            for i in range(inputs.shape[0]):
                jz_raw = float(inputs[i, 0].item())
                # Round to 1 decimal to avoid floating point issues
                jz = round(jz_raw, 1)

                pred_i = preds[i].cpu().numpy()
                label_i = labels[i].cpu().numpy()

                # Store predictions and labels (one per Jz)
                predictions[jz] = {"preds": pred_i, "labels": label_i}

    avg_loss = total_loss / max(len(data_loader), 1)
    return predictions, avg_loss


def save_predictions(predictions: Dict, save_path: str):
    """Save predictions to disk."""
    with open(save_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Saved predictions to {save_path}")


def extract_model_dimensions(settings: Dict, model_dir: str) -> Dict[str, int]:
    """
    Extract model dimensions from settings.json.

    Args:
        settings: Loaded settings dictionary
        model_dir: Path to model directory (used as fallback for n_qubits)

    Returns:
        Dictionary with keys:
        - n_edges: Number of edges (model input dimension)
        - n_qubits: Number of qubits/spins in the system
        - n_outputs: Number of observables being predicted (model output dimension)
    """
    model_hyper_params = settings["model_hyper_params"]

    # Extract from settings with fallbacks
    n_edges = model_hyper_params.get("n_edges") or model_hyper_params.get("n_terms")
    n_qubits = model_hyper_params.get("n_qubits")
    n_outputs = model_hyper_params.get("n_outputs")

    # Fallback for n_qubits: extract from model directory path
    if n_qubits is None:
        model_dir_parts = model_dir.split(os.sep)
        for part in model_dir_parts:
            if part.startswith("spins_"):
                n_qubits = int(part.split("_")[1])
                break
        if n_qubits is None:
            raise ValueError(
                f"Could not extract n_qubits from settings or model directory: {model_dir}")

    if n_edges is None or n_outputs is None:
        raise ValueError(f"Could not extract n_edges or n_outputs from settings")

    return {
        "n_edges": n_edges,
        "n_qubits": n_qubits,
        "n_outputs": n_outputs
    }
