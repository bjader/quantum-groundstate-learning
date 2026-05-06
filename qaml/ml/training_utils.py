"""Training utilities for machine learning models.

This module provides common training and evaluation functions used across
different training scripts.
"""

import random

import numpy as np
import torch

def set_seed(seed: int = 1) -> None:
    """Set random seeds for reproducibility.
    
    Sets seeds for Python's random module, NumPy, PyTorch (CPU and CUDA),
    and configures PyTorch for deterministic behavior.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model to train.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        data_loader: DataLoader providing training batches.
        device: Device to run training on (CPU/CUDA/MPS).
        
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(data_loader)
    
    for batch in data_loader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        preds, _ = model(inputs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(1, n_batches)

def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """Evaluate model on a dataset.
    
    Args:
        model: PyTorch model to evaluate.
        criterion: Loss function.
        data_loader: DataLoader providing evaluation batches.
        device: Device to run evaluation on (CPU/CUDA/MPS).
        
    Returns:
        Average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    n_batches = len(data_loader)
    
    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            
            preds, _ = model(inputs)
            loss = criterion(preds, labels)
            total_loss += loss.item()
    
    return total_loss / max(1, n_batches)
