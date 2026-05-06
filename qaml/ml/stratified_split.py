"""Stratified data splitting utilities for machine learning experiments.

This module provides functions for creating stratified train/validation/test splits
based on Jz parameter regimes, ensuring balanced representation across different
parameter ranges.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Standard Jz regimes used for stratification across all experiments
JZ_REGIMES = [(1.6, 2.0), (2.1, 4.0), (4.1, 6.0)]

def stratified_split_by_jz(
    in_files: List[str],
    jz_values: Dict[str, float],
    test_frac: float,
    val_frac: float,
    rng: np.random.Generator
) -> Tuple[List[int], List[int], List[int]]:
    """Perform stratified split by Jz regime for train/val/test sets.
    
    Splits files into train, validation, and test sets while maintaining
    proportional representation from each Jz regime.
    
    Args:
        in_files: List of file names to split.
        jz_values: Dictionary mapping file names to their Jz values.
        test_frac: Fraction of data to use for test set.
        val_frac: Fraction of remaining data (after test) to use for validation.
        rng: NumPy random number generator for reproducibility.
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices) into in_files.
    """
    regime_indices = [[] for _ in JZ_REGIMES]
    
    # Assign each file to a regime
    for idx, fname in enumerate(in_files):
        jz = jz_values[fname]
        for i, (jz_min, jz_max) in enumerate(JZ_REGIMES):
            if jz_min <= jz <= jz_max:
                regime_indices[i].append(idx)
                break
    
    train_idx, val_idx, test_idx = [], [], []
    
    for regime_idx in regime_indices:
        if len(regime_idx) == 0:
            continue
        
        regime_idx = np.array(regime_idx)
        rng.shuffle(regime_idx)
        
        # Split off test set (only if test_frac > 0)
        if test_frac > 0:
            n_test_r = max(1, int(round(test_frac * len(regime_idx))))
            test_idx.extend(regime_idx[:n_test_r])
            trainval_idx = regime_idx[n_test_r:]
        else:
            trainval_idx = regime_idx
        
        # Split remaining into train and validation
        n_val_r = max(1, int(np.ceil(val_frac * len(trainval_idx)))) if val_frac > 0 else 0
        val_idx.extend(trainval_idx[:n_val_r] if n_val_r > 0 else [])
        train_idx.extend(trainval_idx[n_val_r:])
    
    return train_idx, val_idx, test_idx

def stratified_kfold_split(
    in_files: List[str],
    jz_values: Dict[str, float],
    k: int = 5,
    seed: Optional[int] = None
) -> List[List[int]]:
    """Create stratified k-fold splits based on Jz regimes.
    
    Creates k folds where each fold has proportional representation from
    each Jz regime.
    
    Args:
        in_files: List of file names to split.
        jz_values: Dictionary mapping file names to their Jz values.
        k: Number of folds to create.
        seed: Random seed for reproducibility.
        
    Returns:
        List of k folds, where each fold contains indices into in_files.
    """
    rng = np.random.default_rng(seed)
    
    regime_indices = [[] for _ in JZ_REGIMES]
    
    # Assign each file to a regime
    for idx, fname in enumerate(in_files):
        jz = jz_values[fname]
        for i, (jz_min, jz_max) in enumerate(JZ_REGIMES):
            if jz_min <= jz <= jz_max:
                regime_indices[i].append(idx)
                break
    
    # Create k folds, ensuring each fold has representation from each regime
    folds = [[] for _ in range(k)]
    
    for regime_idx in regime_indices:
        if len(regime_idx) == 0:
            continue
        
        # Shuffle indices within this regime
        regime_idx = np.array(regime_idx)
        rng.shuffle(regime_idx)
        
        # Distribute regime samples across k folds as evenly as possible
        for i, idx in enumerate(regime_idx):
            fold_num = i % k
            folds[fold_num].append(idx)
    
    return folds

def make_stratified_test_split(
    in_files: List[str],
    test_frac: float = 0.2,
    seed: Optional[int] = None,
    jz_values: Optional[Dict[str, float]] = None
) -> Tuple[List[str], List[str]]:
    """Create a stratified train/test split.
    
    If jz_values is provided, performs stratified split by Jz regime.
    Otherwise, performs simple random split.
    
    Args:
        in_files: List of file names to split.
        test_frac: Fraction of data to use for test set.
        seed: Random seed for reproducibility.
        jz_values: Optional dictionary mapping file names to their Jz values.
        
    Returns:
        Tuple of (train_files, test_files).
    """
    rng = np.random.default_rng(seed)
    
    if jz_values is None:
        # Simple random split if no jz values provided
        n = len(in_files)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * n)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return [in_files[i] for i in train_idx], [in_files[i] for i in test_idx]
    
    # Stratified split by jz regime
    regime_indices = [[] for _ in JZ_REGIMES]
    
    for idx, fname in enumerate(in_files):
        jz = jz_values[fname]
        for i, (jz_min, jz_max) in enumerate(JZ_REGIMES):
            if jz_min <= jz <= jz_max:
                regime_indices[i].append(idx)
                break
    
    train_idx, test_idx = [], []
    for regime_idx in regime_indices:
        if len(regime_idx) == 0:
            continue
        regime_idx = np.array(regime_idx)
        rng.shuffle(regime_idx)
        n_test_r = max(1, int(round(test_frac * len(regime_idx))))
        test_idx.extend(regime_idx[:n_test_r])
        train_idx.extend(regime_idx[n_test_r:])
    
    return [in_files[i] for i in train_idx], [in_files[i] for i in test_idx]

def print_split_info(split: Dict, jz_values: Dict[str, float]) -> None:
    """Print dataset split information with Jz ranges.
    
    Args:
        split: Dictionary containing 'train_files', 'val_files', and 'test_files'.
        jz_values: Dictionary mapping file names to their Jz values.
    """
    train_jz = sorted([jz_values[f] for f in split["train_files"]])
    test_jz = sorted([jz_values[f] for f in split["test_files"]]) if split["test_files"] else []
    val_jz = sorted([jz_values[f] for f in split["val_files"]]) if split["val_files"] else []
    
    print(f"\nDataset split:")
    print(f"  Train: {len(split['train_files'])} files, jz range: [{train_jz[0]:.2f}, {train_jz[-1]:.2f}]")
    if val_jz:
        print(f"  Val: {len(split['val_files'])} files, jz range: [{val_jz[0]:.2f}, {val_jz[-1]:.2f}]")
    if test_jz:
        print(f"  Test: {len(split['test_files'])} files, jz range: [{test_jz[0]:.2f}, {test_jz[-1]:.2f}]")
