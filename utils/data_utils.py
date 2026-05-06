import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import dill as pickle


def resolve_data_root(data_root: str, script_file: str) -> str:
    """
    Resolve data_root path to absolute path.
    
    If data_root is relative, it's assumed to be relative to the workspace root.
    This handles the case where models were trained with paths like ../data/...
    from ml_training/ directory.
    
    Args:
        data_root: Path from model settings (may be relative)
        script_file: __file__ from the calling script
        
    Returns:
        Absolute path to data directory
    """
    if os.path.isabs(data_root):
        return data_root
    
    # Get workspace root from script location
    # Most scripts are in figures/*/  or similar, so go up to workspace root
    script_path = Path(script_file).resolve()
    workspace_root = script_path.parent
    
    # Navigate up until we find the workspace root (contains qaml/ and data/)
    while workspace_root.parent != workspace_root:
        if (workspace_root / 'qaml').exists() and (workspace_root / 'data').exists():
            break
        workspace_root = workspace_root.parent
    
    # Strip leading ../ from data_root and join with workspace root
    clean_path = data_root.lstrip('../')
    resolved = workspace_root / clean_path
    
    return str(resolved)


def load_skqd_data(jz: float, data_root: str) -> Optional[Dict]:
    """
    Load SKQD data for a given Jz value.

    Args:
        jz: Jz value
        data_root: Root directory containing SKQD data files

    Returns:
        Dictionary with SKQD data, or None if file not found
    """
    skqd_file = f"XXZ_2d_jz_{jz:.1f}.pkl"
    skqd_path = os.path.join(data_root, skqd_file)

    if not os.path.exists(skqd_path):
        print(f"Warning: SKQD data not found for jz={jz} at {skqd_path}")
        return None

    with open(skqd_path, "rb") as f:
        return pickle.load(f)


def load_dmrg_data(jz: float, dmrg_root: str) -> Optional[Dict]:
    """
    Load DMRG data for a given Jz value.

    Args:
        jz: Jz value
        dmrg_root: Root directory containing DMRG data files

    Returns:
        Dictionary with DMRG data, or None if file not found
    """
    dmrg_file = f"XXZ_2d_jz_{jz:.1f}.pkl"
    dmrg_path = os.path.join(dmrg_root, dmrg_file)

    if not os.path.exists(dmrg_path):
        print(f"Warning: DMRG data not found for jz={jz} at {dmrg_path}")
        return None

    with open(dmrg_path, "rb") as f:
        return pickle.load(f)


def collect_predictions_by_jz(all_predictions: Dict[str, Dict]) -> Tuple[Dict, Dict, Dict]:
    """
    Collect predictions from all splits organized by Jz value.

    Args:
        all_predictions: Dictionary mapping split names to predictions

    Returns:
        Tuple of (predictions_by_jz, labels_by_jz, split_info)
        where split_info maps jz -> split_name
    """
    predictions_by_jz = {}
    labels_by_jz = {}
    split_info = {}

    for split_name, predictions in all_predictions.items():
        for jz in predictions.keys():
            try:
                predictions_by_jz[jz] = predictions[jz]["preds"]
            except KeyError:
                predictions_by_jz[jz] = predictions[jz]["predictions"]
            labels_by_jz[jz] = predictions[jz]["labels"]
            split_info[jz] = split_name

    return predictions_by_jz, labels_by_jz, split_info
