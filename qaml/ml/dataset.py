import os
from typing import List, Optional, Dict, Any
import dill as pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_jz_value(filename: str, data_dict: Optional[Dict] = None) -> float:
    # Try to get from data dict first (primary method)
    if data_dict is not None:
        try:
            return float(data_dict["jz"])
        except (KeyError, ValueError, TypeError):
            pass

    # Fallback to parsing from filename
    try:
        return float(os.path.basename(filename).split("jz_")[1].split(".pkl")[0])
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract jz value from filename '{filename}' or data dict")


def kfold_split_indices(n: int, k: int = 5, seed: int = 1) -> List[List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    base, rem = n // k, n % k
    sizes = [(base + 1 if i < rem else base) for i in range(k)]
    folds, start = [], 0
    for sz in sizes:
        folds.append(perm[start:start+sz])
        start += sz
    return folds

class Heisenberg1DDataset(Dataset):
    """1D Heisenberg Dataset."""

    def __init__(self, root_dir: str, n_inputs: int, mode: str, jz_list: List[float] = None, observable:str = "nn_corr_funcs"):
        """
        Args:
            root_dir (str): Relative path to the data directory.
            n_inputs (int): Number of model inputs which is equal to number of edges in graph.
            mode (str): One of "train", "validation", and "test".
            jz_list (List[float], optional): List of ZZ copulings in XXZ 1D model. Defaults to None.
            observable (str, optional): Observable for training. Defaults to "nn_corr_funcs".
        """
        self._root_dir = os.path.join(root_dir,  mode)
        self._num_inputs = n_inputs
        if jz_list is None:
            self._files = [file for file in os.listdir(self._root_dir) if file[-4:] == '.pkl']
        else:
            self._files = [f"XXZ_1d_jz_{jz:.1f}.pkl" for jz in jz_list]
        self._files = sorted(self._files, key= lambda item: float(item.split("_")[-1][:-4]))
        self._observable = observable

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        with open(os.path.join(self._root_dir, self._files[idx]), "rb") as f:
            data = pickle.load(f)
        data_subset = {"inputs": np.array([data["jz"]] * self._num_inputs), "labels": data[self._observable]}
        return data_subset


class HeisenbergDataset(Dataset):
    def __init__(self, root_dir, n_inputs, observable_key, jz_min=1.6, files=None, preload=True, device=None):
        self.root_dir = root_dir
        self.n_inputs = n_inputs
        self.observable_key = observable_key
        self.jz_min = jz_min
        self.preload = preload

        all_files = files if files is not None else [
            f for f in os.listdir(root_dir) if f.endswith(".pkl")
        ]

        kept = []
        self.jz_values = {}  # store jz per file

        for name in all_files:
            try:
                # Try to load file and extract jz (primary method: from dict, fallback: from filename)
                with open(os.path.join(root_dir, name), "rb") as f:
                    d = pickle.load(f)
                jz_val = extract_jz_value(name, d)
                if jz_val >= jz_min:
                    kept.append(name)
                    self.jz_values[name] = jz_val
            except (ValueError, IOError):
                # Skip files that can't be loaded or don't have valid jz
                continue
        if len(kept) == 0:
            raise RuntimeError("No files satisfy jz >= jz_min")

        with open(os.path.join(root_dir, kept[0]), "rb") as f:
            d0 = pickle.load(f)
        try:
            y0 = np.asarray(d0[observable_key], dtype=np.float32).reshape(-1)
        except KeyError:
            raise RuntimeError(f"No observable data found, options are {d0.keys()}")
        n_out = int(y0.shape[0])

        self.files = kept
        self._n_outputs = n_out

        if preload:
            xs, ys = [], []
            for name in kept:
                with open(os.path.join(root_dir, name), "rb") as f:
                    d = pickle.load(f)

                jz_val = self.jz_values[name]

                x = np.full((n_inputs,), jz_val, dtype=np.float32)
                y = np.asarray(d[observable_key], dtype=np.float32).reshape(-1)
                if y.shape[0] != n_out:
                    raise ValueError(f"Inconsistent observable length: {name}")
                xs.append(torch.from_numpy(x))
                ys.append(torch.from_numpy(y))
            self._X = torch.stack(xs, dim=0)
            self._Y = torch.stack(ys, dim=0)
            if device is not None:
                self._X = self._X.to(device)
                self._Y = self._Y.to(device)
        else:
            self._X = None
            self._Y = None

    @property
    def n_outputs(self): return self._n_outputs

    def filenames(self): return list(self.files)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        if self.preload:
            return {"inputs": self._X[idx], "labels": self._Y[idx]}
        name = self.files[idx]
        with open(os.path.join(self.root_dir, name), "rb") as f:
            d = pickle.load(f)

        jz_val = self.jz_values[name]

        x = np.full((self.n_inputs,), jz_val, dtype=np.float32)
        y = np.asarray(d[self.observable_key], dtype=np.float32).reshape(-1)
        return {"inputs": torch.from_numpy(x), "labels": torch.from_numpy(y)}