import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os


def plot_bar(x: List[int], y: List[int], xlabel: str, ylabel: str, save_path: str, legend: str = None):
    plt.bar(x, y, width=1, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_line(x: List[float], y: List[float], xlabel: str, ylabel: str, save_path: str, legend: str = None):
    plt.plot(x, y, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_multiple_lines(
    x_list: List[List[float]],
    y_list: List[List[float]],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    save_path: str,
    legend_kwargs: Dict[str, Any] = None,
):
    for x, y, label in zip(x_list, y_list, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(**(legend_kwargs or {}))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_multiple_bars(
    x_list: List[List[int]],
    y_list: List[List[int]],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    save_path: str,
    legend_kwargs: Dict[str, Any] = None,
):
    for x, y, label in zip(x_list, y_list, labels):
        plt.bar(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(**(legend_kwargs or {}))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_heatmap(data: np.ndarray, xlabel: str, ylabel: str, save_path: str,  cmap='viridis', annot=False, fmt='.2f', cbar=True, cbar_label=None):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color="white") # show NaNs as white
    heatmap = sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt, square=True, cbar=cbar, ax=ax)
    if cbar and cbar_label:
        heatmap.collections[0].colorbar.set_label(cbar_label, rotation=270, labelpad=15)
                
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()
    
def plot_nn_correlation(
    properties_dict: Dict[str, Dict[float, float | np.ndarray]],
    jz_list: List[float],
    save_path: str,
    diff: bool = False,
):
    fig, axes = plt.subplots(1, len(jz_list), figsize=(5 * len(jz_list), 3))
    for i, jz in enumerate(jz_list):
        axes[i].set_xlabel("NN index (i)")
        if i == 0:
            axes[i].set_ylabel(
                r"$\langle C_{(i,j)} \rangle$"
                if not diff
                else r"$\Delta \langle C_{(i,j)} \rangle$"
            )
        skqd = properties_dict["full"]["nn_corr_funcs"][jz]
        truncated = properties_dict["truncated"]["nn_corr_funcs"][jz]
        if diff:
            axes[i].plot(np.array(truncated) - np.array(skqd))
        else:
            axes[i].plot(skqd, label="SKQD")
            axes[i].plot(truncated, label="Truncated")
            axes[i].legend()
        axes[i].set_title(rf"$J_z = {jz:.1f}$")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def plot_gs_amplitudes(
    gs_dict_truncated_dict: Dict[float, Dict[str, complex]],
    jz_list: List[float],
    save_dir: str,
):
    """
    For all selected Jz, plot sorted real, imaginary, and |value|^2 amplitudes in three separate graphs.
    Each line is a Jz, X axis is "Rank".
    """
    real_lines = []
    imag_lines = []
    abs2_lines = []
    labels = []

    for jz in jz_list:
        jz_key = float(np.round(jz, 1))
        if jz_key not in gs_dict_truncated_dict:
            continue
        state = gs_dict_truncated_dict[jz_key]
        amplitudes = np.array(list(state.values()))
        abs2 = np.abs(amplitudes) ** 2
        sort_idx = np.argsort(-abs2)  # Descending order
        real_lines.append(amplitudes.real[sort_idx])
        imag_lines.append(amplitudes.imag[sort_idx])
        abs2_lines.append(abs2[sort_idx])
        labels.append(rf"$J_z={jz_key}$")

    x_list = [np.arange(len(line)) for line in real_lines]

    # Real part
    plot_multiple_lines(
        x_list,
        real_lines,
        labels,
        xlabel="Rank",
        ylabel=r"$\mathcal{Re}(\psi)$",
        save_path=os.path.join(save_dir, "SKQD_real_amp_rank.pdf"),
        legend_kwargs={"fontsize": 8},
    )

    # Imaginary part
    plot_multiple_lines(
        x_list,
        imag_lines,
        labels,
        xlabel="Rank",
        ylabel=r"$\mathcal{Im}(\psi)$",
        save_path=os.path.join(save_dir, "SKQD_img_amp_rank.pdf"),
        legend_kwargs={"fontsize": 8},
    )

    # Absolute value squared
    plot_multiple_lines(
        x_list,
        abs2_lines,
        labels,
        xlabel="Rank",
        ylabel=r"$|\psi|^2$",
        save_path=os.path.join(save_dir, "SKQD_prob_rank.pdf"),
        legend_kwargs={"fontsize": 8},
    )
