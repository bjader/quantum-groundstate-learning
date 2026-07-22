import os
import pickle
import traceback
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from networkx.drawing.nx_pydot import graphviz_layout

from qaml.analysis.analysis_utils import filter_to_less_neighbours
from qaml.diagonalisation.twod.dmrg import inverse_order_single_site_obs, inverse_order_two_site_list, \
    DMRGResult
from qaml.graph.graph_utils import read_edges_txt
from qaml.observables.single_site import szs_from_state_dict
from qaml.observables.correlation_function import all_k_cycles_undirected
from utils.data_utils import load_skqd_data, load_dmrg_data, collect_predictions_by_jz
from qaml.visualisation.heavy_hex import plot_corr_heavy_hex

# Constants for split-based plotting
SPLIT_COLORS = {"train": "blue", "val": "green", "test": "red"}
SPLIT_MARKERS = {"train": "o", "val": "s", "test": "^"}

def plot_loops_heavy_hex(
    edges, n, loops, loop_vals, cmap=plt.cm.Blues,
    vmin=None, vmax=None, alpha=0.60, title=None,
    cbar_label=r"$\langle Z^{\otimes 12}\rangle$",
    show_labels=True, scale="linear", linthresh=1e-3, base=10
):
    """
    Plot loop observables on heavy-hex lattice.
    
    Args:
        edges: List of edges defining the graph
        n: Number of nodes in the graph
        loops: List of loops (each loop is a list of node indices)
        loop_vals: Values for each loop
        cmap: Colormap for visualization
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        alpha: Transparency of loop polygons
        title: Plot title
        cbar_label: Label for colorbar
        show_labels: Whether to show node labels
        scale: Color scale type ("linear" or "symlog")
        linthresh: Linear threshold for symlog scale
        base: Base for symlog scale
        
    Returns:
        matplotlib figure object
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    try:
        pos = graphviz_layout(G)
    except Exception:
        pos = nx.spring_layout(G)

    fig, ax = plt.subplots(constrained_layout=False)

    vals = np.asarray(loop_vals, dtype=float)
    maxabs = float(np.nanmax(np.abs(vals))) if np.any(~np.isnan(vals)) else 1.0
    maxabs = max(maxabs, 1e-12)

    if vmin is None:
        vmin = -maxabs
    if vmax is None:
        vmax = +maxabs

    if scale == "symlog":
        norm = mpl.colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=base)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for loop, val in zip(loops, vals):
        if np.isnan(val):
            continue
        pts = np.array([pos[int(v)] for v in loop], dtype=float)
        poly = Polygon(pts, closed=True, facecolor=sm.to_rgba(val),
                       edgecolor="none", alpha=alpha)
        poly.set_zorder(1)
        ax.add_patch(poly)

    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.35, width=1.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=70, node_color="k", ax=ax)

    if show_labels:
        nx.draw_networkx_labels(G, pos, labels={i: i for i in range(n)},
                                font_size=5, ax=ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label)

    if title:
        ax.set_title(title)

    ax.set_axis_off()
    return fig

def plot_loop_observable_vs_skqd(
    all_predictions: Dict[str, Dict],
    n_spins: int,
    data_root: str,
    save_dir: str,
    special_predictions: Optional[Dict] = None,
    region_name: str = "Test Set",
    filename_prefix: str = "",
    split_config: Optional[Dict] = None
):
    """
    Plot loop observable predictions vs SKQD data on heavy-hex lattice.
    
    Args:
        all_predictions: Dictionary of predictions by split
        n_spins: Number of spins in the system
        data_root: Root directory for SKQD data
        save_dir: Directory to save plots
        special_predictions: Predictions for special region (e.g., test or boundary)
        region_name: Name of the special region for plot titles (e.g., "Test Set", "Boundary Region")
        filename_prefix: Prefix for output filenames (e.g., "", "boundary_")
        split_config: Optional custom split configuration for plotting
    """

    utils_dir = Path(__file__).parent
    cmaps_dir = utils_dir.parent.parent / "cmaps"

    if n_spins == 115:
        edges_path = str(cmaps_dir / "heavy_hex_edges_d_7.txt")
    elif n_spins == 57:
        edges_path = str(cmaps_dir / "heavy_hex_edges_d_5.txt")
    else:
        raise ValueError(f"Unsupported number of spins: {n_spins}")

    edges, _ = read_edges_txt(edges_path)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Collect predictions using utility function
    corrs_ml_per_jz, _, jz_to_split = collect_predictions_by_jz(all_predictions)

    # Load SKQD data and loop sites for each Jz
    corrs_skqd_per_jz = {}
    loops_per_jz = {}
    
    global_loops = all_k_cycles_undirected(edges, k=12)
    for jz in corrs_ml_per_jz.keys():
        data_skqd = load_skqd_data(jz, data_root)
        if data_skqd is None:
            continue

        try:
            print(data_skqd.keys())
            skqd_loop_vals = np.array(data_skqd["z_loop"])
            corrs_skqd_per_jz[jz] = skqd_loop_vals
            loops_per_jz[jz] = global_loops
        except KeyError as e:
            print(f"Warning: Required data not found for jz={jz}: {e}")
            continue

    if len(corrs_skqd_per_jz) == 0:
        print("No valid loop data found")
        return

    # Filter to only Jz values with SKQD data
    jz_list = sorted(corrs_skqd_per_jz.keys())
    corrs_ml_per_jz = {jz: corrs_ml_per_jz[jz] for jz in jz_list}
    jz_to_split = {jz: jz_to_split[jz] for jz in jz_list}

    # Determine global color scale
    all_vals = np.concatenate([corrs_skqd_per_jz[jz] for jz in jz_list])
    global_vlim = float(np.nanpercentile(np.abs(all_vals), 99))
    global_vlim = max(global_vlim, 1e-12)

    # Plot loop visualizations for special region if provided
    if special_predictions is not None:
        special_jz_list = sorted(special_predictions.keys())
        for jz in special_jz_list:
            if jz not in loops_per_jz:
                continue

            loops = loops_per_jz[jz]

            # ML predictions
            fig = plot_loops_heavy_hex(
                edges=edges,
                n=n_spins,
                loops=loops,
                loop_vals=corrs_ml_per_jz[jz],
                vmin=0,
                vmax=global_vlim,
                title=f"ML Predictions (Jz={jz:.1f}, {region_name})",
                cbar_label=r"$\langle Z^{\otimes 12}\rangle$"
            )
            fig.savefig(os.path.join(save_dir, f"ml_loop_{filename_prefix}jz_{jz:.1f}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)

            # SKQD data
            fig = plot_loops_heavy_hex(
                edges=edges,
                n=n_spins,
                loops=loops,
                loop_vals=corrs_skqd_per_jz[jz],
                vmin=0,
                vmax=global_vlim,
                title=f"SKQD Data (Jz={jz:.1f}, {region_name})",
                cbar_label=r"$\langle Z^{\otimes 12}\rangle$"
            )
            fig.savefig(os.path.join(save_dir, f"skqd_loop_{filename_prefix}jz_{jz:.1f}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)

    # Plot correlations over Jz with split-based colors using shared function
    ml_mat = np.vstack([corrs_ml_per_jz[jz] for jz in jz_list])
    skqd_mat = np.vstack([corrs_skqd_per_jz[jz] for jz in jz_list])

    plot_loop_observables_over_jz(
        ml_mat=ml_mat,
        reference_mat=skqd_mat,
        jz_list=jz_list,
        split_info=jz_to_split,
        save_path=os.path.join(save_dir, "loop_corrs_over_jz.pdf"),
        ylabel=r"$\langle Z^{\otimes 12}\rangle$",
        reference_label="SKQD",
        split_config=split_config
    )

    # Plot relative errors using shared function
    plot_loop_relative_errors(
        ml_mat=ml_mat,
        reference_mat=skqd_mat,
        jz_list=jz_list,
        split_info=jz_to_split,
        save_path=os.path.join(save_dir, "relative_error_vs_jz.pdf"),
        split_config=split_config
    )

def plot_loop_observable_vs_dmrg(
    all_predictions: Dict[str, Dict],
    n_spins: int,
    data_root: str,
    dmrg_root: str,
    save_dir: str,
    special_predictions: Optional[Dict] = None,
    region_name: str = "Test Set",
    filename_prefix: str = "",
    split_config: Optional[Dict] = None
):
    """
    Plot loop observable predictions vs DMRG data on heavy-hex lattice.
    
    Args:
        all_predictions: Dictionary of predictions by split
        n_spins: Number of spins in the system
        data_root: Root directory for SKQD data (to get loop sites)
        dmrg_root: Root directory for DMRG data
        save_dir: Directory to save plots
        special_predictions: Predictions for special region (e.g., test or boundary)
        region_name: Name of the special region for plot titles (e.g., "Test Set", "Boundary Region")
        filename_prefix: Prefix for output filenames (e.g., "", "boundary_")
        split_config: Optional custom split configuration for plotting
    """

    utils_dir = Path(__file__).parent
    cmaps_dir = utils_dir.parent.parent / "cmaps"

    if n_spins == 115:
        edges_path = str(cmaps_dir / "heavy_hex_edges_d_7.txt")
    elif n_spins == 57:
        edges_path = str(cmaps_dir / "heavy_hex_edges_d_5.txt")
    else:
        raise ValueError(f"Unsupported number of spins: {n_spins}")

    edges, _ = read_edges_txt(edges_path)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Collect predictions using utility function
    corrs_ml_per_jz, _, jz_to_split = collect_predictions_by_jz(all_predictions)

    # Load DMRG data and loop sites for each Jz
    corrs_dmrg_per_jz = {}
    loops_per_jz = {}

    for jz in corrs_ml_per_jz.keys():
        # Load SKQD data to get loop sites
        data_skqd = load_skqd_data(jz, data_root)
        if data_skqd is None:
            continue

        try:
            loops = list(data_skqd["z_loop_sites"])
        except KeyError:
            print(f"Warning: No loop sites found for jz={jz}")
            continue

        # Load DMRG data
        data_dmrg = load_dmrg_data(jz, dmrg_root)
        if data_dmrg is None:
            continue

        try:
            dmrg_loop_vals = np.array(data_dmrg["z_loop"])
            corrs_dmrg_per_jz[jz] = dmrg_loop_vals
            loops_per_jz[jz] = loops
        except KeyError:
            print(f"Warning: No loop values found in DMRG data for jz={jz}")
            continue

    if len(corrs_dmrg_per_jz) == 0:
        print("No valid loop data found")
        return

    # Filter to only Jz values with DMRG data
    jz_list = sorted(corrs_dmrg_per_jz.keys())
    corrs_ml_per_jz = {jz: corrs_ml_per_jz[jz] for jz in jz_list}
    jz_to_split = {jz: jz_to_split[jz] for jz in jz_list}

    # Determine global color scale
    all_vals = np.concatenate([corrs_dmrg_per_jz[jz] for jz in jz_list])
    global_vlim = float(np.nanpercentile(np.abs(all_vals), 99))
    global_vlim = max(global_vlim, 1e-12)

    # Plot loop visualizations for special region if provided
    if special_predictions is not None:
        special_jz_list = sorted(special_predictions.keys())
        for jz in special_jz_list:
            if jz not in loops_per_jz:
                continue

            loops = loops_per_jz[jz]

            # ML predictions
            fig = plot_loops_heavy_hex(
                edges=edges,
                n=n_spins,
                loops=loops,
                loop_vals=corrs_ml_per_jz[jz],
                vmin=0,
                vmax=global_vlim,
                title=f"ML Predictions (Jz={jz:.1f}, {region_name})",
                cbar_label=r"$\langle Z^{\otimes 12}\rangle$"
            )
            fig.savefig(os.path.join(save_dir, f"ml_loop_{filename_prefix}jz_{jz:.1f}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)

            # DMRG data
            fig = plot_loops_heavy_hex(
                edges=edges,
                n=n_spins,
                loops=loops,
                loop_vals=corrs_dmrg_per_jz[jz],
                vmin=0,
                vmax=global_vlim,
                title=f"DMRG Data (Jz={jz:.1f}, {region_name})",
                cbar_label=r"$\langle Z^{\otimes 12}\rangle$"
            )
            fig.savefig(os.path.join(save_dir, f"dmrg_loop_{filename_prefix}jz_{jz:.1f}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)

            # Difference plot
            diff = corrs_ml_per_jz[jz] - corrs_dmrg_per_jz[jz]
            fig = plot_loops_heavy_hex(
                edges=edges,
                n=n_spins,
                loops=loops,
                loop_vals=diff,
                cmap=plt.cm.RdBu_r,
                title=f"ML - DMRG (Jz={jz:.1f}, {region_name})",
                cbar_label=r"$\Delta\langle Z^{\otimes 12}\rangle$"
            )
            fig.savefig(
                os.path.join(save_dir, f"diff_ml_minus_dmrg_{filename_prefix}jz_{jz:.1f}.pdf"),
                bbox_inches="tight")
            plt.close(fig)

    # Plot correlations over Jz with split-based colors using shared function
    ml_mat = np.vstack([corrs_ml_per_jz[jz] for jz in jz_list])
    dmrg_mat = np.vstack([corrs_dmrg_per_jz[jz] for jz in jz_list])

    plot_loop_observables_over_jz(
        ml_mat=ml_mat,
        reference_mat=dmrg_mat,
        jz_list=jz_list,
        split_info=jz_to_split,
        save_path=os.path.join(save_dir, "loop_corrs_over_jz_vs_dmrg.pdf"),
        ylabel=r"$\langle Z^{\otimes 12}\rangle$",
        reference_label="DMRG",
        split_config=split_config
    )

    # Plot relative errors using shared function
    plot_loop_relative_errors(
        ml_mat=ml_mat,
        reference_mat=dmrg_mat,
        jz_list=jz_list,
        split_info=jz_to_split,
        save_path=os.path.join(save_dir, "relative_error_vs_dmrg_jz.pdf"),
        split_config=split_config
    )

def compute_relative_error(predictions: np.ndarray, reference: np.ndarray,
                           min_denom: float = 1e-12, as_percentage: bool = True) -> np.ndarray:
    """
    Compute relative error between predictions and reference values.
    
    Args:
        predictions: Predicted values
        reference: Reference (ground truth) values
        min_denom: Minimum denominator to avoid division by zero
        as_percentage: If True, return as percentage (multiply by 100)
        
    Returns:
        Array of relative errors
    """
    denom = np.maximum(np.abs(reference), min_denom)
    rel_err = np.abs(predictions - reference) / denom

    if as_percentage:
        rel_err *= 100

    return rel_err

def plot_avg_relative_error_vs_jz(
    predictions_by_jz: Dict,
    reference_by_jz: Dict,
    split_info: Dict,
    save_path: str,
    ylabel: str = r"Average Relative Error (%)",
    title: str = "ML vs Reference",
    min_denom: float = 1e-12
):
    """
    Plot average relative error vs Jz with split-based coloring.
    
    Args:
        predictions_by_jz: Dictionary mapping jz -> predictions
        reference_by_jz: Dictionary mapping jz -> reference values
        split_info: Dictionary mapping jz -> split_name
        save_path: Path to save the plot
        ylabel: Y-axis label
        title: Plot title
        min_denom: Minimum denominator for relative error calculation
    """
    jz_list = sorted(predictions_by_jz.keys())

    plt.figure(figsize=(10, 6))

    for split_name in ["train", "val", "test"]:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        split_errors = []

        for jz in split_jzs:
            preds = predictions_by_jz[jz]
            refs = reference_by_jz[jz]
            rel_err = compute_relative_error(preds, refs, min_denom=min_denom, as_percentage=True)
            split_errors.append(float(np.nanmean(rel_err)))

        if len(split_jzs) > 0:
            plt.scatter(split_jzs, split_errors,
                        marker=SPLIT_MARKERS[split_name],
                        color=SPLIT_COLORS[split_name],
                        label=split_name.capitalize(),
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_values_with_reference_overlay(
    predictions_by_jz: Dict,
    reference_by_jz: Dict,
    split_info: Dict,
    save_path: str,
    ylabel: str = "Observable Value",
    title: str = "ML Predictions vs Reference",
    ylim: Optional[Tuple[float, float]] = None
):
    """
    Plot observable values over Jz with reference overlay.
    
    Args:
        predictions_by_jz: Dictionary mapping jz -> predictions
        reference_by_jz: Dictionary mapping jz -> reference values
        split_info: Dictionary mapping jz -> split_name
        save_path: Path to save the plot
        ylabel: Y-axis label
        title: Plot title
        ylim: Optional y-axis limits
    """
    jz_list = sorted(predictions_by_jz.keys())

    plt.figure(figsize=(10, 6))

    # Plot reference data (gray, smaller, background)
    for jz in jz_list:
        plt.scatter([jz] * len(reference_by_jz[jz]), reference_by_jz[jz],
                    marker='x', color='gray', s=15, alpha=0.4, edgecolors='none',
                    label='Reference' if jz == jz_list[0] else '')

    # Plot ML predictions (colored by split, foreground)
    for split_name in ["train", "val", "test"]:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        if len(split_jzs) > 0:
            for jz in split_jzs:
                plt.scatter([jz] * len(predictions_by_jz[jz]), predictions_by_jz[jz],
                            marker=SPLIT_MARKERS[split_name],
                            color=SPLIT_COLORS[split_name],
                            s=20, alpha=0.5, edgecolors='none')

    # Add legend
    for split_name in ["train", "val", "test"]:
        plt.scatter([], [], marker=SPLIT_MARKERS[split_name],
                    color=SPLIT_COLORS[split_name],
                    label=f'ML {split_name.capitalize()}', s=100, alpha=0.7,
                    edgecolors='black', linewidths=0.5)

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title)
    plt.legend(fontsize=10, loc='best')
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_two_panel_observable(
    predictions_by_jz: Dict,
    reference_by_jz: Dict,
    split_info: Dict,
    save_path: str,
    top_ylabel: str = "Observable Value",
    bottom_ylabel: str = r"Relative Error (%)",
    title: str = "ML vs Reference",
    compute_aggregate_fn=None,
    min_denom: float = 1e-12
):
    """
    Create a two-panel plot with observable values (top) and relative error (bottom).
    
    Args:
        predictions_by_jz: Dictionary mapping jz -> predictions
        reference_by_jz: Dictionary mapping jz -> reference values
        split_info: Dictionary mapping jz -> split_name
        save_path: Path to save the plot
        top_ylabel: Y-axis label for top panel
        bottom_ylabel: Y-axis label for bottom panel
        title: Plot title
        compute_aggregate_fn: Optional function to compute aggregate value from array
                            (e.g., mean, staggered magnetization). If None, uses mean.
        min_denom: Minimum denominator for relative error calculation
    """
    if compute_aggregate_fn is None:
        compute_aggregate_fn = np.mean

    jz_list = sorted(predictions_by_jz.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: Aggregate observable values
    agg_ml = {jz: float(compute_aggregate_fn(predictions_by_jz[jz])) for jz in jz_list}
    agg_ref = {jz: float(compute_aggregate_fn(reference_by_jz[jz])) for jz in jz_list}

    # Plot reference
    ax1.scatter(jz_list, [agg_ref[jz] for jz in jz_list],
                marker='x', color='gray', s=150, alpha=0.7, linewidths=2,
                label='Reference', zorder=1)

    # Plot ML predictions by split
    for split_name in ["train", "val", "test"]:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        agg_vals = [agg_ml[jz] for jz in split_jzs]

        if len(split_jzs) > 0:
            ax1.scatter(split_jzs, agg_vals,
                        marker=SPLIT_MARKERS[split_name],
                        color=SPLIT_COLORS[split_name],
                        label=f'ML {split_name.capitalize()}',
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5, zorder=2)

    ax1.set_ylabel(top_ylabel, fontsize=14)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Relative error
    for split_name in ["train", "val", "test"]:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        rel_errors = []
        for jz in split_jzs:
            denom = agg_ref[jz] if agg_ref[jz] != 0 else np.nan
            rel_err = abs((agg_ml[jz] - agg_ref[jz]) / denom) * 100 if not np.isnan(
                denom) else np.nan
            rel_errors.append(rel_err)

        if len(split_jzs) > 0:
            ax2.scatter(split_jzs, rel_errors,
                        marker=SPLIT_MARKERS[split_name],
                        color=SPLIT_COLORS[split_name],
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    ax2.set_xlabel(r"$J_z$", fontsize=14)
    ax2.set_ylabel(bottom_ylabel, fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_loop_observables_over_jz(
    ml_mat: np.ndarray,
    reference_mat: np.ndarray,
    jz_list: list,
    split_info: Dict,
    save_path: str,
    ylabel: str = r"$\langle Z^{\otimes 12}\rangle$",
    reference_label: str = "SKQD",
    split_config: Optional[Dict[str, Dict]] = None
):
    """
    Plot loop observable values over Jz with split-based colors.
    
    Args:
        ml_mat: Matrix of ML predictions (n_jz x n_loops)
        reference_mat: Matrix of reference values (n_jz x n_loops)
        jz_list: List of Jz values
        split_info: Dictionary mapping jz -> split_name
        save_path: Path to save the plot
        ylabel: Y-axis label
        reference_label: Label for reference data (e.g., "SKQD" or "DMRG")
        split_config: Optional custom split configuration
    """
    # Use default or custom split configuration
    if split_config is None:
        split_colors = SPLIT_COLORS
        split_markers = SPLIT_MARKERS
        split_labels = {"train": "Train", "val": "Val", "test": "Test"}
        split_order = ["train", "val", "test"]
    else:
        split_colors = split_config.get("colors", SPLIT_COLORS)
        split_markers = split_config.get("markers", SPLIT_MARKERS)
        split_labels = split_config.get("labels", {})
        split_order = list(split_colors.keys())

    n_loops = ml_mat.shape[1]

    plt.figure(figsize=(10, 6))

    # Plot by split to get different colors
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        split_indices = [jz_list.index(jz) for jz in split_jzs]

        if len(split_indices) > 0:
            # Plot ML predictions for this split (filled markers)
            for ell in range(n_loops):
                ml_vals = [ml_mat[idx, ell] for idx in split_indices]
                plt.plot(split_jzs, ml_vals,
                         color=split_colors[split_name],
                         marker='o', linestyle='none', markersize=5,
                         markerfacecolor=split_colors[split_name],
                         markeredgecolor=split_colors[split_name],
                         alpha=0.6)

            # Plot reference data for this split (hollow markers)
            for ell in range(n_loops):
                ref_vals = [reference_mat[idx, ell] for idx in split_indices]
                plt.plot(split_jzs, ref_vals,
                         color=split_colors[split_name],
                         marker='o', linestyle='none', markersize=6,
                         markerfacecolor='none',
                         markeredgecolor=split_colors[split_name],
                         markeredgewidth=1.5,
                         alpha=0.6)

    # Add legend
    for split_name in split_order:
        plt.plot([], [], color=split_colors[split_name], marker='o', linestyle='none',
                 markersize=7, markerfacecolor=split_colors[split_name],
                 markeredgecolor=split_colors[split_name],
                 label=f'{split_labels.get(split_name, split_name.capitalize())} ML', alpha=0.7)
        plt.plot([], [], color=split_colors[split_name], marker='o', linestyle='none',
                 markersize=7, markerfacecolor='none',
                 markeredgecolor=split_colors[split_name], markeredgewidth=1.5,
                 label=f'{split_labels.get(split_name, split_name.capitalize())} {reference_label}',
                 alpha=0.7)

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_loop_relative_errors(
    ml_mat: np.ndarray,
    reference_mat: np.ndarray,
    jz_list: list,
    split_info: Dict,
    save_path: str,
    split_config: Optional[Dict[str, Dict]] = None,
    min_denom: float = 1e-12
):
    """
    Plot relative errors for loop observables with split-based colors.
    
    Args:
        ml_mat: Matrix of ML predictions (n_jz x n_loops)
        reference_mat: Matrix of reference values (n_jz x n_loops)
        jz_list: List of Jz values
        split_info: Dictionary mapping jz -> split_name
        save_path: Path to save the plot
        split_config: Optional custom split configuration
        min_denom: Minimum denominator for relative error calculation
    """
    # Use default or custom split configuration
    if split_config is None:
        split_colors = SPLIT_COLORS
        split_markers = SPLIT_MARKERS
        split_labels = {"train": "Train", "val": "Val", "test": "Test"}
        split_order = ["train", "val", "test"]
    else:
        split_colors = split_config.get("colors", SPLIT_COLORS)
        split_markers = split_config.get("markers", SPLIT_MARKERS)
        split_labels = split_config.get("labels", {})
        split_order = list(split_colors.keys())

    n_loops = ml_mat.shape[1]

    # Compute relative errors
    denom = np.maximum(np.abs(reference_mat), min_denom)
    rel_err_mat = np.abs(ml_mat - reference_mat) / denom * 100  # Convert to percentage

    plt.figure(figsize=(10, 6))

    # Plot individual loop errors by split
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        split_indices = [jz_list.index(jz) for jz in split_jzs]

        if len(split_indices) > 0:
            for ell in range(n_loops):
                err_vals = [rel_err_mat[idx, ell] for idx in split_indices]
                plt.plot(split_jzs, err_vals, 'x', markersize=5,
                         color=split_colors[split_name], alpha=0.5)

    # Plot mean error with error bars by split
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        split_indices = [jz_list.index(jz) for jz in split_jzs]

        if len(split_indices) > 0:
            mean_errs = [np.mean(rel_err_mat[idx, :]) for idx in split_indices]
            stderr_vals = [
                np.std(rel_err_mat[idx, :], ddof=1) / np.sqrt(n_loops) if n_loops > 1 else 0
                for idx in split_indices]

            plt.errorbar(split_jzs, mean_errs, yerr=stderr_vals,
                         fmt='o', color=split_colors[split_name],
                         markersize=8, capsize=5, linewidth=2,
                         label=f'{split_labels.get(split_name, split_name.capitalize())} Mean ± SE',
                         alpha=0.7)

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel("Relative Error (%)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_z_observable_vs_skqd_with_splits(
    all_predictions: Dict[str, Dict],
    save_dir: str,
    split_config: Optional[Dict[str, Dict]] = None,
    training_cutoff: Optional[float] = None
):
    """
    Plot Z observable predictions vs SKQD data for all splits combined.
    Includes: relative error, Z values with SKQD overlay, and staggered magnetization.
    
    Args:
        all_predictions: Dictionary mapping split names to predictions
        save_dir: Directory to save plots
        split_config: Optional custom split configuration with keys:
            - colors: Dict mapping split names to colors
            - markers: Dict mapping split names to markers
            - labels: Dict mapping split names to display labels
            If None, uses default train/val/test configuration
        training_cutoff: Optional Jz value to mark training cutoff with vertical line
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Use default or custom split configuration
    if split_config is None:
        split_colors = SPLIT_COLORS
        split_markers = SPLIT_MARKERS
        split_labels = {"train": "Train", "val": "Val", "test": "Test"}
        split_order = ["train", "val", "test"]
    else:
        split_colors = split_config.get("colors", SPLIT_COLORS)
        split_markers = split_config.get("markers", SPLIT_MARKERS)
        split_labels = split_config.get("labels", {})
        split_order = list(split_colors.keys())

    # Collect all data from all splits
    z_vals_ml, z_vals_skqd, split_info = collect_predictions_by_jz(all_predictions)
    jz_list = sorted(z_vals_ml.keys())

    # 1. Plot average relative error vs Jz
    plt.figure(figsize=(10, 6))
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        split_errors = []
        for jz in split_jzs:
            preds = z_vals_ml[jz]
            labels = z_vals_skqd[jz]
            rel_err = compute_relative_error(preds, labels, as_percentage=True)
            split_errors.append(float(np.nanmean(rel_err)))

        if len(split_jzs) > 0:
            plt.scatter(split_jzs, split_errors,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        label=split_labels.get(split_name, split_name.capitalize()),
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        plt.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel(r"$\Delta Z$ (%)", fontsize=14)
    plt.title("ML vs SKQD: Z Observable (All Splits)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_rel_error_vs_jz.pdf"))
    plt.close()

    # 2. Plot Z values over Jz with SKQD reference overlay
    plt.figure(figsize=(10, 6))

    # Plot SKQD reference data (gray, smaller, background)
    for jz in jz_list:
        plt.scatter([jz] * len(z_vals_skqd[jz]), z_vals_skqd[jz],
                    marker='x', color='gray', s=15, alpha=0.4, edgecolors='none',
                    label='SKQD' if jz == jz_list[0] else '')

    # Plot ML predictions (colored by split, foreground)
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        if len(split_jzs) > 0:
            for jz in split_jzs:
                plt.scatter([jz] * len(z_vals_ml[jz]), z_vals_ml[jz],
                            marker=split_markers[split_name],
                            color=split_colors[split_name],
                            s=20, alpha=0.5, edgecolors='none')

    # Add legend
    for split_name in split_order:
        plt.scatter([], [], marker=split_markers[split_name],
                    color=split_colors[split_name],
                    label=f'ML {split_labels.get(split_name, split_name.capitalize())}',
                    s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        plt.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel(r"$\langle Z \rangle$", fontsize=14)
    plt.title("Z Observable: ML Predictions vs SKQD Reference", fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "z_over_jz.pdf"))
    plt.close()

    # 3. Staggered magnetization with SKQD overlay and relative error subplot
    def compute_staggered_magnetization(z_vals):
        """Compute staggered magnetization from Z values."""
        eta = np.sign(z_vals)
        eta = np.where(eta == 0, 1.0, eta)
        return np.mean(eta * z_vals)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top: Staggered magnetization values
    ms_ml_all = {jz: float(compute_staggered_magnetization(z_vals_ml[jz])) for jz in jz_list}
    ms_skqd_all = {jz: float(compute_staggered_magnetization(z_vals_skqd[jz])) for jz in jz_list}

    # Plot SKQD reference
    ax1.scatter(jz_list, [ms_skqd_all[jz] for jz in jz_list],
                marker='x', color='gray', s=150, alpha=0.7, linewidths=2,
                label='SKQD Reference', zorder=1)

    # Plot ML predictions by split
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        ms_vals = [ms_ml_all[jz] for jz in split_jzs]

        if len(split_jzs) > 0:
            ax1.scatter(split_jzs, ms_vals,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        label=f'ML {split_labels.get(split_name, split_name.capitalize())}',
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5, zorder=2)

    if training_cutoff is not None:
        ax1.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    ax1.set_ylabel(r"$m_s = \frac{1}{N}\sum_i \eta_i\langle Z_i\rangle$", fontsize=14)
    ax1.set_title("Staggered Magnetization: ML vs SKQD", fontsize=14)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Bottom: Relative error
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        rel_errors = []
        for jz in split_jzs:
            denom = ms_skqd_all[jz] if ms_skqd_all[jz] != 0 else np.nan
            rel_err = abs((ms_ml_all[jz] - ms_skqd_all[jz]) / denom) * 100 if not np.isnan(
                denom) else np.nan
            rel_errors.append(rel_err)

        if len(split_jzs) > 0:
            ax2.scatter(split_jzs, rel_errors,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        ax2.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel(r"$J_z$", fontsize=14)
    ax2.set_ylabel(r"$\Delta m_s$ (%)", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "staggered_magnetization_vs_jz.pdf"))
    plt.close()
    print(f"Saved Z vs SKQD plots to {save_dir}")

def plot_z_observable_vs_dmrg_with_splits(
    all_predictions: Dict[str, Dict],
    n_spins: int,
    dmrg_root: str,
    save_dir: str,
    split_config: Optional[Dict[str, Dict]] = None,
    training_cutoff: Optional[float] = None,
    include_skqd_comparison: bool = False,
    plot_boundary_focus: bool = False
):
    """
    Plot Z observable predictions vs DMRG data with all splits combined.
    Includes: relative error and staggered magnetization with DMRG overlay.
    
    Args:
        all_predictions: Dictionary mapping split names to predictions
        n_spins: Number of spins in the system
        dmrg_root: Root directory containing DMRG data files
        save_dir: Directory to save plots
        split_config: Optional custom split configuration (see plot_z_observable_vs_skqd_with_splits)
        training_cutoff: Optional Jz value to mark training cutoff with vertical line
        include_skqd_comparison: If True, also show SKQD errors vs DMRG
        plot_boundary_focus: If True, create additional boundary region focus plot
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Use default or custom split configuration
    if split_config is None:
        split_colors = SPLIT_COLORS
        split_markers = SPLIT_MARKERS
        split_labels = {"train": "Train", "val": "Val", "test": "Test"}
        split_order = ["train", "val", "test"]
    else:
        split_colors = split_config.get("colors", SPLIT_COLORS)
        split_markers = split_config.get("markers", SPLIT_MARKERS)
        split_labels = split_config.get("labels", {})
        split_order = list(split_colors.keys())

    # Collect ML predictions and load DMRG data
    z_vals_ml = {}
    z_vals_skqd = {}
    z_vals_dmrg = {}
    split_info = {}

    for split_name, predictions in all_predictions.items():
        for jz in predictions.keys():
            dmrg_data = load_dmrg_data(jz, dmrg_root)
            if dmrg_data is None:
                continue

            print(dmrg_data.keys())
            dmrg_result: DMRGResult = dmrg_data["result"]
            dmrg_obs = dmrg_result.psi.expectation_value("Sz") * 2
            order = dmrg_result.bfs_order
            dmrg_obs = inverse_order_single_site_obs(n_spins, order, dmrg_obs)

            z_vals_ml[jz] = predictions[jz]["preds"]
            z_vals_skqd[jz] = predictions[jz]["labels"]
            z_vals_dmrg[jz] = dmrg_obs
            split_info[jz] = split_name

    if len(z_vals_dmrg) == 0:
        print("Warning: No DMRG data found")
        return

    jz_list = sorted(z_vals_dmrg.keys())

    # 1. Plot average relative error vs Jz
    fig_width = 12 if include_skqd_comparison else 10
    plt.figure(figsize=(fig_width, 6))

    if include_skqd_comparison:
        # Plot SKQD errors (lighter, smaller markers)
        for split_name in split_order:
            split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
            skqd_errors = []
            for jz in split_jzs:
                rel_err = compute_relative_error(z_vals_skqd[jz], z_vals_dmrg[jz],
                                                 as_percentage=True)
                skqd_errors.append(float(np.nanmean(rel_err)))

            if len(split_jzs) > 0:
                plt.scatter(split_jzs, skqd_errors,
                            marker=split_markers[split_name],
                            color=split_colors[split_name],
                            label=f'{split_labels.get(split_name, split_name.capitalize())} SKQD',
                            s=60, alpha=0.4, edgecolors='black', linewidths=0.3)

    # Plot ML errors (darker, larger markers)
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        split_errors = []
        for jz in split_jzs:
            rel_err = compute_relative_error(z_vals_ml[jz], z_vals_dmrg[jz], as_percentage=True)
            split_errors.append(float(np.nanmean(rel_err)))

        if len(split_jzs) > 0:
            plt.scatter(split_jzs, split_errors,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        label=f'{split_labels.get(split_name, split_name.capitalize())} ML',
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        plt.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel(r"$\Delta Z$ (%)", fontsize=14)
    plt.title("ML vs DMRG: Z Observable (All Splits)", fontsize=14)
    legend_ncol = 2 if include_skqd_comparison else 1
    plt.legend(fontsize=9, loc='best', ncol=legend_ncol)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_rel_error_vs_dmrg_jz_all_splits.pdf"))
    plt.close()

    # 2. Staggered magnetization with DMRG overlay and relative error subplot
    def compute_staggered_magnetization_dmrg(z_vals, z_dmrg):
        """Compute staggered magnetization using DMRG pattern."""
        eta = np.sign(z_dmrg)
        eta = np.where(eta == 0, 1.0, eta)
        return np.mean(eta * z_vals)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top: Staggered magnetization values
    ms_ml_all = {jz: float(compute_staggered_magnetization_dmrg(z_vals_ml[jz], z_vals_dmrg[jz]))
                 for jz in jz_list}
    ms_skqd_all = {jz: float(compute_staggered_magnetization_dmrg(z_vals_skqd[jz], z_vals_dmrg[jz]))
                   for jz in jz_list}
    ms_dmrg_all = {jz: float(compute_staggered_magnetization_dmrg(z_vals_dmrg[jz], z_vals_dmrg[jz]))
                   for jz in jz_list}

    # Plot DMRG reference
    ax1.scatter(jz_list, [ms_dmrg_all[jz] for jz in jz_list],
                marker='x', color='gray', s=150, alpha=0.7, linewidths=2,
                label='DMRG Reference', zorder=1)

    if include_skqd_comparison:
        # Plot SKQD predictions by split (lighter, smaller)
        for split_name in split_order:
            split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
            ms_vals = [ms_skqd_all[jz] for jz in split_jzs]

            if len(split_jzs) > 0:
                ax1.scatter(split_jzs, ms_vals,
                            marker=split_markers[split_name],
                            color=split_colors[split_name],
                            label=f'SKQD {split_labels.get(split_name, split_name.capitalize())}',
                            s=60, alpha=0.4, edgecolors='black', linewidths=0.3, zorder=2)

    # Plot ML predictions by split (darker, larger)
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        ms_vals = [ms_ml_all[jz] for jz in split_jzs]

        if len(split_jzs) > 0:
            ax1.scatter(split_jzs, ms_vals,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        label=f'ML {split_labels.get(split_name, split_name.capitalize())}',
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5, zorder=3)

    if training_cutoff is not None:
        ax1.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    ax1.set_ylabel(r"$m_s = \frac{1}{N}\sum_i \eta_i\langle Z_i\rangle$", fontsize=14)
    ax1.set_title("Staggered Magnetization: ML vs DMRG", fontsize=14)
    legend_ncol = 2 if include_skqd_comparison else 1
    ax1.legend(fontsize=8, loc='best', ncol=legend_ncol)
    ax1.grid(True, alpha=0.3)

    # Bottom: Relative error
    if include_skqd_comparison:
        # Plot SKQD errors (lighter)
        for split_name in split_order:
            split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
            skqd_errors = []
            for jz in split_jzs:
                denom = ms_dmrg_all[jz] if ms_dmrg_all[jz] != 0 else np.nan
                rel_err = abs((ms_skqd_all[jz] - ms_dmrg_all[jz]) / denom) * 100 if not np.isnan(
                    denom) else np.nan
                skqd_errors.append(rel_err)

            if len(split_jzs) > 0:
                ax2.scatter(split_jzs, skqd_errors,
                            marker=split_markers[split_name],
                            color=split_colors[split_name],
                            s=60, alpha=0.4, edgecolors='black', linewidths=0.3)

    # Plot ML errors (darker)
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        ml_errors = []
        for jz in split_jzs:
            denom = ms_dmrg_all[jz] if ms_dmrg_all[jz] != 0 else np.nan
            rel_err = abs((ms_ml_all[jz] - ms_dmrg_all[jz]) / denom) * 100 if not np.isnan(
                denom) else np.nan
            ml_errors.append(rel_err)

        if len(split_jzs) > 0:
            ax2.scatter(split_jzs, ml_errors,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        ax2.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel(r"$J_z$", fontsize=14)
    ax2.set_ylabel(r"$\Delta m_s$ (%)", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "staggered_magnetization_vs_jz.pdf"))
    plt.close()

    # 3. Optional boundary region focus plot
    if plot_boundary_focus:
        # Find boundary split jzs
        boundary_split = None
        for split_name in split_order:
            if "boundary" in split_name.lower() or "test" in split_name.lower():
                boundary_split = split_name
                break

        if boundary_split is not None:
            boundary_jzs = [jz for jz in jz_list if split_info.get(jz) == boundary_split]

            if len(boundary_jzs) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                               gridspec_kw={'height_ratios': [2, 1]})

                # Top panel: Individual <Z> values scattered
                for jz in boundary_jzs:
                    ax1.scatter([jz] * len(z_vals_dmrg[jz]), z_vals_dmrg[jz],
                                marker='x', color='gray', s=20, alpha=0.5, edgecolors='none',
                                label='DMRG' if jz == boundary_jzs[0] else '', zorder=1)

                for jz in boundary_jzs:
                    ax1.scatter([jz] * len(z_vals_skqd[jz]), z_vals_skqd[jz],
                                marker='s', color='orange', s=25, alpha=0.6, edgecolors='none',
                                label='SKQD' if jz == boundary_jzs[0] else '', zorder=2)

                for jz in boundary_jzs:
                    ax1.scatter([jz] * len(z_vals_ml[jz]), z_vals_ml[jz],
                                marker='o', color='red', s=25, alpha=0.6, edgecolors='none',
                                label='ML' if jz == boundary_jzs[0] else '', zorder=3)

                ax1.set_ylabel(r"$\langle Z \rangle$", fontsize=14)
                ax1.set_title(f"Boundary Region: ML Predicts SKQD Error", fontsize=14)
                ax1.legend(fontsize=12, loc='best')
                ax1.set_ylim(-1.1, 1.1)
                ax1.grid(True, alpha=0.3)

                # Bottom panel: Average relative error vs DMRG
                skqd_boundary_errors = []
                ml_boundary_errors = []

                for jz in boundary_jzs:
                    avg_z_dmrg = float(np.mean(np.abs(z_vals_dmrg[jz])))
                    avg_z_skqd = float(np.mean(np.abs(z_vals_skqd[jz])))
                    avg_z_ml = float(np.mean(np.abs(z_vals_ml[jz])))

                    denom = avg_z_dmrg if avg_z_dmrg != 0 else np.nan
                    skqd_err = abs((avg_z_skqd - avg_z_dmrg) / denom) * 100 if not np.isnan(
                        denom) else np.nan
                    ml_err = abs((avg_z_ml - avg_z_dmrg) / denom) * 100 if not np.isnan(
                        denom) else np.nan

                    skqd_boundary_errors.append(skqd_err)
                    ml_boundary_errors.append(ml_err)

                ax2.scatter(boundary_jzs, skqd_boundary_errors,
                            marker='s', color='orange', s=120, alpha=0.7,
                            label='SKQD Error', edgecolors='black', linewidths=0.5)
                ax2.scatter(boundary_jzs, ml_boundary_errors,
                            marker='o', color='red', s=120, alpha=0.7,
                            label='ML Error', edgecolors='black', linewidths=0.5)

                ax2.set_xlabel(r"$J_z$", fontsize=14)
                ax2.set_ylabel(r"$\Delta \langle |Z| \rangle$ vs DMRG (%)", fontsize=14)
                ax2.legend(fontsize=12, loc='best')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "boundary_region_focus.pdf"))
                plt.close()
                print(f"Saved boundary region focus plot to {save_dir}")

    print(f"Saved Z vs DMRG plots to {save_dir}")

def extract_nn_from_observable_key(observable_key: str) -> int:
    """
    Extract the nearest neighbor (nn) value from observable_key.
    E.g., "2_nn_corr_funcs_zz_basis_opt" -> 2
    """
    if "_nn_corr_funcs" in observable_key:
        parts = observable_key.split("_")
        for i, part in enumerate(parts):
            if part == "nn" and i > 0:
                try:
                    return int(parts[i - 1])
                except ValueError:
                    pass
    return 2  # Default fallback

def plot_correlation_functions_vs_skqd_with_splits(
    all_predictions: Dict[str, Dict],
    observable_key: str,
    save_dir: str,
    split_config: Optional[Dict[str, Dict]] = None,
    training_cutoff: Optional[float] = None
):
    """
    Plot correlation function predictions vs SKQD data for all splits combined.
    
    Args:
        all_predictions: Dictionary mapping split names to predictions
        observable_key: Observable key string (e.g., "2_nn_corr_funcs_zz_basis_opt")
        save_dir: Directory to save plots
        split_config: Optional custom split configuration (see plot_z_observable_vs_skqd_with_splits)
        training_cutoff: Optional Jz value to mark training cutoff with vertical line
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Use default or custom split configuration
    if split_config is None:
        split_colors = SPLIT_COLORS
        split_markers = SPLIT_MARKERS
        split_labels = {"train": "Train", "val": "Val", "test": "Test"}
        split_order = ["train", "val", "test"]
    else:
        split_colors = split_config.get("colors", SPLIT_COLORS)
        split_markers = split_config.get("markers", SPLIT_MARKERS)
        split_labels = split_config.get("labels", {})
        split_order = list(split_colors.keys())

    # Collect all data from all splits
    corr_vals_ml, corr_vals_skqd, split_info = collect_predictions_by_jz(all_predictions)
    jz_list = sorted(corr_vals_ml.keys())

    # 1. Plot average relative error vs Jz
    plt.figure(figsize=(10, 6))
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        split_errors = []
        for jz in split_jzs:
            preds = corr_vals_ml[jz]
            labels = corr_vals_skqd[jz]
            rel_err = compute_relative_error(preds, labels, as_percentage=True)
            split_errors.append(float(np.mean(rel_err)))

        if len(split_jzs) > 0:
            plt.scatter(split_jzs, split_errors,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        label=split_labels.get(split_name, split_name.capitalize()),
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        plt.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel(r"Average Relative Error (%)", fontsize=14)
    plt.title(f"ML vs SKQD: {observable_key} (All Splits)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_rel_error_vs_jz.pdf"))
    plt.close()

    # 2. Plot correlation values over Jz with SKQD reference overlay
    plt.figure(figsize=(10, 6))

    # Plot SKQD reference data (gray, smaller, background)
    for jz in jz_list:
        plt.scatter([jz] * len(corr_vals_skqd[jz]), corr_vals_skqd[jz],
                    marker='x', color='gray', s=15, alpha=0.4, edgecolors='none',
                    label='SKQD' if jz == jz_list[0] else '')

    # Plot ML predictions (colored by split, foreground)
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        if len(split_jzs) > 0:
            for jz in split_jzs:
                plt.scatter([jz] * len(corr_vals_ml[jz]), corr_vals_ml[jz],
                            marker=split_markers[split_name],
                            color=split_colors[split_name],
                            s=20, alpha=0.5, edgecolors='none')

    # Add legend
    for split_name in split_order:
        plt.scatter([], [], marker=split_markers[split_name],
                    color=split_colors[split_name],
                    label=f'ML {split_labels.get(split_name, split_name.capitalize())}',
                    s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        plt.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel(r"Correlation Function Value", fontsize=14)
    plt.title(f"{observable_key}: ML Predictions vs SKQD Reference", fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "corr_values_over_jz.pdf"))
    plt.close()

    # 3. Plot mean correlation value with error bars
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top: Mean correlation values
    mean_ml_all = {jz: float(np.mean(corr_vals_ml[jz])) for jz in jz_list}
    mean_skqd_all = {jz: float(np.mean(corr_vals_skqd[jz])) for jz in jz_list}

    # Plot SKQD reference
    ax1.scatter(jz_list, [mean_skqd_all[jz] for jz in jz_list],
                marker='x', color='gray', s=150, alpha=0.7, linewidths=2,
                label='SKQD Reference', zorder=1)

    # Plot ML predictions by split
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        mean_vals = [mean_ml_all[jz] for jz in split_jzs]

        if len(split_jzs) > 0:
            ax1.scatter(split_jzs, mean_vals,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        label=f'ML {split_labels.get(split_name, split_name.capitalize())}',
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5, zorder=2)

    if training_cutoff is not None:
        ax1.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    ax1.set_ylabel(r"Mean Correlation Value", fontsize=14)
    ax1.set_title(f"{observable_key}: ML vs SKQD", fontsize=14)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Bottom: Relative error
    for split_name in split_order:
        split_jzs = [jz for jz in jz_list if split_info.get(jz) == split_name]
        rel_errors = []
        for jz in split_jzs:
            denom = mean_skqd_all[jz] if mean_skqd_all[jz] != 0 else np.nan
            rel_err = abs((mean_ml_all[jz] - mean_skqd_all[jz]) / denom) * 100 if not np.isnan(
                denom) else np.nan
            rel_errors.append(rel_err)

        if len(split_jzs) > 0:
            ax2.scatter(split_jzs, rel_errors,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        ax2.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel(r"$J_z$", fontsize=14)
    ax2.set_ylabel(r"Relative Error (%)", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mean_corr_vs_jz.pdf"))
    plt.close()
    print(f"Saved correlation vs SKQD plots to {save_dir}")

def plot_correlation_functions_vs_dmrg_with_splits(
    all_predictions: Dict[str, Dict],
    observable_key: str,
    n_spins: int,
    dmrg_root: str,
    data_root: str,
    save_dir: str,
    nn: int = 2,
    split_config: Optional[Dict[str, Dict]] = None,
    training_cutoff: Optional[float] = None,
    include_skqd_comparison: bool = False
):
    """
    Generate comparison plots against DMRG reference data for correlation functions.
    
    Args:
        all_predictions: Dictionary mapping split names to predictions
        observable_key: Observable key string
        n_spins: Number of spins in the system
        dmrg_root: Root directory containing DMRG data files
        data_root: Root directory containing SKQD data files (for alignment)
        save_dir: Directory to save plots
        nn: Nearest neighbor value
        split_config: Optional custom split configuration
        training_cutoff: Optional Jz value to mark training cutoff
        include_skqd_comparison: If True, also show SKQD errors vs DMRG
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Use default or custom split configuration
    if split_config is None:
        split_colors = SPLIT_COLORS
        split_markers = SPLIT_MARKERS
        split_labels = {"train": "Train", "val": "Val", "test": "Test"}
        split_order = ["train", "val", "test"]
    else:
        split_colors = split_config.get("colors", SPLIT_COLORS)
        split_markers = split_config.get("markers", SPLIT_MARKERS)
        split_labels = split_config.get("labels", {})
        split_order = list(split_colors.keys())

    fig_width = 12 if include_skqd_comparison else 10
    plt.figure(figsize=(fig_width, 6))

    for split_name in split_order:
        predictions = all_predictions.get(split_name, {})
        if len(predictions) == 0:
            continue

        jz_list = sorted(predictions.keys())
        skqd_errors = []
        ml_errors = []

        for jz in jz_list:
            preds = predictions[jz]["preds"]
            labels = predictions[jz]["labels"]

            # Load DMRG data
            data_dmrg = load_dmrg_data(jz, dmrg_root)
            if data_dmrg is None:
                continue

            try:
                dmrg_result: DMRGResult = data_dmrg["result"]
                order = dmrg_result.bfs_order
                pairs_dmrg = data_dmrg[f"{nn}_nn_corr_pairs"]

                try:
                    corr_dmrg = data_dmrg[f"{nn}_nn_corr_funcs_zz"]
                except KeyError:
                    psi = dmrg_result.psi
                    corr_dmrg = []
                    for pair in pairs_dmrg:
                        zz_corr = psi.expectation_value_term([("Sz", pair[0]), ("Sz", pair[1])]) * 4
                        corr_dmrg.append(zz_corr)

                # Use inverse_order_two_site_list to reorder both pairs and correlations
                reord_pairs_dmrg, reord_corr_dmrg = inverse_order_two_site_list(
                    n_spins, order, pairs_dmrg, corr_dmrg, sort_pairs=False
                )

                # Load SKQD pairs to establish the ordering
                data_skqd = load_skqd_data(jz, data_root)
                if data_skqd is not None:
                    pairs_skqd = data_skqd[f"{nn}_nn_corr_pairs_zz_basis_opt"]

                    # Create mapping from DMRG pairs to correlations
                    dmrg_dict = {tuple(pair): corr for pair, corr in
                                 zip(reord_pairs_dmrg, reord_corr_dmrg)}

                    # Align DMRG correlations to match SKQD/ML order
                    dmrg_vals_aligned = []
                    preds_aligned = []
                    labels_aligned = []
                    for i, pair in enumerate(pairs_skqd):
                        pair_tuple = tuple(pair)
                        if pair_tuple in dmrg_dict and i < len(preds):
                            dmrg_vals_aligned.append(dmrg_dict[pair_tuple])
                            preds_aligned.append(preds[i])
                            labels_aligned.append(labels[i])

                    if len(dmrg_vals_aligned) == 0:
                        continue

                    dmrg_vals = np.array(dmrg_vals_aligned)
                    preds = np.array(preds_aligned)
                    labels = np.array(labels_aligned)
                else:
                    dmrg_vals = np.array(reord_corr_dmrg)

                # Compute relative errors
                if include_skqd_comparison:
                    skqd_rel_err = compute_relative_error(labels, dmrg_vals, as_percentage=True)
                    skqd_errors.append(np.mean(skqd_rel_err))

                ml_rel_err = compute_relative_error(preds, dmrg_vals, as_percentage=True)
                ml_errors.append(np.mean(ml_rel_err))

            except KeyError as e:
                print(f"Warning: {observable_key} not found in DMRG data for jz={jz}: {e}")
                continue

        if len(ml_errors) > 0:
            if include_skqd_comparison and len(skqd_errors) > 0:
                # Plot SKQD errors (lighter)
                plt.scatter(jz_list[:len(skqd_errors)], skqd_errors,
                            marker=split_markers[split_name],
                            color=split_colors[split_name],
                            label=f'{split_labels.get(split_name, split_name.capitalize())} SKQD',
                            s=60, alpha=0.4, edgecolors='black', linewidths=0.3)

            # Plot ML errors (darker)
            plt.scatter(jz_list[:len(ml_errors)], ml_errors,
                        marker=split_markers[split_name],
                        color=split_colors[split_name],
                        label=f'{split_labels.get(split_name, split_name.capitalize())} ML',
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    if training_cutoff is not None:
        plt.axvline(x=training_cutoff, color='gray', linestyle='--', alpha=0.5,
                    label=f'Training cutoff (jz={training_cutoff})')

    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel("Average Relative Error (%)", fontsize=14)
    plt.title(f"ML vs DMRG: {observable_key} (All Splits)", fontsize=14)
    legend_ncol = 2 if include_skqd_comparison else 1
    plt.legend(fontsize=9, loc='best', ncol=legend_ncol)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_rel_error_vs_dmrg_jz_all_splits.pdf"))
    plt.close()
    print(f"Saved correlation vs DMRG plots to {save_dir}")

def plot_heavy_hex_correlations_vs_skqd(
    predictions: Dict,
    observable_key: str,
    n_spins: int,
    data_root: str,
    save_dir: str,
    nn: int = 2,
    split_name: str = "test"
):
    """
    Generate heavy hex lattice visualizations comparing ML vs SKQD.
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    heavy_hex_dir = os.path.join(save_dir, "heavy_hex_plots")
    Path(heavy_hex_dir).mkdir(parents=True, exist_ok=True)

    if n_spins == 115:
        distance = 7
    elif n_spins == 57:
        distance = 5

    for jz, pred_data in predictions.items():
        print(f"  Generating heavy hex plots vs SKQD for Jz={jz} ({split_name})...")

        ml_preds = pred_data["preds"]
        skqd_labels = pred_data["labels"]

        # Load SKQD data to get pairs and z values
        skqd_file = f"XXZ_2d_jz_{jz:.1f}.pkl"
        skqd_path = os.path.join(data_root, skqd_file)

        if not os.path.exists(skqd_path):
            print(f"    Warning: SKQD data not found at {skqd_path}")
            continue

        with open(skqd_path, "rb") as f:
            data_skqd = pickle.load(f)

        try:
            pairs_skqd = data_skqd[f"{nn}_nn_corr_pairs_zz_basis_opt"]
            zs_skqd = data_skqd.get("z_basis_opt", None)

            if zs_skqd is None:
                zs_skqd = szs_from_state_dict(data_skqd["ground_state_dict"])
                zs_skqd = np.array(zs_skqd[::-1]) * 2
        except KeyError as e:
            print(f"    Warning: Required data not found in SKQD file: {e}")
            continue

        # Filter to desired nn for plotting
        nn_to_plot = min(nn, 2)  # Limit to 2 for cleaner visualization
        pairs_filtered, ml_preds_filtered = filter_to_less_neighbours(
            pairs_skqd, ml_preds, nn_to_plot, distance
        )
        _, skqd_labels_filtered = filter_to_less_neighbours(
            pairs_skqd, skqd_labels, nn_to_plot, distance
        )

        # Plot ML predictions
        plot = plot_corr_heavy_hex(pairs_filtered, n_spins, ml_preds_filtered, zs_skqd,
                                   plt.cm.seismic)
        plot.savefig(os.path.join(heavy_hex_dir, f"ml_{split_name}_jz_{jz:.1f}.pdf"))
        plot.close()

        # Plot SKQD data
        plot = plot_corr_heavy_hex(pairs_filtered, n_spins, skqd_labels_filtered, zs_skqd,
                                   plt.cm.seismic)
        plot.savefig(os.path.join(heavy_hex_dir, f"skqd_{split_name}_jz_{jz:.1f}.pdf"))
        plot.close()

        # Plot difference (ML - SKQD)
        diff_ml_skqd = np.array(ml_preds_filtered) - np.array(skqd_labels_filtered)
        plot = plot_corr_heavy_hex(pairs_filtered, n_spins, diff_ml_skqd, np.zeros(len(zs_skqd)),
                                   plt.cm.RdBu_r)
        plot.savefig(
            os.path.join(heavy_hex_dir, f"diff_ml_minus_skqd_{split_name}_jz_{jz:.1f}.pdf"))
        plot.close()

def plot_heavy_hex_correlations_vs_dmrg(
    predictions: Dict,
    observable_key: str,
    n_spins: int,
    data_root: str,
    dmrg_root: str,
    save_dir: str,
    nn: int = 2,
    split_name: str = "test",
):
    """
    Generate heavy hex lattice visualizations comparing ML vs DMRG.
    Uses proper DMRG reordering with inverse_order_two_site_list.
    
    Args:
        predictions: Dictionary of predictions by Jz
        observable_key: Observable key string
        n_spins: Number of spins
        data_root: Root directory for SKQD data
        dmrg_root: Root directory for DMRG data
        save_dir: Directory to save plots
        nn: Nearest neighbor value
        split_name: Name of the split (e.g., "test")
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    heavy_hex_dir = os.path.join(save_dir, "heavy_hex_plots")
    Path(heavy_hex_dir).mkdir(parents=True, exist_ok=True)

    for jz, pred_data in predictions.items():
        print(f"  Generating heavy hex plots vs DMRG for Jz={jz} ({split_name})...")

        ml_preds = pred_data["preds"]

        # Load SKQD data to get pairs and z values
        skqd_file = f"XXZ_2d_jz_{jz:.1f}.pkl"
        skqd_path = os.path.join(data_root, skqd_file)

        if not os.path.exists(skqd_path):
            print(f"    Warning: SKQD data not found at {skqd_path}")
            continue

        with open(skqd_path, "rb") as f:
            data_skqd = pickle.load(f)

        try:
            pairs_skqd = data_skqd[f"{nn}_nn_corr_pairs_zz_basis_opt"]
            zs_skqd = data_skqd.get("z_basis_opt", None)

            if zs_skqd is None:
                zs_skqd = szs_from_state_dict(data_skqd["ground_state_dict"])
                zs_skqd = np.array(zs_skqd[::-1]) * 2
        except KeyError as e:
            print(f"    Warning: Required data not found in SKQD file: {e}")
            continue

        # Load DMRG data
        dmrg_file = f"XXZ_2d_jz_{jz:.1f}.pkl"
        dmrg_path = os.path.join(dmrg_root, dmrg_file)

        if not os.path.exists(dmrg_path):
            print(f"    Warning: DMRG data not found at {dmrg_path}")
            continue

        try:
            with open(dmrg_path, "rb") as f:
                data_dmrg = pickle.load(f)

            dmrg_result: DMRGResult = data_dmrg["result"]
            psi = dmrg_result.psi
            order = dmrg_result.bfs_order

            pairs_dmrg = data_dmrg[f"{nn}_nn_corr_pairs"]

            # Compute or load DMRG ZZ correlations
            try:
                corr_dmrg = data_dmrg[f"{nn}_nn_corr_funcs_zz"]
            except KeyError:
                corr_dmrg = []
                for pair in pairs_dmrg:
                    zz_corr = psi.expectation_value_term([("Sz", pair[0]), ("Sz", pair[1])]) * 4
                    corr_dmrg.append(zz_corr)

            # CRITICAL: Use inverse_order_two_site_list to reorder BOTH pairs AND correlations together
            reord_pairs_dmrg, reord_corr_dmrg = inverse_order_two_site_list(
                n_spins, order, pairs_dmrg, corr_dmrg, sort_pairs=False
            )

            # Get DMRG single-site magnetizations
            zs_dmrg = psi.expectation_value("Sz") * 2
            zs_dmrg = inverse_order_single_site_obs(n_spins, order, zs_dmrg)
            if zs_dmrg[0] < 0:
                zs_dmrg *= -1

            # Filter to desired nn for plotting
            nn_to_plot = min(nn, 2)

            # Convert to lists for filtering
            reord_pairs_dmrg = list(reord_pairs_dmrg)
            reord_corr_dmrg = list(reord_corr_dmrg)
            pairs_skqd = list(pairs_skqd)
            ml_preds = list(ml_preds)

            # Filter BOTH datasets to the same nn
            reord_pairs_dmrg_filtered, reord_corr_dmrg_filtered = filter_to_less_neighbours(
                reord_pairs_dmrg, reord_corr_dmrg, nn_to_plot
            )
            pairs_skqd_filtered, ml_preds_filtered = filter_to_less_neighbours(
                pairs_skqd, ml_preds, nn_to_plot
            )

            # Create mapping from SKQD pairs to ML predictions
            pairs_skqd_dict = {tuple(pair): ml_pred for pair, ml_pred in
                              zip(pairs_skqd_filtered, ml_preds_filtered)}

            # Align ML predictions to match DMRG pair ordering
            ml_preds_aligned = []
            for pair in reord_pairs_dmrg_filtered:
                pair_tuple = tuple(pair)
                pair_tuple_rev = tuple(pair[::-1])
                if pair_tuple in pairs_skqd_dict:
                    ml_preds_aligned.append(pairs_skqd_dict[pair_tuple])
                elif pair_tuple_rev in pairs_skqd_dict:
                    ml_preds_aligned.append(pairs_skqd_dict[pair_tuple_rev])
                else:
                    # If pair not in ML data, use NaN (won't be plotted)
                    ml_preds_aligned.append(np.nan)

            ml_preds_aligned = np.array(ml_preds_aligned)
            dmrg_corr_aligned = np.array(reord_corr_dmrg_filtered)

            # Plot DMRG data (all filtered pairs)
            plot = plot_corr_heavy_hex(reord_pairs_dmrg_filtered, n_spins, reord_corr_dmrg_filtered,
                                       zs_dmrg, plt.cm.seismic)
            plot.savefig(os.path.join(heavy_hex_dir, f"dmrg_{split_name}_jz_{jz:.1f}.pdf"))
            plot.close()

            # Plot difference (ML - DMRG) using aligned data
            diff_ml_dmrg = ml_preds_aligned - dmrg_corr_aligned
            plot = plot_corr_heavy_hex(reord_pairs_dmrg_filtered, n_spins, diff_ml_dmrg, zs_dmrg,
                                       plt.cm.RdBu_r)
            plot.savefig(
                os.path.join(heavy_hex_dir, f"diff_ml_minus_dmrg_{split_name}_jz_{jz:.1f}.pdf"))
            plot.close()

            # Count valid (non-NaN) differences
            valid_diffs = diff_ml_dmrg[~np.isnan(diff_ml_dmrg)]
            print(f"    Matched {len(valid_diffs)} pairs between ML and DMRG")
            if len(valid_diffs) > 0:
                print(f"    Mean absolute difference: {np.mean(np.abs(valid_diffs)):.4f}")
                print(f"    Max absolute difference: {np.max(np.abs(valid_diffs)):.4f}")

        except Exception as e:
            print(f"    Warning: Error processing DMRG data: {e}")
            traceback.print_exc()
