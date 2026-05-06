"""
Script to generate top panel of Figure 2

Unified script to compute and plot basis-optimized observables (z, zz, xx, z_loop, x_loop)
for both n=57 and n=115 datasets, comparing against chi=320 DMRG.

Key features:
- Handles both datasets with appropriate heavy hex distances (d=5 for n=57, d=7 for n=115)
- Computes all basis-optimized observables
- Creates combined plots showing both systems on the same axes
"""

import os
import sys
from pathlib import Path
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from networkx.drawing.nx_agraph import graphviz_layout

# Fix for unpickling objects from original /qaml/ repo
import qaml.diagonalisation
import qaml.diagonalisation.twod
import qaml.diagonalisation.twod.dmrg
sys.modules['diagonalisation'] = qaml.diagonalisation
sys.modules['diagonalisation.twod'] = qaml.diagonalisation.twod
sys.modules['diagonalisation.twod.dmrg'] = qaml.diagonalisation.twod.dmrg

from qaml.analysis.analysis_utils import corrs_to_matrix, filter_to_less_neighbours
from qaml.diagonalisation.twod.dmrg import DMRGResult, inverse_order_single_site_obs, inverse_order_two_site_list
from qaml.graph.graph_utils import read_edges_txt

# Configure matplotlib to use LaTeX for text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# ============================================================================
# Configuration for both datasets
# ============================================================================

DATASETS = {
    57: {
        'root_dir_skqd': '../../data/spins_57/skqd/ts_1_kd_11_shots_100k_ibm_boston_1773299045/recovery_random_flip',
        'heavy_hex_distance': 5,
        'edges_path': '../../cmaps/heavy_hex_edges_d_5.txt',
        'chi_max': 320,  # Use chi=320 for n=57
    },
    115: {
        'root_dir_skqd': '../../data/spins_115/skqd/ts_1_kd_11_shots_100k_ibm_boston_1773150437_1773854302_mixed/recovery_random_flip',
        'heavy_hex_distance': 7,
        'edges_path': '../../cmaps/heavy_hex_edges_d_7.txt',
        'chi_max': 320,  # Use chi=320 for n=115
    }
}

# DMRG parameters
TRUNC_CUT = 1e-6
DMRG_NN = 2
SKQD_NN = 2
NN_TO_PLOT = 2

# Jz range
JZS = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]

# ============================================================================
# Helper functions
# ============================================================================

def plot_loops_heavy_hex(edges, n, loops, loop_vals, cmap=plt.cm.Blues,
                         vmin=None, vmax=None, alpha=0.60, title=None,
                         cbar_label=r"$\langle Z^{\otimes 12}\rangle$",
                         show_labels=True, scale="linear", linthresh=1e-3, base=10):
    """Plot loop observables on heavy hex lattice."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = graphviz_layout(G)

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

    edge_art = nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.35, width=1.0, ax=ax)
    if edge_art is not None:
        try:
            edge_art.set_zorder(2)
        except Exception:
            pass

    node_art = nx.draw_networkx_nodes(G, pos, node_size=70, node_color="k", ax=ax)
    if node_art is not None:
        try:
            node_art.set_zorder(3)
        except Exception:
            pass

    if show_labels:
        text_dict = nx.draw_networkx_labels(G, pos, labels={i: i for i in range(n)},
                                           font_size=5, ax=ax)
        try:
            for t in text_dict.values():
                t.set_zorder(4)
        except Exception:
            pass

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label)

    if title:
        ax.set_title(title)

    ax.set_axis_off()
    return fig

def compute_dmrg_loop_if_missing(data_dmrg, dmrg_result, loops_old, key="z_loop", operator="Sz"):
    """Compute DMRG loop observables if not already present."""
    if key in data_dmrg and f"{key}_sites" in data_dmrg:
        return np.asarray(data_dmrg[key]), list(data_dmrg[f"{key}_sites"])

    print(f"No DMRG {key} data found, computing...")

    psi = dmrg_result.psi
    n = psi.L
    order = dmrg_result.bfs_order

    pos = np.empty(n, dtype=int)
    for new, old in enumerate(order):
        pos[int(old)] = int(new)

    corrs = []
    scale = 2 ** len(loops_old[0])  # S = Pauli/2 => Pauli^⊗k = 2^k S^⊗k

    for loop in loops_old:
        loop_chain = [int(pos[i]) for i in loop]
        loop_chain.sort()
        term = [(operator, i) for i in loop_chain]
        val = psi.expectation_value_term(term) * scale
        corrs.append(val)

    corrs = np.asarray(corrs)
    data_dmrg[key] = corrs
    data_dmrg[f"{key}_sites"] = loops_old

    print(f"max |{key}| =", float(np.max(np.abs(corrs))))
    return corrs, loops_old

# ============================================================================
# Data loading and processing
# ============================================================================

def load_data_for_system(n, config, jzs):
    """Load SKQD and DMRG data for a given system size."""
    root_dir_skqd = config['root_dir_skqd']
    chi_max = config['chi_max']
    root_dir_dmrg = f"../../data/spins_{n}/dmrg/chi_max_{chi_max:d}_trunc_{TRUNC_CUT:.0e}"
    heavy_hex_distance = config['heavy_hex_distance']
    
    data = {
        'z_skqd': {}, 'z_dmrg': {},
        'zz_skqd': {}, 'zz_dmrg': {}, 'zz_pairs_skqd': {}, 'zz_pairs_dmrg': {},
        'z_loop_skqd': {}, 'z_loop_dmrg': {}, 'z_loop_sites': {},
        'ms_skqd': {}, 'ms_dmrg': {},  # Staggered magnetization
        'valid_jzs': []
    }
    
    for jz in jzs:
        jz = round(jz, 1)
        print(f"n={n}, Jz={jz}")
        
        # Load SKQD data
        try:
            with open(os.path.join(root_dir_skqd, f"XXZ_2d_jz_{jz:.1f}.pkl"), "rb") as f:
                data_skqd = pickle.load(f)
        except (FileNotFoundError, EOFError) as e:
            print(f"  Skipping Jz {jz}: {e}")
            continue
        
        # Load DMRG data
        try:
            with open(os.path.join(root_dir_dmrg, f"XXZ_2d_jz_{jz:.1f}.pkl"), "rb") as f:
                data_dmrg = pickle.load(f)
        except FileNotFoundError:
            print(f"  No DMRG data found, skipping Jz {jz}")
            continue
        
        dmrg_result: DMRGResult = data_dmrg["result"]
        order = dmrg_result.bfs_order
        
        # ---- Z observable and staggered magnetization ----
        try:
            z_skqd = data_skqd["z_basis_opt"]
            z_dmrg = dmrg_result.psi.expectation_value("Sz") * 2
            z_dmrg = inverse_order_single_site_obs(n, order, z_dmrg)
            data['z_skqd'][jz] = np.array(z_skqd)
            data['z_dmrg'][jz] = np.array(z_dmrg)
            
            # Compute staggered magnetization
            # eta determines the two-color convention from DMRG
            eta = np.sign(z_dmrg)
            eta = np.where(eta == 0, 1.0, eta)
            data['ms_dmrg'][jz] = float(np.mean(eta * z_dmrg))
            data['ms_skqd'][jz] = float(np.mean(eta * z_skqd))
        except KeyError:
            print(f"  No z_basis_opt data for Jz {jz}")
        
        # ---- ZZ correlators ----
        try:
            corrs_zz_skqd = data_skqd[f"{SKQD_NN:d}_nn_corr_funcs_zz_basis_opt"]
            pairs_zz_skqd = data_skqd[f"{SKQD_NN:d}_nn_corr_pairs_zz_basis_opt"]
            
            pairs_dmrg = data_dmrg[f"{DMRG_NN:d}_nn_corr_pairs"]
            
            # Compute or load DMRG ZZ correlators
            try:
                corrs_zz_dmrg = data_dmrg[f"{DMRG_NN:d}_nn_corr_funcs_zz"]
            except KeyError:
                print(f"  Computing DMRG ZZ correlators for Jz {jz}")
                corrs_zz_dmrg = []
                for pair in pairs_dmrg:
                    zz_corr = dmrg_result.psi.expectation_value_term([("Sz", pair[0]), ("Sz", pair[1])]) * 4
                    corrs_zz_dmrg.append(zz_corr)
                data_dmrg[f"{DMRG_NN:d}_nn_corr_funcs_zz"] = corrs_zz_dmrg
                with open(os.path.join(root_dir_dmrg, f"XXZ_2d_jz_{jz:.1f}.pkl"), "wb") as f:
                    pickle.dump(data_dmrg, f)
            
            reord_pairs_dmrg, _ = inverse_order_two_site_list(n, order, pairs_dmrg, corrs_zz_dmrg, False)
            
            # Filter to nearest neighbors
            pairs_zz_skqd = list(pairs_zz_skqd)
            corrs_zz_skqd = list(corrs_zz_skqd)
            pairs_zz_skqd, corrs_zz_skqd = filter_to_less_neighbours(
                pairs_zz_skqd, corrs_zz_skqd, NN_TO_PLOT, heavy_hex_distance
            )
            
            reord_pairs_dmrg = list(reord_pairs_dmrg)
            corrs_zz_dmrg = list(corrs_zz_dmrg)
            reord_pairs_dmrg, corrs_zz_dmrg = filter_to_less_neighbours(
                reord_pairs_dmrg, corrs_zz_dmrg, NN_TO_PLOT, heavy_hex_distance
            )
            
            data['zz_skqd'][jz] = corrs_zz_skqd
            data['zz_dmrg'][jz] = corrs_zz_dmrg
            data['zz_pairs_skqd'][jz] = pairs_zz_skqd
            data['zz_pairs_dmrg'][jz] = reord_pairs_dmrg
        except KeyError:
            print(f"  No zz_basis_opt data for Jz {jz}")
        
        # ---- Z loop observables ----
        try:
            corrs_z_loop_skqd = np.asarray(data_skqd["z_loop"])
            sites_z_loop_skqd = list(data_skqd["z_loop_sites"])
            
            # Check if we need to compute before calling the function
            need_to_save_z = "z_loop" not in data_dmrg or "z_loop_sites" not in data_dmrg
            
            corrs_z_loop_dmrg, sites_z_loop_dmrg = compute_dmrg_loop_if_missing(
                data_dmrg, dmrg_result, sites_z_loop_skqd, key="z_loop", operator="Sz"
            )
            
            data['z_loop_skqd'][jz] = corrs_z_loop_skqd
            data['z_loop_dmrg'][jz] = corrs_z_loop_dmrg
            data['z_loop_sites'][jz] = sites_z_loop_skqd
            
            # Save updated DMRG data if we computed new loops
            if need_to_save_z:
                with open(os.path.join(root_dir_dmrg, f"XXZ_2d_jz_{jz:.1f}.pkl"), "wb") as f:
                    pickle.dump(data_dmrg, f)
        except KeyError:
            print(f"  No z_loop data for Jz {jz}")
        
        data['valid_jzs'].append(jz)
    
    return data

# ============================================================================
# Plotting functions
# ============================================================================

def plot_z_observables(data_57, data_115, output_dir):
    """Plot Z observables for both systems with 3 Jz ranges."""
    fig_dir = os.path.join(output_dir, "z_basis_opt")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    
    # Define Jz ranges: full, low (1.1-1.4), high (1.6-6.0)
    jz_ranges = [
        ("full", None, None, "z_error_comparison_n57_n115.pdf"),
        ("low", 1.1, 1.4, "z_error_comparison_n57_n115_jz_1.1_1.4.pdf"),
        ("high", 1.6, 6.0, "z_error_comparison_n57_n115_jz_1.6_6.0.pdf"),
    ]
    
    for range_name, jz_min, jz_max, filename in jz_ranges:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        
        for n, data, marker, label in [(57, data_57, 'v', r'$n=57$'), (115, data_115, 's', r'$n=115$')]:
            jzs = sorted([jz for jz in data['valid_jzs'] if jz in data['z_skqd']])
            
            # Filter by Jz range
            if jz_min is not None and jz_max is not None:
                jzs = [jz for jz in jzs if jz_min <= jz <= jz_max]
            
            if not jzs:
                continue
            
            avg_rel_diffs = []
            std_rel_diffs = []
            
            for jz in jzs:
                z_skqd = data['z_skqd'][jz]
                z_dmrg = data['z_dmrg'][jz]
                denom = np.where(z_dmrg != 0, z_dmrg, np.nan)
                rel_diff = np.abs((z_dmrg - z_skqd) / denom) * 100
                avg_rel_diffs.append(float(np.nanmean(rel_diff)))
                std_rel_diffs.append(float(np.nanstd(rel_diff)))
            
            ax.errorbar(jzs, avg_rel_diffs, yerr=std_rel_diffs, fmt=marker,
                       capsize=3, capthick=0.8, label=label, markersize=4,
                       linewidth=1.2, markerfacecolor='none', markeredgewidth=1.0)
        
        ax.set_xlabel(r"$J_z$")
        ax.set_ylabel(r"$\Delta Z$ (\%)")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(fig_dir, filename.replace('.pdf', '.png')), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        
        print(f"Saved Z observable comparison plot ({range_name} range)")

def plot_correlator_observables(data_57, data_115, output_dir, obs_type='zz'):
    """Plot ZZ correlator observables for both systems with 3 Jz ranges."""
    fig_dir = os.path.join(output_dir, f"{obs_type}_basis_opt")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    
    skqd_key = f'{obs_type}_skqd'
    dmrg_key = f'{obs_type}_dmrg'
    pairs_skqd_key = f'{obs_type}_pairs_skqd'
    pairs_dmrg_key = f'{obs_type}_pairs_dmrg'
    
    # Define Jz ranges: full, low (1.1-1.4), high (1.6-6.0)
    jz_ranges = [
        ("full", None, None, f"{obs_type}_error_comparison_n57_n115.pdf"),
        ("low", 1.1, 1.4, f"{obs_type}_error_comparison_n57_n115_jz_1.1_1.4.pdf"),
        ("high", 1.6, 6.0, f"{obs_type}_error_comparison_n57_n115_jz_1.6_6.0.pdf"),
    ]
    
    for range_name, jz_min, jz_max, filename in jz_ranges:
        fig, ax = plt.subplots(figsize=(3.5, 1.8))
        
        for n, data, marker, label in [(57, data_57, 'v', r'$n=57$'), (115, data_115, 's', r'$n=115$')]:
            jzs = sorted([jz for jz in data['valid_jzs'] if jz in data[skqd_key]])
            
            # Filter by Jz range
            if jz_min is not None and jz_max is not None:
                jzs = [jz for jz in jzs if jz_min <= jz <= jz_max]
            
            if not jzs:
                continue
            
            avg_rel_diffs = []
            std_rel_diffs = []
            
            for jz in jzs:
                corrs_skqd = data[skqd_key][jz]
                corrs_dmrg = data[dmrg_key][jz]
                pairs_skqd = data[pairs_skqd_key][jz]
                pairs_dmrg = data[pairs_dmrg_key][jz]
                
                # Convert to matrices (like the original script does)
                corr_matrix_skqd = corrs_to_matrix(corrs_skqd, pairs_skqd)
                corr_matrix_dmrg = corrs_to_matrix(corrs_dmrg, pairs_dmrg)
                
                # Compute relative error (matching original script exactly)
                rel_diff_matrix = np.abs(
                    (corr_matrix_dmrg - corr_matrix_skqd) / corr_matrix_dmrg
                ) * 100
                
                avg_rel_diffs.append(float(np.nanmean(rel_diff_matrix)))
                std_rel_diffs.append(float(np.nanstd(rel_diff_matrix)))
            
            ax.errorbar(jzs, avg_rel_diffs, yerr=std_rel_diffs, fmt=marker,
                       capsize=3, capthick=0.8, label=label, markersize=4,
                       linewidth=1.2, markerfacecolor='none', markeredgewidth=1.0)
        
        ax.set_xlabel(r"$J_z$")
        ax.set_ylabel(r"$\Delta C_{ZZ}$ (\%)")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(fig_dir, filename.replace('.pdf', '.png')), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        
        print(f"Saved {obs_type.upper()} correlator comparison plot ({range_name} range)")

def plot_loop_observables(data_57, data_115, output_dir, obs_type='z_loop'):
    """Plot Z loop observables for both systems with 3 Jz ranges."""
    loop_label = "Z"
    fig_dir = os.path.join(output_dir, f"{obs_type}_basis_opt")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    
    skqd_key = f'{obs_type}_skqd'
    dmrg_key = f'{obs_type}_dmrg'
    
    # Define Jz ranges: full, low (1.1-1.4), high (1.6-6.0)
    jz_ranges = [
        ("full", None, None, f"{obs_type}_error_comparison_n57_n115.pdf"),
        ("low", 1.1, 1.4, f"{obs_type}_error_comparison_n57_n115_jz_1.1_1.4.pdf"),
        ("high", 1.6, 6.0, f"{obs_type}_error_comparison_n57_n115_jz_1.6_6.0.pdf"),
    ]
    
    for range_name, jz_min, jz_max, filename in jz_ranges:
        fig, ax = plt.subplots(figsize=(3.5, 1.8))
        
        for n, data, marker, label in [(57, data_57, 'v', r'$n=57$'), (115, data_115, 's', r'$n=115$')]:
            jzs = sorted([jz for jz in data['valid_jzs'] if jz in data[skqd_key]])
            
            # Filter by Jz range
            if jz_min is not None and jz_max is not None:
                jzs = [jz for jz in jzs if jz_min <= jz <= jz_max]
            
            if not jzs:
                continue
            
            avg_rel_errors = []
            std_rel_errors = []
            
            for jz in jzs:
                corrs_skqd = np.array(data[skqd_key][jz])
                corrs_dmrg = np.array(data[dmrg_key][jz])
                
                # Compute relative error as percentage
                eps = 1e-12
                denom = np.maximum(np.abs(corrs_dmrg), eps)
                rel_err = np.abs(corrs_skqd - corrs_dmrg) / denom * 100
                
                avg_rel_errors.append(float(np.mean(rel_err)))
                n_loops = len(corrs_skqd)
                stderr = (np.std(rel_err, ddof=1) / np.sqrt(n_loops)) if n_loops > 1 else 0
                std_rel_errors.append(float(stderr))
            
            ax.errorbar(jzs, avg_rel_errors, yerr=std_rel_errors, fmt=marker,
                       capsize=3, capthick=0.8, label=label, markersize=4,
                       linewidth=1.2, markerfacecolor='none', markeredgewidth=1.0)
        
        ax.set_xlabel(r"$J_z$")
        ax.set_ylabel(r"$\Delta Z_{\mathrm{loop}}$ (\%)")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(fig_dir, filename.replace('.pdf', '.png')), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        
        print(f"Saved {obs_type} comparison plot ({range_name} range)")

def plot_staggered_magnetization(data_57, data_115, output_dir):
    """Plot staggered magnetization raw values for both systems with 3 Jz ranges."""
    fig_dir = os.path.join(output_dir, "staggered_magnetization")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    
    # Define Jz ranges: full, low (1.1-1.4), high (1.6-6.0)
    jz_ranges = [
        ("full", None, None, "staggered_mag_comparison_n57_n115.pdf"),
        ("low", 1.1, 1.4, "staggered_mag_comparison_n57_n115_jz_1.1_1.4.pdf"),
        ("high", 1.6, 6.0, "staggered_mag_comparison_n57_n115_jz_1.6_6.0.pdf"),
    ]
    
    for range_name, jz_min, jz_max, filename in jz_ranges:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        
        # Plot for each system size
        for n, data, marker_skqd, marker_dmrg, color_idx in [
            (57, data_57, 'v', '^', 0),
            (115, data_115, 's', 'o', 1)
        ]:
            jzs = sorted([jz for jz in data['valid_jzs'] if jz in data['ms_skqd']])
            
            # Filter by Jz range
            if jz_min is not None and jz_max is not None:
                jzs = [jz for jz in jzs if jz_min <= jz <= jz_max]
            
            if not jzs:
                continue
            
            ms_skqd_arr = np.array([data['ms_skqd'][jz] for jz in jzs])
            ms_dmrg_arr = np.array([data['ms_dmrg'][jz] for jz in jzs])
            
            # Get color from default color cycle
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            color = colors[color_idx]
            
            # Plot SKQD and DMRG with different markers but same color
            ax.plot(jzs, ms_skqd_arr, marker=marker_skqd, linestyle='-',
                   label=rf'$n={n}$ SKQD', markersize=4, linewidth=1.2,
                   color=color, markerfacecolor='none', markeredgewidth=1.0, alpha=0.7)
            ax.plot(jzs, ms_dmrg_arr, marker=marker_dmrg, linestyle='--',
                   label=rf'$n={n}$ DMRG', markersize=4, linewidth=1.2,
                   color=color, markerfacecolor='none', markeredgewidth=1.0)
        
        ax.set_xlabel(r"$J_z$")
        ax.set_ylabel(r"$m_s = \frac{1}{N}\sum_i \eta_i\langle Z_i\rangle$")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0,
                 fontsize=7, ncol=2)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5, which='both')
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(os.path.join(fig_dir, filename.replace('.pdf', '.png')), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        
        print(f"Saved staggered magnetization comparison plot ({range_name} range)")

# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("Unified Basis-Optimized Observable Analysis")
    print("Comparing n=57 (chi=80) and n=115 (chi=320) against DMRG")
    print("="*80)
    
    # Load data for both systems
    print("\nLoading data for n=57...")
    data_57 = load_data_for_system(57, DATASETS[57], JZS)
    
    print("\nLoading data for n=115...")
    data_115 = load_data_for_system(115, DATASETS[115], JZS)
    
    # Print mean and median errors across Jz for each observable
    print("\n" + "="*80)
    print("Mean and Median Errors Across Jz")
    print("="*80)
    
    for n, data in [(57, data_57), (115, data_115)]:
        print(f"\n--- n={n} ---")
        
        # Z observable
        if data['z_skqd']:
            jzs = sorted([jz for jz in data['valid_jzs'] if jz in data['z_skqd']])
            errors = []
            for jz in jzs:
                z_skqd = data['z_skqd'][jz]
                z_dmrg = data['z_dmrg'][jz]
                denom = np.where(z_dmrg != 0, z_dmrg, np.nan)
                rel_diff = np.abs((z_dmrg - z_skqd) / denom) * 100
                errors.append(float(np.nanmean(rel_diff)))
            print(f"Z:      Mean = {np.mean(errors):.3f}%, Median = {np.median(errors):.3f}%")
        
        # ZZ correlators
        if data['zz_skqd']:
            jzs = sorted([jz for jz in data['valid_jzs'] if jz in data['zz_skqd']])
            errors = []
            for jz in jzs:
                corrs_skqd = data['zz_skqd'][jz]
                corrs_dmrg = data['zz_dmrg'][jz]
                pairs_skqd = data['zz_pairs_skqd'][jz]
                pairs_dmrg = data['zz_pairs_dmrg'][jz]
                corr_matrix_skqd = corrs_to_matrix(corrs_skqd, pairs_skqd)
                corr_matrix_dmrg = corrs_to_matrix(corrs_dmrg, pairs_dmrg)
                rel_diff_matrix = np.abs((corr_matrix_dmrg - corr_matrix_skqd) / corr_matrix_dmrg) * 100
                errors.append(float(np.nanmean(rel_diff_matrix)))
            print(f"ZZ:     Mean = {np.mean(errors):.3f}%, Median = {np.median(errors):.3f}%")
        
        # Z loop
        if data['z_loop_skqd']:
            jzs = sorted([jz for jz in data['valid_jzs'] if jz in data['z_loop_skqd']])
            errors = []
            for jz in jzs:
                corrs_skqd = np.array(data['z_loop_skqd'][jz])
                corrs_dmrg = np.array(data['z_loop_dmrg'][jz])
                eps = 1e-12
                denom = np.maximum(np.abs(corrs_dmrg), eps)
                rel_err = np.abs(corrs_skqd - corrs_dmrg) / denom * 100
                errors.append(float(np.mean(rel_err)))
            print(f"Z_loop: Mean = {np.mean(errors):.3f}%, Median = {np.median(errors):.3f}%")
        
        # Staggered magnetization
        if data['ms_skqd']:
            jzs = sorted([jz for jz in data['valid_jzs'] if jz in data['ms_skqd']])
            errors = []
            for jz in jzs:
                ms_skqd = data['ms_skqd'][jz]
                ms_dmrg = data['ms_dmrg'][jz]
                rel_err = np.abs((ms_dmrg - ms_skqd) / ms_dmrg) * 100 if ms_dmrg != 0 else 0
                errors.append(float(rel_err))
            print(f"m_s:    Mean = {np.mean(errors):.3f}%, Median = {np.median(errors):.3f}%")
    
    # Create output directory
    output_dir = "plots/unified_obs_comparison"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots (3 Jz ranges each: full, 1.1-1.4, 1.6-6.0)
    print("\n" + "="*80)
    print("Generating comparison plots...")
    print("="*80)
    
    print("\n1. Z observables (3 Jz ranges)...")
    plot_z_observables(data_57, data_115, output_dir)
    
    print("\n2. ZZ correlators (3 Jz ranges)...")
    plot_correlator_observables(data_57, data_115, output_dir, obs_type='zz')
    
    print("\n3. Z loop observables (3 Jz ranges)...")
    plot_loop_observables(data_57, data_115, output_dir, obs_type='z_loop')
    
    print("\n4. Staggered magnetization (3 Jz ranges)...")
    plot_staggered_magnetization(data_57, data_115, output_dir)
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("Each observable has 3 plots: full range, Jz=1.1-1.4, Jz=1.6-6.0")
    print("="*80)

if __name__ == "__main__":
    main()
