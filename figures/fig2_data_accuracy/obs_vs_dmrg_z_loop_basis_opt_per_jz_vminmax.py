"""
Similar plot to obs_vs_dmrg_z_loop_basis_opt.py but with a tighter colorbar range. Used for
bottom panel of Figure 2.
"""

import os
import sys
from pathlib import Path

import dill as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from networkx.drawing.nx_pydot import graphviz_layout

# Reduced DMRG data are plain dicts with precomputed z_loop, so no qaml unpickling shims are needed.
from qaml.graph.graph_utils import read_edges_txt

def plot_loops_heavy_hex(
    edges,
    n,
    loops,
    loop_vals,
    cmap=plt.cm.viridis,
    vmin=None,
    vmax=None,
    alpha=0.60,
    title=None,
    cbar_label=r"$\langle Z^{\otimes 12}\rangle$",
    show_labels=True,
    scale="linear",  # "linear" or "symlog"
    linthresh=1e-3,  # only used for symlog
    base=10,
):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    try:
        pos = graphviz_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(constrained_layout=False)

    vals = np.asarray(loop_vals, dtype=float)
    maxabs = float(np.nanmax(np.abs(vals))) if np.any(~np.isnan(vals)) else 1.0
    maxabs = max(maxabs, 1e-12)

    if vmin is None:
        vmin = -maxabs
    if vmax is None:
        vmax = +maxabs

    if scale == "symlog":
        norm = mpl.colors.SymLogNorm(
            linthresh=linthresh,
            vmin=vmin,
            vmax=vmax,
            base=base,
        )
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for loop, val in zip(loops, vals):
        if np.isnan(val):
            continue
        pts = np.array([pos[int(v)] for v in loop], dtype=float)
        poly = Polygon(
            pts,
            closed=True,
            facecolor=sm.to_rgba(val),
            edgecolor="none",
            alpha=alpha,
        )
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
        text_dict = nx.draw_networkx_labels(G, pos, labels={i: i for i in range(n)}, font_size=5,
                                            ax=ax)
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

def compute_dmrg_x_loop_if_missing(
    data_dmrg,
    dmrg_result,
    loops_old,
    key="x_loop",
):
    if key in data_dmrg and f"{key}_sites" in data_dmrg:
        return np.asarray(data_dmrg[key]), list(data_dmrg[f"{key}_sites"])

    print("No DMRG loop data found, computing...")

    psi = dmrg_result.psi
    n = psi.L
    order = dmrg_result.bfs_order  # order[new_chain] = old_label

    pos = np.empty(n, dtype=int)  # pos[old_label] = new_chain
    for new, old in enumerate(order):
        pos[int(old)] = int(new)

    corrs = []
    scale = 2**len(loops_old[0])  # Sx = X/2 => X^⊗k = 2^k (Sx)^⊗k

    for loop in loops_old:
        loop_chain = [int(pos[i]) for i in loop]
        loop_chain.sort()
        term = [("Sz", i) for i in loop_chain]
        val = psi.expectation_value_term(term) * scale
        corrs.append(val)

    corrs = np.asarray(corrs)
    data_dmrg[key] = corrs
    data_dmrg[f"{key}_sites"] = loops_old

    print("max |loop| =", float(np.max(np.abs(corrs))))
    return corrs, loops_old

n = 115

dir_skqd = "recovery_random_flip"
root_dir_skqd = f"../../data/spins_{n}/skqd/{dir_skqd}"

chi_max = 320
trunc_cut = 1e-6
dmrg_nn = 2
root_dir_dmrg = f"../../data/spins_{n}/dmrg/chi_max_{chi_max:d}_trunc_{trunc_cut:.0e}_reduced"

jzs = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]
jzs = [1.6, 3.0, 4.0]

if n == 115:
    edges_path = "../../cmaps/heavy_hex_edges_d_7.txt"
elif n == 57:
    edges_path = "../../cmaps/heavy_hex_edges_d_5.txt"
edges, n_spins = read_edges_txt(edges_path)

corrs_skqd_per_jz = {}
corrs_dmrg_per_jz = {}
loops_per_jz = {}

valid_jzs = []
for jz in list(jzs):
    print("Jz =", jz)

    try:
        with open(os.path.join(root_dir_skqd, f"XXZ_2d_jz_{jz:.1f}.pkl"), "rb") as f:
            data_skqd = pickle.load(f)
        corrs_skqd = np.asarray(data_skqd["z_loop"])
        sites_skqd = list(data_skqd["z_loop_sites"])
    except FileNotFoundError:
        print(f"No SKQD data file found, skipping Jz {jz}")
        continue
    except KeyError:
        print(f"No SKQD loop data found, skipping Jz {jz}")
        continue

    with open(os.path.join(root_dir_dmrg, f"XXZ_2d_jz_{jz:.1f}.pkl"), "rb") as f:
        data_dmrg = pickle.load(f)

    # Reduced DMRG files always carry precomputed z_loop / z_loop_sites.
    key = "z_loop"
    corrs_dmrg = np.asarray(data_dmrg[key])
    sites_dmrg = list(data_dmrg[f"{key}_sites"])

    if sites_dmrg != sites_skqd:
        print("Warning: sites_dmrg != sites_skqd (not aligning, as requested).")

    corrs_skqd_per_jz[jz] = corrs_skqd
    corrs_dmrg_per_jz[jz] = corrs_dmrg
    loops_per_jz[jz] = sites_skqd
    valid_jzs.append(jz)

if len(valid_jzs) == 0:
    print("No Jz values with data found. Exiting.")
    exit()

valid_jzs = sorted(valid_jzs)

fig_dir = f"plots/spins_{n}_skqd/{dir_skqd}/loop_shading_z12_per_jz_vminmax/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)

# Use a non-diverging colormap to avoid misleading zero representation
cmap = plt.cm.PuBu

for jz in valid_jzs:
    loops = loops_per_jz[jz]
    
    # Compute per-Jz vmin and vmax from combined DMRG and SKQD data
    combined_vals = np.concatenate([
        np.asarray(corrs_skqd_per_jz[jz], float),
        np.asarray(corrs_dmrg_per_jz[jz], float)
    ])
    # Filter out NaN values
    combined_vals = combined_vals[~np.isnan(combined_vals)]
    
    if len(combined_vals) == 0:
        print(f"Warning: No valid data for Jz={jz}, skipping")
        continue

    minval = float(np.min(combined_vals))
    maxval = float(np.max(combined_vals))
    vmin_jz = minval - (maxval - minval) * 1.25
    vmax_jz = maxval + (maxval - minval) * 1.25
    
    # Ensure we have a valid range
    if vmax_jz - vmin_jz < 1e-12:
        vmin_jz = vmin_jz - 1e-12
        vmax_jz = vmax_jz + 1e-12
    
    print(f"Jz={jz:.1f}: vmin={vmin_jz:.6f}, vmax={vmax_jz:.6f}")

    fig = plot_loops_heavy_hex(
        edges=edges,
        n=n,
        loops=loops,
        loop_vals=corrs_skqd_per_jz[jz],
        cmap=cmap,
        vmin=vmin_jz,
        vmax=vmax_jz,
        scale="linear",
        alpha=0.60,
        title=fr"SKQD loop correlators ($J_z={jz:.1f}$)",
        cbar_label=r"$\langle Z^{\otimes 12}\rangle$",
        show_labels=True,
    )
    fig.savefig(os.path.join(fig_dir, f"loop_shading_skqd_jz_{jz:.1f}.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig = plot_loops_heavy_hex(
        edges=edges,
        n=n,
        loops=loops,
        loop_vals=corrs_dmrg_per_jz[jz],
        cmap=cmap,
        vmin=vmin_jz,
        vmax=vmax_jz,
        scale="linear",
        alpha=0.60,
        title=fr"DMRG loop correlators ($J_z={jz:.1f}$)",
        cbar_label=r"$\langle Z^{\otimes 12}\rangle$",
        show_labels=True,
    )
    fig.savefig(os.path.join(fig_dir, f"loop_shading_dmrg_jz_{jz:.1f}.pdf"), bbox_inches="tight")
    plt.close(fig)

jz_list = valid_jzs
n_loops = len(loops_per_jz[jz_list[0]])

skqd_mat = np.vstack([corrs_skqd_per_jz[jz] for jz in jz_list])
dmrg_mat = np.vstack([corrs_dmrg_per_jz[jz] for jz in jz_list])

colors = plt.cm.tab20(np.linspace(0, 1, max(n_loops, 1)))

plt.figure()
for ell in range(n_loops):
    plt.plot(jz_list, skqd_mat[:, ell], color=colors[ell % len(colors)], linewidth=1.7)
    plt.plot(jz_list, dmrg_mat[:, ell], color=colors[ell % len(colors)], linewidth=1.7,
             linestyle="--")

plt.plot([], [], color="k", linewidth=1.7, label="SKQD")
plt.plot([], [], color="k", linewidth=1.7, linestyle="--", label="DMRG")

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"$J_z$", fontsize=14)
plt.ylabel(r"$\langle Z^{\otimes 12}\rangle$", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "z12_loop_corrs_over_jz_skqd_vs_dmrg.pdf"))
plt.close()

# Relative error matrix: shape (len(jz_list), n_loops)
eps = 1e-12
denom = np.maximum(np.abs(dmrg_mat), eps)
rel_err_mat = np.abs(skqd_mat - dmrg_mat) / denom

plt.figure()
for ell in range(n_loops):
    plt.plot(
        jz_list,
        rel_err_mat[:, ell],
        linestyle="None",
        marker="x",
        markersize=5,
        color=colors[ell % len(colors)],
        alpha=0.9,
    )

mean_err = np.mean(rel_err_mat, axis=1)
stderr = (np.std(rel_err_mat, axis=1, ddof=1) / np.sqrt(n_loops)) if n_loops > 1 else np.zeros_like(mean_err)

plt.errorbar(
    jz_list,
    mean_err,
    yerr=stderr,
    fmt="x",
    color="k",
    markersize=7,
    capsize=3,
    linewidth=1.2,
    label="mean ± SE",
)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"$J_z$", fontsize=14)
plt.ylabel("relative error", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "relative_error_skqd_vs_dmrg_over_jz_markers_mean_se.pdf"))
plt.close()
