from __future__ import annotations

import numpy as np
import matplotlib as mpl
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from networkx.drawing.nx_pydot import graphviz_layout


class DivergingPowerNorm(Normalize):
    """
    Normalization that applies symmetric power scaling around a center point (vcenter).
    Compresses the middle range while keeping vcenter mapped to 0.5 in the colormap.
    
    For diverging colormaps (like seismic), this keeps 0.0 at white/center
    and compresses values near zero while expanding the extremes.
    
    The transformation is symmetric: |x|^gamma * sign(x)
    """
    def __init__(self, vmin=-1.0, vmax=1.0, vcenter=0.0, gamma=3.0, clip=False):
        self.vcenter = vcenter
        self.gamma = gamma
        super().__init__(vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        x = np.array(value, copy=True)
        
        # Split into positive and negative parts relative to vcenter
        mask_pos = x >= self.vcenter
        mask_neg = x < self.vcenter
        
        result = np.zeros_like(x, dtype=float)
        
        # Positive side: map [vcenter, vmax] to [0.5, 1.0]
        if np.any(mask_pos):
            x_pos = x[mask_pos]
            # Normalize to [0, 1]
            normalized = (x_pos - self.vcenter) / (self.vmax - self.vcenter)
            # Apply symmetric power transform (compress middle, expand extremes)
            powered = np.sign(normalized) * np.power(np.abs(normalized), self.gamma)
            # Map to [0.5, 1.0]
            result[mask_pos] = 0.5 + 0.5 * powered
        
        # Negative side: map [vmin, vcenter] to [0.0, 0.5]
        if np.any(mask_neg):
            x_neg = x[mask_neg]
            # Normalize to [-1, 0] then flip to [0, 1] for processing
            normalized = (x_neg - self.vcenter) / (self.vcenter - self.vmin)
            # Apply symmetric power transform
            powered = np.sign(normalized) * np.power(np.abs(normalized), self.gamma)
            # Map to [0.0, 0.5]
            result[mask_neg] = 0.5 + 0.5 * powered
        
        return np.ma.masked_array(result, mask=np.ma.getmask(value))


def plot_sz_heavy_hex(edges, n, sz_list):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = graphviz_layout(G)
    vmin, vmax = -0.5, 0.5
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=220, node_color=sz_list, vmin=vmin, vmax=vmax, cmap="coolwarm"
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    nx.draw_networkx_labels(G, pos, labels={i: i for i in range(n)}, font_size=7)
    cbar = plt.colorbar(nodes)
    cbar.set_label(r"$\langle S_z \rangle$")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_corr_heavy_hex(edges, n, corr_list, sz_list=None, cmap=plt.cm.seismic, gamma=5.0):
    """
    Plot correlations on heavy hex lattice using a custom DivergingPowerNorm colormap to assign
    most colour change within the region |obs| > 0.7.
    
    Args:
        edges: List of edge tuples
        n: Number of nodes
        corr_list: List of correlation values for edges
        sz_list: Optional list of single-site Z values for nodes
        cmap: Colormap to use (should be diverging, e.g., seismic)
        gamma: Power normalization exponent. Higher values compress the middle range more.
               Default 3.0 compresses values near 0 while expanding extremes (±0.7 to ±1.0).
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = graphviz_layout(G)

    fig, ax = plt.subplots(constrained_layout=False)

    corr_vmax = 1.0
    
    # Use DivergingPowerNorm to compress middle range around 0 while keeping
    # 0.0 at the center (white) of the diverging colormap.
    norm = DivergingPowerNorm(vmin=-corr_vmax, vmax=corr_vmax, vcenter=0.0, gamma=gamma)
    
    # Manually normalize edge colors
    corr_array = np.array(corr_list)
    normalized_colors = np.asarray(norm(corr_array))

    # Draw nodes with same normalization if sz_list provided
    if sz_list is not None:
        sz_array = np.array(sz_list)
        normalized_sz = np.asarray(norm(sz_array))
        nodes = nx.draw_networkx_nodes(
            G, pos, node_size=80, node_color=normalized_sz, vmin=0, vmax=1, cmap=cmap
        )
    else:
        nodes = nx.draw_networkx_nodes(G, pos, node_size=80)

    # Labels & edges
    nx.draw_networkx_labels(G, pos, labels={i: i for i in range(n)}, font_size=5)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges, alpha=0.5, width=1.0,
        edge_color=normalized_colors, edge_cmap=cmap, edge_vmin=0, edge_vmax=1
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Use axes_grid1 to append two separate colorbar axes
    divider = make_axes_locatable(ax)

    # Build visually uniform colorbars explicitly in axis coordinates [0, 1].
    # The bar itself is a linear ramp in colormap space; the tick labels are the
    # inverse-mapped physical values corresponding to equally spaced visual positions.
    n_ticks = 9
    tick_positions_axis = np.linspace(0.0, 1.0, n_ticks)
    signed_positions = 2.0 * tick_positions_axis - 1.0
    tick_data_values = corr_vmax * np.sign(signed_positions) * np.power(np.abs(signed_positions), 1.0 / gamma)
    colorbar_axis_values = np.linspace(0.0, 1.0, 512).reshape(-1, 1)

    # Right side for edges
    cax_right = divider.append_axes("right", size="5%", pad=0.18)
    cax_right.imshow(
        colorbar_axis_values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, 1.0, 0.0, 1.0),
    )
    cax_right.set_xticks([])
    cax_right.set_yticks(tick_positions_axis)
    cax_right.set_yticklabels([f"{v:.2f}" for v in tick_data_values])
    cax_right.set_ylabel(r"$\langle Z_iZ_j \rangle$")

    if sz_list is not None:
        cax_left = divider.append_axes("left", size="5%", pad=0.05)
        cax_left.imshow(
            colorbar_axis_values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        cax_left.set_xticks([])
        cax_left.set_yticks(tick_positions_axis)
        cax_left.set_yticklabels([f"{v:.2f}" for v in tick_data_values])
        cax_left.set_ylabel(r"$\langle Z \rangle$")
        
        cax_left.yaxis.set_ticks_position('left')
        cax_left.yaxis.set_label_position('left')

    ax.set_axis_off()
    return plt

def plot_heavy_hex_differences(
    edges, n, z_diff, zz_diff, corr_pairs,
    title=None, cmap=plt.cm.Reds, vmin_z=None, vmax_z=None,
    vmin_zz=None, vmax_zz=None
):
    """
    Plot absolute differences on heavy hex lattice with simple linear colormaps.
    
    Args:
        edges: List of edges defining the graph
        n: Number of nodes
        z_diff: Difference in <Z> values (length n)
        zz_diff: Difference in <ZZ> correlations (length = number of corr pairs)
        corr_pairs: List of correlation pairs [(i,j), ...]
        title: Plot title
        cmap: Colormap (default: Reds for absolute differences)
        vmin_z, vmax_z: Color limits for node differences (auto-scaled from data if None)
        vmin_zz, vmax_zz: Color limits for edge differences (auto-scaled from data if None)
    
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
    
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=False)
    
    # Determine color limits if not provided (auto-scale from data)
    # Use the same scale for both Z and ZZ for consistency
    if vmin_z is None or vmax_z is None or vmin_zz is None or vmax_zz is None:
        z_max = np.nanmax(np.abs(z_diff))
        zz_max = np.nanmax(np.abs(zz_diff))
        combined_max = max(z_max, zz_max)

        if vmin_z is None:
            vmin_z = 0 if np.all(z_diff >= 0) else -combined_max
        if vmax_z is None:
            vmax_z = combined_max
        if vmin_zz is None:
            vmin_zz = 0 if np.all(zz_diff >= 0) else -combined_max
        if vmax_zz is None:
            vmax_zz = combined_max
    
    # Draw nodes with <Z> differences
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=200, node_color=z_diff,
        vmin=vmin_z, vmax=vmax_z, cmap=cmap, ax=ax
    )
    
    # Draw edges with <ZZ> differences
    # Create edge color list matching the order of edges
    edge_colors = []
    for edge in edges:
        # Find this edge in corr_pairs
        edge_tuple = tuple(sorted(edge))
        try:
            idx = [tuple(sorted(pair)) for pair in corr_pairs].index(edge_tuple)
            edge_colors.append(zz_diff[idx])
        except ValueError:
            # Edge not in correlation pairs, use 0
            edge_colors.append(0.0)
    
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, alpha=0.8, width=2.0,
        edge_color=edge_colors, edge_cmap=cmap,
        edge_vmin=vmin_zz, edge_vmax=vmax_zz, ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={i: i for i in range(n)}, font_size=6, ax=ax)
    
    # Create colorbars
    divider = make_axes_locatable(ax)
    
    # Right colorbar for edges (ZZ correlations)
    cax_right = divider.append_axes("right", size="5%", pad=0.05)
    sm_edges = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin_zz, vmax=vmax_zz), cmap=cmap
    )
    sm_edges.set_array([])
    cbar_edges = fig.colorbar(sm_edges, cax=cax_right)
    cbar_edges.set_label(r"$\Delta \langle ZZ \rangle$", fontsize=12)
    
    # Left colorbar for nodes (Z values)
    cax_left = divider.append_axes("left", size="5%", pad=0.05)
    cbar_nodes = fig.colorbar(nodes, cax=cax_left)
    cbar_nodes.set_label(r"$\Delta \langle Z \rangle$", fontsize=12)
    cax_left.yaxis.set_ticks_position('left')
    cax_left.yaxis.set_label_position('left')
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    ax.set_axis_off()
    return fig

