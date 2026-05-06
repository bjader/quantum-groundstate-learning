import os
from collections import defaultdict
from typing import Dict, List, Callable, Any, Counter

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from qiskit.transpiler import CouplingMap

from qaml.observables.correlation_function import get_n_nearest_neighbours
from qaml.observables.utils import state_dict2array, array2state_dict
from qaml.analysis.plot_utils import plot_bar


def analyze_overlapping_bitstrings(save_dir: str, gs_dict_truncated_dict):
    bitstring_occurrences = defaultdict(set)
    for dict_name, bit_amplitudes in gs_dict_truncated_dict.items():
        for bitstring in bit_amplitudes.keys():
            bitstring_occurrences[bitstring].add(dict_name)
    bitstring_counts = {
        bitstring: len(dsets) for bitstring, dsets in bitstring_occurrences.items()
    }
    histogram_data = Counter(bitstring_counts.values())
    x = sorted(histogram_data.keys())
    y = [histogram_data[n] for n in x]
    plot_bar(
        x,
        y,
        xlabel=r"Number of $J_{z}$ values the bitstrings were sampled from",
        ylabel="Count of Bitstrings",
        save_path=os.path.join(
            save_dir, "histogram_number_of_overlapping_bitstrings.pdf"
        ),
    )


def truncate_state_cumulative(
    state_dict: Dict[str, complex], cum_prob_amp_threshold: float, num_spins: int
) -> Dict[str, complex]:
    """Truncate the quantum state by selecting bitstrings whose probability amplitudes sum up to more than the threshold in the order of their significance.

    Args:
        state_dict (Dict[str, complex]): A quantum state represented by a dictionary mapping bitstrings to amplitudes (Z-basis).
        cum_prob_amp_threshold (float): Cumulative probability amplitudes threshold, e.g. 0.7.
        num_spins (int): number of spins.

    Returns:
        Dict[str, complex]: State dictionary that consist of significant bitstrings.
    """
    state_array = state_dict2array(
        state_dict=state_dict, size_support=len(state_dict.keys())
    )  # Sorts the bitstrings by their prob amplitudes when constructing the array.
    idx = len(state_dict.keys())  # Use all bitstrings.
    if cum_prob_amp_threshold < 1.0:
        # Obtain only the significant bitstrings.
        idx_query = np.where(
            np.cumsum(np.abs(state_array[:, 1])**2) >= cum_prob_amp_threshold
        )
        idx = idx_query[0][0] + 1

    state_array_significant = state_array[:idx, :]
    state_dict_significant = array2state_dict(state_array_significant, num_spins)
    return state_dict_significant

def truncate_state_top_M(
    state_dict: Dict[str, complex], M:int, num_spins: int
) -> Dict[str, complex]:
    """Truncate the quantum state by selecting bitstrings 

    Args:
        state_dict (Dict[str, complex]): A quantum state represented by a dictionary mapping bitstrings to amplitudes (Z-basis).
        M (int): The number of significant bitstrings.
        num_spins (int): Number of spins.

    Returns:
        Dict[str, complex]: State dictionary that consist of significant bitstrings.
    """
    
    state_array = state_dict2array(
            state_dict=state_dict, size_support=len(state_dict.keys()) 
    )
    state_array_significant = state_array[:M, :]
    state_dict_significant = array2state_dict(state_array_significant, num_spins)
    return state_dict_significant

def calc_properties(
    state_dict: Dict[str, complex],
    prop_funcs: Dict[str, Callable],
    prop_kwargs: Dict[str, Dict[str, Any]],
) -> Dict[str, float | List[float]]:
    """Calculate the properties of a quantum state specified by the state dictionary.

    Args:
        state_dict (Dict[str, complex]): A quantum state represented by a dictionary mapping bitstrings to amplitudes (Z-basis).
        prop_funcs (Dict[str, Callable]): Functions to calculate the properties.
        prop_kwargs (Dict[str, Dict[str, Any]]): Keyword arguments for the property functions.

    Returns:
        Dict[str, float|List[float]]: Calculated properties.
    """

    props = {}
    for k, func in prop_funcs.items():
        if k in prop_kwargs:
            props[k] = func(state_dict, **prop_kwargs[k])
        else:
            props[k] = func(state_dict)

    return props

def two_body_corr_funcs2square_array(two_body_corr_funcs: np.ndarray):
    num_spins = int((1 + np.sqrt(1 + 8 * two_body_corr_funcs.shape[0])) // 2) 
    square_array = np.zeros((num_spins, num_spins))
    tri_indices = np.triu_indices(num_spins, k=1) # k=1 excludes diagonal
    square_array[tri_indices] = two_body_corr_funcs
    square_array += square_array.T 
    np.fill_diagonal(square_array, 1.)
    return square_array

def n_nearest_neighbor_corr_funcs2square_array(n_nearest_neighbor_corr_funcs, n_spins, n_neighbor, fill_diagonal: bool=False):
    square_array = np.full((n_spins, n_spins), np.nan)
    if fill_diagonal:
        np.fill_diagonal(square_array, 1.0)
    idx = 0
    for i in range(n_spins):
        for k in range(1, n_neighbor):
            j = i + k
            if j < n_spins:
                val = n_nearest_neighbor_corr_funcs[idx]
                square_array[i, j] = val
                square_array[j, i] = val # symmetric
                idx += 1
    return square_array


def corrs_to_matrix(corrs, pairs):
    n = np.max(pairs) + 1
    corr_matrix_1 = np.full((n, n), np.nan)
    for i, val in enumerate(corrs):
        q1 = pairs[i][0]
        q2 = pairs[i][1]
        corr_matrix_1[q1][q2] = val
    return corr_matrix_1


def draw_graph(n, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = graphviz_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=220)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    nx.draw_networkx_labels(G, pos, labels={i: i for i in range(n)}, font_size=7)
    plt.show()
    plt.close()


def filter_to_less_neighbours(pairs, corrs, nns, heavy_hex_distance=7):
    allowed_pairs = get_n_nearest_neighbours(CouplingMap.from_heavy_hex(heavy_hex_distance).get_edges(), nns)
    filtered_pairs = []
    filtered_corrs = []
    for i, pair in enumerate(pairs):
        if pair in allowed_pairs or pair[::-1] in allowed_pairs:
            filtered_pairs.append(pair)
            filtered_corrs.append(corrs[i])

    return filtered_pairs, filtered_corrs
