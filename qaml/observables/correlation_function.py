from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import networkx as nx
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.qubit import project_operator_to_subspace

from qaml.skqd.basis_optimise import obp_modified_hamiltonian
from .utils import flip_bits
import numpy as np

def corr_single_pair(state_dict: Dict[str, complex], i: int, j: int) -> float:
    """Calculates the two-body correlation function, i.e. 1/3(X_iX_j + Y_iY_j + Z_iZ_j), for a single pair, i.e. two specified indices.

    Args:
        state_dict (Dict[str, complex]): A quantum state. Dict mapping bitstrings to amplitudes (Z-basis).
        i (int): Index of the first spin.
        j (int): Index of the second spin.

    Returns:
        float: The expectation value of the two-body correlation function for the chosen i and j.
    """

    val = 0.0
    for bitstring, amp in state_dict.items():
        # Z ⊗ Z
        zi = 1 if bitstring[i] == "0" else -1
        zj = 1 if bitstring[j] == "0" else -1
        val += abs(amp)**2 * zi * zj

        # X ⊗ X and Y ⊗ Y
        sign = 1 if bitstring[i] != bitstring[j] else -1  # phase factor from Y_i Y_j
        flipped = flip_bits(bitstring, i, j)
        if flipped in state_dict:
            amp2 = state_dict[flipped]
            val += (1 + sign) * (amp.conjugate() * amp2).real

    val /= 3
    return val.real

def corr_single_pair_modified_basis(state_dict, i, j, optimized_circuit, zz_only):
    bs_matrix, _ = counts_to_arrays(state_dict)
    betas = np.array(list(state_dict.values()))

    n = len(list(state_dict.keys())[0])
    if zz_only:
        obs = SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1)], num_qubits=n)
    else:
        obs = SparsePauliOp.from_sparse_list([("XX", [i, j], 1/3), ("YY", [i, j], 1/3), ("ZZ", [i, j], 1/3)], num_qubits=n)

    zz_tilde = obp_modified_hamiltonian(obs , optimized_circuit)

    zz_proj = project_operator_to_subspace(bs_matrix, zz_tilde)

    # 5. Compute <psi| zz_proj |psi>
    psi = betas.astype(complex)
    Hpsi = zz_proj @ psi
    val = np.vdot(psi, Hpsi)

    return val.real

def zz_corr_single_pair(state_dict, i, j):
    val = 0.0
    for bitstring, amp in state_dict.items():
        # Z ⊗ Z
        zi = 1 if bitstring[i] == "0" else -1
        zj = 1 if bitstring[j] == "0" else -1
        val += abs(amp)**2 * zi * zj

    return val.real

def xx_corr_single_pair(state_dict, i, j):
    val = 0.0
    for bitstring, amp in state_dict.items():

        flipped = flip_bits(bitstring, i, j)
        amp2 = state_dict.get(flipped)
        if amp2 is not None:
            val += (amp.conjugate() * amp2).real

    return val.real

# TODO Rename all neighbor to neighbour
def n_nearest_neighbor_corr_1d(
    state_dict: Dict[str, complex], n_neighbor: int
) -> np.ndarray:
    """Calculates n nearest neighbor correlation functions in 1D, i.e. 1/3(X_iX_j + Y_iY_j + Z_iZ_j).

    Args:
        state_dict (Dict[str, complex]): A quantum state. Dict mapping bitstrings to amplitudes (Z-basis).
        n_neighbor (int): Number of neighbors to calculate the correlation functions for.

    Returns:
        np.ndarray: List of expectation values of all n nearest neighbor correlation functionsin 1D.
    """
    n = len(next(iter(state_dict)))
    corr = []
    for i in range(n - 1):
        upper_bound = min(i + n_neighbor, n)
        for j in range(i + 1, upper_bound):
            # Correlation function between two nearest neighbors.
            value = corr_single_pair(state_dict, i, j)
            corr.append(value)
    return np.array(corr)


def n_nearest_neighbor_corr_nd(
    state_dict: Dict[str, complex], edges, n_neighbor: int, xx_only=True,
) -> Tuple[np.ndarray, list]:
    corr_func_pairs = get_n_nearest_neighbours(edges, n_neighbor)

    corr = []
    for i, pair in enumerate(corr_func_pairs):
        print(f"Computing pair {i+1} of {len(corr_func_pairs)}")
        n = len(list(state_dict.keys())[0])
        # q0 is the rightmost bit in Qiskit
        i, j = n - pair[0] - 1, n - pair[1] - 1
        if xx_only:
            value = xx_corr_single_pair(state_dict, i, j)
        else:
            value = corr_single_pair(state_dict, i, j)
        corr.append(value)

    return np.array(corr), corr_func_pairs

def n_nearest_neighbor_corr_nd_modified_basis(
    state_dict: Dict[str, complex], edges, n_neighbor: int, opt_qc, zz_only=False,
) -> Tuple[np.ndarray, list]:
    corr_func_pairs = get_n_nearest_neighbours(edges, n_neighbor)

    corr = []
    for i, pair in enumerate(corr_func_pairs):
        print(f"Computing pair {i+1} of {len(corr_func_pairs)}")
        value = corr_single_pair_modified_basis(state_dict, i, j, opt_qc, zz_only)
        corr.append(value)

    return np.array(corr), corr_func_pairs

def _compute_corr_pair_executor(pair, state_dict, zz_only, opt_qc):
    i = pair[0]
    j= pair[1]
    return corr_single_pair_modified_basis(state_dict, i, j, opt_qc, zz_only)

def n_nearest_neighbor_corr_nd_modified_basis_parallel(
    state_dict: Dict[str, complex],
    edges,
    n_neighbor: int,
    opt_qc,
    zz_only: bool = False,
    max_workers: int | None = None,
    chunksize: int = 8,
    show_progress: bool = True,
):

    corr_func_pairs = get_n_nearest_neighbours(edges, n_neighbor)

    if max_workers is None:
        max_workers = cpu_count()

    corr: List[complex] = []
    total = len(corr_func_pairs)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:

        fn = partial(_compute_corr_pair_executor, state_dict=state_dict, zz_only=zz_only, opt_qc=opt_qc)
        mapped = ex.map(fn, corr_func_pairs, chunksize=chunksize)

        if show_progress:
            for k, value in enumerate(mapped, start=1):
                print(f"Computing pair {k} of {total}")
                corr.append(value)
        else:
            corr = list(mapped)

    return np.array(corr), corr_func_pairs


def get_n_nearest_neighbours(edges, n_neighbor):
    g = nx.Graph()
    g.add_edges_from(edges)
    neighbour_pairs = []
    for n in sorted(g.nodes()):
        # For this node, first the n-nearest neighbour sites
        dists = nx.single_source_shortest_path_length(g, n, cutoff=n_neighbor)
        for distance in range(1, n_neighbor + 1):
            # Append nearest neighbour, then next-nearest neighbour, then...
            neighbor_nodes = [v for v, d in dists.items() if d == distance]
            neighbour_pairs.extend((n, v) for v in sorted(neighbor_nodes))
    # Remove duplicates that are just the same pair reversed
    neighbour_pairs = [list(s) for s in set([frozenset(item) for item in neighbour_pairs])]
    return neighbour_pairs

def all_k_cycles_undirected(edges, k=12):
    g = nx.Graph()
    g.add_edges_from(edges)

    adj = {u: tuple(sorted(g.neighbors(u))) for u in g.nodes()}
    cycles = set()

    def canonical_cycle(cyc):
        m = min(cyc)
        i = cyc.index(m)
        r1 = cyc[i:] + cyc[:i]
        rev = list(reversed(cyc))
        j = rev.index(m)
        r2 = rev[j:] + rev[:j]
        t1 = tuple(r1)
        t2 = tuple(r2)
        return t1 if t1 < t2 else t2

    for s in sorted(g.nodes()):
        stack = [(s, [s])]
        while stack:
            v, path = stack.pop()

            if len(path) == k:
                if s in adj[v]:
                    cycles.add(canonical_cycle(path))
                continue

            for w in adj[v]:
                if w == s:
                    continue
                if w in path:
                    continue
                if w < s:
                    continue
                stack.append((w, path + [w]))

    return [list(c) for c in sorted(cycles)]
