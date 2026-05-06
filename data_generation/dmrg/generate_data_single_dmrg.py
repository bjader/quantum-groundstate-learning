import argparse
import logging
import os
from pathlib import Path

import dill as pickle
import networkx as nx
import numpy as np

from qaml.diagonalisation.twod.dmrg import run_heisenberg_2d, reindex_edges
from qaml.graph.graph_utils import read_edges_txt

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--edges_fn", default="heavy_hex_edges_d_7.txt",
                        help="Name of file with undirected edges, assuming it lives in 2d_xxz/cmaps")
    parser.add_argument("-jz", type=float, default=6.0, help="ZZ_coupling_const")
    parser.add_argument(
        "-nn",
        "--num_neighbor",
        type=int,
        default=2,
        help="Number of spins in the neighborhood for the calculations of correlation functions",
    )
    parser.add_argument("-sd", "--save_dir", type=str, default="../../data",
                        help="Save Directory")
    parser.add_argument(
        "-chi", "--chi_max", nargs="?", type=int, default=320, help="Max bond dimension"
    )
    parser.add_argument(
        "-tr", "--trunc_cut", nargs="?", type=float, default=1e-6, help="SVD truncation"
    )

    parser.add_argument(
        "--e1", action=argparse.BooleanOptionalAction, default=True, help="Calculate first excited state"
    )

    return parser.parse_args()


def n_nearest_neighbor_corr_funcs_2d(
    n_spins: int, edges: list, n_neighbor: int, jz: float, chi_max: int, trunc_cut: float):
    """Generate two body correlation functions for 2d XXZ model.

    Args:
        n_spins (int): Number of spins.
        n_neighbor (int): Distance of neighbours to calculate for per site for. E.g.
        1-> nearest neighbor, 2-> next-nearest neighbour
        jz (float): Jz coupling
    """
    # DMRG simulation of the same system.
    logger.info(f"Starting DMRG for jz: {args.jz:.1f}")
    result = run_heisenberg_2d(n_spins, edges, jz=jz, chi=chi_max, trunc_cut=trunc_cut, calculate_e1=args.e1)
    e, psi, order = result.e, result.psi, result.bfs_order
    logger.info("DMRG completed")

    logger.info("Calculating Sz")
    z = psi.expectation_value("Sz")*2

    g = nx.Graph()
    dmrg_edges = reindex_edges(edges, order)
    g.add_edges_from(dmrg_edges)
    pairs = []
    for n in sorted(g.nodes()):
        # For this node, first the n-nearest neighbour sites
        dists = nx.single_source_shortest_path_length(g, n, cutoff=n_neighbor)
        for distance in range(1, n_neighbor + 1):
            # Append nearest neighbour, then next-nearest neighbour, then...
            neighbor_nodes = [v for v, d in dists.items() if d == distance]
            pairs.extend((n, v) for v in sorted(neighbor_nodes))

    # Remove duplicates that are just the same pair reversed
    pairs = [list(s) for s in set([frozenset(item) for item in pairs])]
    xx, yy, zz = [], [], []
    logger.info("Calculating correlation functions")
    for pair in pairs:
        j, k = pair
        xx.append(4*psi.expectation_value_term([("Sx", j), ("Sx", k)]))
        yy.append(4*psi.expectation_value_term([("Sy", j), ("Sy", k)]))
        zz.append(4*psi.expectation_value_term([("Sz", j), ("Sz", k)]))

    return xx, yy, zz, z, pairs, result


if __name__ == "__main__":
    args = parse_args()
    # Try to find cmaps directory in common locations
    edges_root_dirs = ["../../cmaps", "./cmaps", "../cmaps"]
    for edge_root_dir in edges_root_dirs:
        edges_path = os.path.join(edge_root_dir, args.edges_fn)
        if os.path.exists(edges_path):
            break
    edges, n = read_edges_txt(edges_path)
    xx, yy, zz, z, corr_pairs, result = n_nearest_neighbor_corr_funcs_2d(
        n, edges, args.num_neighbor, args.jz, args.chi_max, args.trunc_cut
    )
    logger.info(f"Correlation functions completed for Jz={args.jz}")
    save_dir = os.path.join(
        args.save_dir,
        f"spins_{n:d}/dmrg/chi_max_{args.chi_max:d}_trunc_{args.trunc_cut:.0e}",
    )
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    file_path = os.path.join(save_dir, f"XXZ_2d_jz_{args.jz:.1f}.pkl")

    if os.path.isfile(file_path):
        # Open the file if there is already data stored, e.g. different observables.
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        f.close()
    else:
        data = {}

    data["dmrg_params"] = {"chi_max": args.chi_max, "trunc_cut": args.trunc_cut}

    # n nearest neighbor correlation functions
    data[f"{args.num_neighbor:d}_nn_corr_xx"] = xx
    data[f"{args.num_neighbor:d}_nn_corr_pairs"] = corr_pairs
    data[f"{args.num_neighbor:d}_nn_corr_yy"] = yy
    data[f"{args.num_neighbor:d}_nn_corr_zz"] = zz
    data["z"] = z
    data["result"] = result

    with open(os.path.join(save_dir, f"XXZ_2d_jz_{args.jz:.1f}.pkl"), "wb") as f:
        pickle.dump(data, f)
    f.close()
