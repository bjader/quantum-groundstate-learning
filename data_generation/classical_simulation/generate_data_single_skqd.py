"""
NOTE: To run belief propagation simulation (using ITNBackend) you need to install the old version
of qiskit-tnqs, previously called ITensorNetworksQiskit, at this specific commit:
https://github.com/qiskit-community/qiskit-tnqs/tree/400ee6857059b935cbb4438cc684a0382c6806e5
"""

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("qaml").setLevel(logging.INFO)
logging.getLogger("skqd").setLevel(logging.INFO)
logging.getLogger("custom_backends").setLevel(logging.INFO)

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import dill as pickle
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit_aer import AerSimulator
from skopt.space import Real, Integer

from qaml.graph.graph_utils import read_edges_txt
from qaml.graph.heavy_hex_2d_coords import d7_2d_coords, d5_2d_coords
from qaml.observables.correlation_function import n_nearest_neighbor_corr_nd
from qaml.skqd import SKQDRunner
from qaml.skqd.hyperparam_optimise import optimize_skqd_gp
from qaml.skqd import ITNBackend
from qaml.skqd.utils import shots_format

# Workflow:
# 0. Load the fixed parameters and search parameters from the Json file.
# 1. For the given coupling J_z, optimize dt using Bayesian optimisation (GP) using 10x less shots to save runtime.
# 2. Run SKQD again for the optimal dt using the originally provided number of shots
# 3. Calculate the two-body correlation functions.
# 4. Save the two-body correlation functions, the SKQD ground state, and the optimal parameters.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-jz", type=float, help="ZZ_coupling_const")
    parser.add_argument("-e", "--edges_fn", default="heavy_hex_edges_d_7.txt",
                        help="Name of file with undirected edges, assuming it lives in 2d_xxz/cmaps")
    parser.add_argument(
        "-sp", "--search_params", type=str, help="JSON-formatted search-params"
    )
    parser.add_argument(
        "-fp", "--fixed_params", type=str, help="JSON-formatted fixed-params"
    )
    parser.add_argument("-sd", "--save_dir", type=str, help="Save Directory")

    parser.add_argument("-nn", "--num_neighbor", type=int, default=4)

    parser.add_argument("-b", "--backend", type=str, default="ITN", choices=['ITN', 'Aer', 'Aer_sv'])

    parser.add_argument("-x", "--chi", type=int, default=2,
                        help="Bond dimension if using ITN backend")

    parser.add_argument("-tt", "--trunc_thr", type=str, default="1e-3",
                        help="Truncation threshold if using the Qiskit Aer MPS backend")

    return parser.parse_args()


def generate_data_xxz_2d(
    jz: float,
    cmap: CouplingMap,
    fixed_params: Dict[str, Any],
    search_params: Dict[str, Dict[str, List[float | int] | str]],
) -> Dict[str, str | float | Any]:
    """
    Returns:
        Dict[str, str | float | Any]: Dictionary of
         - best ground state energy and ground state dictionary,
         - the corresponding optmisation method,
         - percentage difference from the DMRG solution,
         - the Jz value
    """

    jxx = (
        1.0 / 4
    )  # Since we are using Pauli operators, this is the same as jxx=1.0 for spin operators
    jz /= 4  # Same as above
    H_op = generate_xyz_hamiltonian(cmap, coupling_constants=(jxx, jxx, jz))
    fixed_params["H_op"] = H_op

    # Define the search space for parameters to optimise with Gaussian processes
    search_space = []
    for k, v in search_params.items():
        if v["dtype"] == "float":
            search_space.append(Real(*v["bounds"], name=k))
        elif v["dtype"] == "int":
            search_space.append(Integer(*v["bounds"], name=k))
        else:
            raise NotImplementedError

    gp_minimize_kwargs = {"n_calls": 10, "n_initial_points": 3, "random_state": 42, "verbose": True}

    result = optimize_skqd_gp(search_space, fixed_params, gp_minimize_kwargs)

    # Output the best parameters
    logger.info("Best parameters found:")
    for name, val in zip([d.name for d in search_space], result.x):
        logger.info(f"{name}: {val}")
    logger.info(f"Best ground state energy: {result.fun}")

    optimized_params_gp = {d.name: result.x[i] for i, d in enumerate(search_space)}
    params_gp = {**fixed_params, **optimized_params_gp}

    runner = SKQDRunner(**params_gp)
    gs_en_gp, ground_state_gp_dict, counts = runner.run()

    method, (skqd_e, ground_state_dict, params) = "gp", (gs_en_gp, ground_state_gp_dict, params_gp)

    logger.info(f"jz: {jz * 4:.1f}")

    return {
        "method": method,
        "params": params,
        "counts": counts,
        "ground_state_energy": skqd_e,
        "ground_state_dict": ground_state_dict,
        "jz": jz * 4,
    }


def main():
    args = parse_args()
    logger.info(args)

    # Define fixed parameters (not part of optimization)

    # Try to find cmaps directory in common locations
    edges_root_dirs = ["../../cmaps", "./cmaps", "../cmaps"]
    for edge_root_dir in edges_root_dirs:
        edges_path = os.path.join(edge_root_dir, args.edges_fn)
        if os.path.exists(edges_path):
            break

    edges, n = read_edges_txt(edges_path)
    # Prep `Neel` state as the reference state for evolution
    qc_state_prep = QuantumCircuit(n)
    g = nx.Graph()
    g.add_edges_from(edges)
    color = nx.algorithms.bipartite.color(g)
    to_flip = [q for q, c in color.items() if c == 0]
    qc_state_prep.x(to_flip)

    cmap = CouplingMap(edges)

    with open(args.fixed_params, "r") as f:
        fixed_params_external = json.load(f)
    f.close()

    trunc_thr = float(args.trunc_thr)
    if args.backend == "Aer":
        backend = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_truncation_threshold=trunc_thr
        )
    if args.backend == "Aer_sv":
        backend = AerSimulator()
    elif args.backend == "ITN":
        if n == 115:
            itn_qmap = d7_2d_coords()
        elif n == 57:
            itn_qmap = d5_2d_coords
        else:
            logger.warning(
                "Only cached ITNQ qmap is for the 115 or 57 qubit layout, using graph isomorphism instead")
            itn_qmap = None
        backend = ITNBackend(chi=args.chi, qmap=itn_qmap)

    fixed_params = {
        "n_qubits": n,
        "qc_state_prep": qc_state_prep,
        "backend": backend,
        "num_ones": len(to_flip),
        "colouring": "nx",
        "cmap": cmap,
        "parallelise_itn": False,
    }

    for k, v in fixed_params_external.items():
        fixed_params[k] = v

    # Define the search parameters for the parameters to optimize
    with open(args.search_params, "r") as f:
        search_params = json.load(f)
    f.close()

    # Create the directory to save.
    save_dir = os.path.join(
        args.save_dir,
        f"spins_{n:d}",
        "skqd_sim",
        f"ts_{fixed_params['num_trotter_steps']}_kd_{fixed_params['krylov_dim']}_shots_"
        f"{shots_format(fixed_params['shots'])}_dt_{search_params['dt']['bounds'][0]:.2f}_"
        f"{search_params['dt']['bounds'][1]:.2f}",
    )
    if args.backend == "Aer":
        save_dir += f"_backend_Aer_trunc_thr_{trunc_thr}"
    elif args.backend == "Aer_sv":
        save_dir += f"_backend_Aer_sv"
    else:
        save_dir += f"_backend_ITN_chi_{args.chi}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(save_dir, "train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(save_dir, "test")).mkdir(parents=True, exist_ok=True)

    logger.info(f"...Start generating data for jz: {args.jz:.1f}...")

    start = time.time()
    # Run SKQD with search space optimisation.
    result = generate_data_xxz_2d(
        jz=args.jz,
        cmap=cmap,
        fixed_params=fixed_params,
        search_params=search_params,
    )

    end = time.time()
    logger.info(f"Bayesian optimisation and SKQD time taken {end - start}")

    # Calculate the correlation functions
    start = time.time()
    corr_funcs, corr_func_pairs = n_nearest_neighbor_corr_nd(result["ground_state_dict"], edges, args.num_neighbor)
    result[f"{args.num_neighbor:d}_nn_corr_funcs"] = corr_funcs
    result[f"{args.num_neighbor:d}_nn_corr_pairs"] = corr_func_pairs
    end = time.time()
    logger.info(f"Correlation function calculation time taken {end - start}")

    if args.jz >= 1.6:
        with open(os.path.join(save_dir, "train", f"XXZ_2d_jz_{args.jz:.1f}.pkl"), "wb") as f:
            pickle.dump(result, f)
    elif args.jz <= 1.4:
        with open(os.path.join(save_dir, "test", f"XXZ_2d_jz_{args.jz:.1f}.pkl"), "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    main()
