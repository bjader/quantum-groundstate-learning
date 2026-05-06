import argparse
import logging
import os

import dill as pickle
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian

from qaml.graph.graph_utils import read_edges_txt, get_coloured_edges
from qaml.observables.correlation_function import get_n_nearest_neighbours, all_k_cycles_undirected
from qaml.observables.modified_basis import expectation_value_modified_basis
from qaml.skqd.basis_optimise import build_variational_circuit
from qaml.skqd.utils import energy_and_variance_in_obp_subspace

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-jz", type=float, required=True, help="Jz value")
parser.add_argument("--data_timestamp", type=str, help="Data timestamp")
parser.add_argument("--nn", type=int, default=2, help="Number of nearest neighbors for correlation functions")
parser.add_argument("--force", action="store_true", help="Force recalculation of all observables even if they exist")
args = parser.parse_args()

logger.info(args)

n = 57
jz = args.jz
data_timestamp = args.data_timestamp
nn = args.nn
force_recalc = args.force

def compute_obs_and_store(data, key, obs, bs_matrix, betas, qc_opt, stats=False):
    """Compute observables serially (no multiprocessing) for better cluster compatibility."""
    vals = []
    total = len(obs)
    for i, observable in enumerate(obs, start=1):
        logger.info(f"Computing {key} observable {i} of {total}")
        val = expectation_value_modified_basis(bs_matrix, betas, observable, qc_opt)
        vals.append(val)

    vals = np.array(vals)
    if stats:
        logger.info(f"{key}: min={np.nanmin(vals):.6f}, max={np.nanmax(vals):.6f}, mean={np.nanmean(vals):.6f}")
    data[key] = vals
    return vals

if __name__ == "__main__":
    if n==115:
        edges_path = "../cmaps/heavy_hex_edges_d_7.txt"
    elif n==57:
        edges_path = "../cmaps/heavy_hex_edges_d_5.txt"
    edges, n_spins = read_edges_txt(edges_path)
    logger.info(f"Number of qubits: {n_spins}")

    edges = get_coloured_edges(n_spins, edges)

    root_dir_skqd = f"../vopt/data/spins_{n}/skqd/ts_1_kd_11_shots_100k_ibm_boston_{data_timestamp}/recovery_random_flip"

    logger.info(f"Processing jz: {jz:.1f}")
    try:
        with open(os.path.join(root_dir_skqd, f"XXZ_2d_jz_{jz:.1f}.pkl"), "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"No data found for jz: {jz:.1f}, exiting...")
        exit(1)

    # Try modern path first (with recovery_post_select), then fall back to legacy path
    saved_opt_res_dir_modern = f"../vopt/saved_opt_res/n_{n}_jz_{jz}_ts_1_kd_11_shots_100k_ibm_boston_{data_timestamp}_recovery_post_select_colouring_optimal_iter_50_num_bs_opt_all_blocks_132_eta_0.1_stage2"
    saved_opt_res_dir_legacy = f"../vopt/saved_opt_res/n_{n}_jz_{jz}_ts_1_kd_11_shots_100k_ibm_boston_{data_timestamp}_colouring_optimal_iter_50_num_bs_opt_all_blocks_132_eta_0.1_stage2"
    
    res = None
    for saved_opt_res_dir in [saved_opt_res_dir_modern, saved_opt_res_dir_legacy]:
        try:
            with open(f"{saved_opt_res_dir}/final_res.pkl", "rb") as f:
                res = pickle.load(f)
            logger.info(f"Loaded optimization results from: {saved_opt_res_dir}")
            break
        except FileNotFoundError:
            continue
    
    if res is None:
        logger.error(f"No optimised basis change found for jz: {jz:.1f}, exiting...")
        exit(1)

    # Load parameters from the last iteration of stage 2 (which is optimal for stage 2)
    last_iter = max(k for k in res.keys() if isinstance(k, int))
    param_optimized = res[last_iter]["params"]
    logger.info(f"Loaded parameters from stage 2 iteration {last_iter}")
    
    # Load the expected final energy for validation
    expected_final_energy = res.get("final_global_energy")
    if expected_final_energy is not None:
        logger.info(f"Expected final_global_energy from optimization: {expected_final_energy:.8f}")
    else:
        logger.warning("final_global_energy not found in optimization results")
    
    if "hopt" in root_dir_skqd:
        logger.info("Using expanded set of bitstrings from multiple runs")
        state_dict = data["ground_state_dict"]
        bs_matrix, _ = counts_to_arrays(state_dict)
        betas = np.array(list(state_dict.values()))
    else:
        bs_matrix = res["bs_matrix"]
        betas = res["betas"]

    qc_opt = build_variational_circuit(n_spins, param_optimized, edges)

    if force_recalc or not all(k in data for k in ["energy_vopt", "energy_var_vopt"]):
        try:
            logger.info("Computing energy and variance")
            cmap = CouplingMap(edges)
            jxx = 1.0
            H = generate_xyz_hamiltonian(cmap, coupling_constants=(jxx / 4, jxx / 4, jz / 4))
            E_vopt, var_vopt = energy_and_variance_in_obp_subspace(H, qc_opt, bs_matrix, betas)
            data["energy_vopt"] = E_vopt
            data["energy_var_vopt"] = var_vopt
            logger.info(f"[vopt] Energy={E_vopt:.8f}  Var={var_vopt:.3e}")
            
            # Validate against expected energy from optimization
            if expected_final_energy is not None:
                energy_diff = abs(E_vopt - expected_final_energy)
                if energy_diff < 1e-6:
                    logger.info(f"✓ Energy validation passed: calculated energy matches expected (diff={energy_diff:.2e})")
                else:
                    logger.warning(f"⚠ Energy validation failed: calculated={E_vopt:.8f}, expected={expected_final_energy:.8f}, diff={energy_diff:.2e}")
                    logger.warning("  This may indicate incorrect parameters are being used!")

        except Exception as e:
            logger.error(f"Failed OBP vopt energy/variance computation for jz={jz:.1f}: {e}")
    else:
        logger.info("Skipping energy/variance calculation (already exists)")

    corr_func_pairs = get_n_nearest_neighbours(edges, nn)

    for pauli_2q, key_suffix in (("XX", "xx_basis_opt"), ("ZZ", "zz_basis_opt")):
        key_vals = f"{nn}_nn_corr_funcs_{key_suffix}"
        key_pairs = f"{nn}_nn_corr_pairs_{key_suffix}"
        if force_recalc or key_vals not in data:
            logger.info(f"Computing {pauli_2q} correlation functions")
            obs = [SparsePauliOp.from_sparse_list([(pauli_2q, [i, j], 1)], num_qubits=n_spins)
                   for
                   i, j in corr_func_pairs]
            compute_obs_and_store(data, key_vals, obs, bs_matrix, betas, qc_opt, stats=True)
            data[key_pairs] = corr_func_pairs
        else:
            logger.info(f"Skipping {pauli_2q} correlation functions (already exist)")

    if force_recalc or "z_basis_opt" not in data:
        logger.info("Computing Z expectation value")
        obs = [SparsePauliOp.from_sparse_list([("Z", [i], 1)], num_qubits=n_spins) for i in
               range(n_spins)]
        compute_obs_and_store(data, "z_basis_opt", obs, bs_matrix, betas, qc_opt)
    else:
        logger.info("Skipping Z expectation values (already exist)")

    loops = all_k_cycles_undirected(edges, k=12)

    for pauli, key in (("X", "x_loop"), ("Z", "z_loop")):
        key_sites = f"{key}_sites"
        if force_recalc or key not in data:
            logger.info(f"{len(loops)} loops of length 12 found")
            logger.info(f"Loops: {loops}")
            obs = [
                SparsePauliOp.from_sparse_list([(pauli * len(loop), list(loop), 1)],
                                               num_qubits=n_spins)
                for loop in loops
            ]
            compute_obs_and_store(data, key, obs, bs_matrix, betas, qc_opt)
            data[key_sites] = loops
        else:
            logger.info(f"Skipping {pauli} loop observables (already exist)")

    logger.info(f"x_loop_sites: {data.get('x_loop_sites')}")
    logger.info(f"x_loop: {data.get('x_loop')}")

    with open(os.path.join(root_dir_skqd, f"XXZ_2d_jz_{jz:.1f}.pkl"), "wb") as f:
        pickle.dump(data, f)
    
    logger.info(f"Successfully processed and saved data for jz={jz:.1f}")
