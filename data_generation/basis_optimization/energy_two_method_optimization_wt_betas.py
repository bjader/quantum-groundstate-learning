import os
import time
import argparse
from itertools import islice

import dill as pickle
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from qiskit.transpiler import CouplingMap
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.qubit import solve_qubit
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit import QuantumCircuit
from pathlib import Path

from qaml.graph.graph_utils import read_edges_txt
from qaml.skqd.basis_optimise import (
    cost,
    obp_modified_hamiltonian,
    build_variational_circuit,
)

import logging
import sys
from datetime import datetime

import gc
import psutil

# -----------------------------
# Argument parsing (from SLURM .sh or command line)
# -----------------------------
parser = argparse.ArgumentParser(description="Two-stage SKQD optimization script.")
parser.add_argument(
    "--edges_path", type=str, default="../../cmaps/heavy_hex_edges_d_7.txt"
)
parser.add_argument("--jxx", type=float, default=1.0)
parser.add_argument("--jz", type=float, default=1.6)
parser.add_argument("--num_bs_opt", type=int, default=100)  # For stage 1
parser.add_argument("--num_bs_opt_2", type=int, default=300000)  # For stage 2 (full)
parser.add_argument("--eta", type=float, default=0.1)
parser.add_argument("--count_num_stage1", type=int, default=20)
parser.add_argument("--count_num_stage2", type=int, default=40)
parser.add_argument("--idea", type=str, default="idea_0_optimized_new_betas_two_ansatz")

args = parser.parse_args()

edges_path = args.edges_path
jxx = args.jxx
jz = args.jz
num_bs_opt = args.num_bs_opt
num_bs_opt_2 = args.num_bs_opt_2
eta = args.eta
count_num_stage1 = args.count_num_stage1
count_num_stage2 = args.count_num_stage2
idea = args.idea


# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Print the current working directory
print("Current working directory (PWD):", os.getcwd())

# Print the absolute path to the log directory
print("Log directory absolute path:", os.path.abspath(log_dir))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"run_{timestamp}.txt")


# Define a custom log filter
class FilterSpecificLogs(logging.Filter):
    def filter(self, record):
        # Ignore logs with certain substrings in the message
        unwanted_messages = ["size of the 0-th observable", "Backpropagated"]
        return not any(message in record.getMessage() for message in unwanted_messages)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),  # keep printing to terminal
    ],
)

# Add the filter to the logger
logger = logging.getLogger(__name__)

# Apply filter to all handlers, including the StreamHandler
for handler in logging.root.handlers:
    handler.addFilter(FilterSpecificLogs())


# Redirect stdout and stderr to logger
class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        pass


sys.stdout = StreamToLogger(logger)
sys.stderr = StreamToLogger(logger, logging.ERROR)

# Example of logging
logger.info("This message will be logged.")
logger.info("Backpropagated 9 DAG multi-graph slices")  # This should be ignored
logger.info("size of the 0-th observable: 739")  # This should also be ignored
# note: these two lines are ignored as sqd addon also prints such lines which then gets logged.


def log_memory(logger, tag=""):
    """
    Logs RSS (resident set size) and available system memory.
    """
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024**3)
    avail_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(
        f"[MEM]{'[' + tag + ']' if tag else ''} "
        f"RSS={rss_gb:.2f} GB | Available={avail_gb:.2f} GB"
    )


def free_memory(logger=None, tag=""):
    """
    Force Python + NumPy garbage collection.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    if logger:
        log_memory(logger, tag=tag)


time_0 = time.time()
# -----------------------------
# Setup
# -----------------------------
edges, n_spins = read_edges_txt(edges_path)
print(f"Number of qubits: {n_spins}")
cmap = CouplingMap(edges)
H = generate_xyz_hamiltonian(cmap, coupling_constants=(jxx / 4, jxx / 4, jz / 4))

load_saved_res = False
# res_fn = f"n_{n_spins}_jz_{jz}_ts_1_kd_10_shots_100k_dt_0.00_0.05_backend_ITN_chi_8_iter_20_num_bs_opt_100_blocks_132.pkl"
# load data from previous optimization to load betas and params
res_fn = f"n_{n_spins}_jz_{jz}_ts_1_kd_10_shots_100k_dt_0.00_0.05_backend_ITN_chi_8_iter_20_num_bs_opt_100_blocks_132_eta_0.1.pkl"

scipy_kwargs = {"k": 2, "which": "SA"}

if n_spins == 19:
    skqd_dir = f"../../data/spins_{n_spins}/skqd_sim/ts_1_kd_11_shots_100k_dt_0.05_0.15_backend_Aer_sv"
else:
    skqd_dir = f"../../data/spins_{n_spins}/skqd_sim/ts_1_kd_10_shots_100k_dt_0.00_0.05_backend_ITN_chi_8"

dmrg_dir = f"../../data/spins_{n_spins}/dmrg/chi_max_80_trunc_1e-06"

with open(os.path.join(skqd_dir, "train", f"XXZ_2d_jz_{jz:.1f}.pkl"), "rb") as f:
    data = pickle.load(f)

with open(os.path.join(dmrg_dir, f"XXZ_2d_jz_{jz:.1f}.pkl"), "rb") as dmrg_f:
    dmrg_res = pickle.load(dmrg_f)

log_memory(logger, "after loading pickles")

e0 = dmrg_res["result"].e
gs_dict = data["ground_state_dict"]
#gs_dict = data["ground_state_vector"]
skqd_e = data["ground_state_energy"]
print(f"SKQD energy {skqd_e}")


# -----------------------------
# Ansatz construction
# -----------------------------
G = nx.Graph()
G.add_nodes_from(range(n_spins))
G.add_edges_from(edges)

L = nx.line_graph(G)
edge_to_color = nx.coloring.greedy_color(L, strategy="smallest_last")
ansatz_edges = [
    (i, j)
    for color in sorted(set(edge_to_color.values()))
    for (i, j), c in edge_to_color.items()
    if c == color
]

params_per_block = 1
numparams = params_per_block * len(ansatz_edges)
params = np.zeros(numparams)

qc_dual = build_variational_circuit(n_spins, params, ansatz_edges)


# -----------------------------
# Cost functions
# -----------------------------
def costf(num_qubits, params, numparams, bs_matrix_local=None, betas_local=None):
    qc_here = build_variational_circuit(num_qubits, params, ansatz_edges)
    return cost(params, H, bs_matrix_local, betas_local, ansatz_edges, qc_here)


def costp(
    num_qubits, params, numparams, index1, bs_matrix_local=None, betas_local=None
):
    shifted = params.copy()
    shifted[index1] += np.pi / 2
    return costf(num_qubits, shifted, numparams, bs_matrix_local, betas_local)


def costm(
    num_qubits, params, numparams, index1, bs_matrix_local=None, betas_local=None
):
    shifted = params.copy()
    shifted[index1] -= np.pi / 2
    return costf(num_qubits, shifted, numparams, bs_matrix_local, betas_local)


# -----------------------------
# Gradient / optimization loop (Stage 1 + Stage 2)
# -----------------------------
def gradient_two_stage(
    num_qubits,
    params,
    numparams,
    bs_matrix_stage1,
    betas_stage1,
    bs_matrix_all,
    betas_stage2,
    eta=0.1,
    eps_grad=1e-12,
    eps_cost=1e-12,
    count_num_stage1=20,
    count_num_stage2=40,
    gradcount=200,
    qc=None,
    write_fn_stage1=None,
    write_fn_stage2=None,
):
    # ---------------- Stage 1 ----------------
    gcount = 0
    count = 0
    param_current = params.copy()
    log_memory(logger, f"iter {count} start")

    # Compute initial cost and energy
    cost_value = costf(
        num_qubits, param_current, numparams, bs_matrix_stage1, betas_stage1
    )

    # initial second-diagonalization energy
    optimized_circuit = build_variational_circuit(
        num_qubits, param_current, ansatz_edges
    )
    H_tilde_optimized = obp_modified_hamiltonian(H, optimized_circuit)
    new_energy, new_eigenstates = solve_qubit(
        bs_matrix_all, H_tilde_optimized, k=1, which="SA"
    )
    global_energy = new_energy[0]
    # normalize betas
    # betas_stage1[:] = new_eigenstates[:, np.argmin(new_energy)]
    # betas_stage1[:] /= np.sqrt(np.sum(np.abs(betas_stage1)**2))

    results_stage1 = {}
    results_stage1[count] = {
        "cost": cost_value,
        "params": param_current.copy(),
        "final_global_energy": global_energy,
        "betas": betas_stage1.copy(),
    }
    print(f"Iteration {count}: Cost={cost_value}, 2nd diag energy={global_energy}")

    min_global_energy = np.inf
    best_params_stage1 = param_current.copy()

    while count < count_num_stage1 and gcount < gradcount:
        # Compute gradient and update params
        grad_vec = np.zeros(numparams)
        for i in range(numparams):
            fp = costp(
                num_qubits, param_current, numparams, i, bs_matrix_stage1, betas_stage1
            )  # old betas now
            fm = costm(
                num_qubits, param_current, numparams, i, bs_matrix_stage1, betas_stage1
            )
            grad_vec[i] = (fp - fm) / 2

        grad_norm_sq = np.linalg.norm(grad_vec) ** 2
        if grad_norm_sq <= eps_grad:
            gcount += 1

        param_proposed = param_current - eta * grad_vec

        cost_new = costf(
            num_qubits, param_proposed, numparams, bs_matrix_stage1, betas_stage1
        )
        if cost_value - cost_new >= eta * grad_norm_sq:
            eta *= 2
            param_current = param_proposed
        elif cost_value - cost_new < (eta / 2) * grad_norm_sq:
            eta /= 2
            param_current = param_current - eta * grad_vec
        else:
            param_current = param_proposed

        # 5. Second diagonalization to potentially update betas
        free_memory(logger, f"iter {count} after diag")
        # Second diagonalization
        qc_here = build_variational_circuit(num_qubits, param_current, ansatz_edges)
        H_tilde = obp_modified_hamiltonian(H, qc_here)
        new_energy, new_eigenstates = solve_qubit(
            bs_matrix_all, H_tilde, k=1, which="SA"
        )
        global_energy = new_energy[0]
        cost_value = costf(
            num_qubits, param_current, numparams, bs_matrix_stage1, betas_stage1
        )
        count += 1

        results_stage1[count] = {
            "cost": cost_value,
            "params": param_current.copy(),
            "final_global_energy": global_energy,
            "betas": betas_stage1.copy(),
        }
        print(f"Iteration {count}: Cost={cost_value}, 2nd diag energy={global_energy}")

        # Save checkpoint
        if write_fn_stage1:
            exp_dir = write_fn_stage1.replace(".pkl", "")
            write_dir = Path(f"{exp_dir}")
            write_dir.mkdir(parents=True, exist_ok=True)
            with open(f"{write_dir}/{count}.pkl", "wb") as f:
                pickle.dump(results_stage1, f)
        free_memory(logger, "after checkpoint save")

        tol = 1e-8

        # Track minimum
        if global_energy < min_global_energy - tol:
            min_global_energy = global_energy
            best_params_stage1 = param_current.copy()
        else:
            print("[Stage1] Global energy increased, stopping stage 1")
            break

    # ---------------- Stage 2 ----------------
    print(
        f"Starting Stage 2 with best Stage1 params, min global energy={min_global_energy}"
    )
    param_current = best_params_stage1.copy()
    # Compute initial cost and energy
    optimized_circuit = build_variational_circuit(
        num_qubits, param_current, ansatz_edges
    )
    H_tilde_optimized = obp_modified_hamiltonian(H, optimized_circuit)
    new_energy, new_eigenstates = solve_qubit(
        bs_matrix_all, H_tilde_optimized, k=1, which="SA"
    )
    global_energy = new_energy[0]
    cost_value = costf(
        num_qubits, param_current, numparams, bs_matrix_all, betas_stage2
    )
    gcount1 = 0
    count1 = 0
    results_stage2 = {}
    results_stage2[count1] = {
        "cost": cost_value,
        "params": param_current.copy(),
        "final_global_energy": new_energy,
        "betas": betas_stage2.copy(),
    }
    print(f"Iteration-opt2 {count1}: Cost={cost_value}, 2nd diag energy={new_energy}")
    min_global_energy = np.inf
    best_params_stage2 = param_current
    while count1 < count_num_stage2 and gcount1 < gradcount:
        grad_vec = np.zeros(numparams)
        for i in range(numparams):
            fp = costp(
                num_qubits, param_current, numparams, i, bs_matrix_all, betas_stage2
            )
            fm = costm(
                num_qubits, param_current, numparams, i, bs_matrix_all, betas_stage2
            )
            grad_vec[i] = (fp - fm) / 2

        grad_norm_sq = np.linalg.norm(grad_vec) ** 2
        if grad_norm_sq <= eps_grad:
            gcount1 += 1

        # 2. Proposed parameter update
        param_proposed = param_current - eta * grad_vec
        cost_new = costf(
            num_qubits, param_proposed, numparams, bs_matrix_all, betas_stage2
        )
        if cost_value - cost_new >= eta * grad_norm_sq:
            eta *= 2
            param_current = param_proposed
        elif cost_value - cost_new < (eta / 2) * grad_norm_sq:
            eta /= 2
            param_current = param_current - eta * grad_vec
        else:
            param_current = param_proposed

            # 5. Second diagonalization to potentially update betas
            free_memory(logger, f"[Stage2] iter {count1} after diag")
        # Second diagonalization

        qc_here = build_variational_circuit(num_qubits, param_current, ansatz_edges)
        H_tilde = obp_modified_hamiltonian(H, qc_here)
        new_energy, eigenstates = solve_qubit(bs_matrix_all, H_tilde, k=1, which="SA")
        global_energy = new_energy[0]
        cost_value = costf(
            num_qubits, param_current, numparams, bs_matrix_all, betas_stage2
        )
        count1 += 1

        free_memory(logger, "after checkpoint save")

        if global_energy < min_global_energy - tol:
            min_global_energy = global_energy
            best_params_stage2 = param_current
            results_stage2[count1] = {
                "cost": cost_value,
                "params": best_params_stage2.copy(),
                "final_global_energy": global_energy,
                "betas": betas_stage2.copy(),
            }
            print(
                f"[Stage2] Iter {count1}: Cost={cost_value}, 2nd diag energy={global_energy}"
            )
            if write_fn_stage2:
                exp_dir = write_fn_stage2.replace(".pkl", "")
                write_dir = Path(f"{exp_dir}")
                write_dir.mkdir(parents=True, exist_ok=True)
                with open(f"{write_dir}/{count1}.pkl", "wb") as f:
                    pickle.dump(results_stage2, f)

        else:
            print("[Stage2] Global energy increased, stopping stage 2")
            print(
                f"Global energy increased from {min_global_energy} to {global_energy}"
            )
            count1 -= 1
            break

    return results_stage1, results_stage2, count, count1


# -----------------------------
# Run two-stage optimization
# -----------------------------
# Output paths - relative to current directory
path_stage1 = f"saved_opt_res/{idea}/stage1"
path_stage2 = f"saved_opt_res/{idea}/stage2"
os.makedirs(path_stage1, exist_ok=True)
os.makedirs(path_stage2, exist_ok=True)

res_optimized_fn = f"n_{n_spins}_jz_{jz}_{skqd_dir.split('/')[-1]}_iter_{count_num_stage1}_{count_num_stage2}_num_bs_opt_{num_bs_opt}_blocks_{len(ansatz_edges)}_eta_{eta}"

path1 = f"{path_stage1}/{res_optimized_fn}"
os.makedirs(path1, exist_ok=True)

path2 = f"{path_stage2}/{res_optimized_fn}"
os.makedirs(path2, exist_ok=True)

print(f"JZ: {jz}")
print(f"num_bs_opt: {num_bs_opt}")
print(f"num_bs_opt_2: {num_bs_opt_2}")
print(f"count_num_stage1: {count_num_stage1}")
print(f"count_num_stage2: {count_num_stage2}")
print(f"eta: {eta}")
print(f"idea: {idea}")
print(f"path_stage1: {path_stage1}")
print(f"path_stage2: {path_stage2}")
print(f"path1: {path1}")
print(f"path2: {path2}")

gs_dict = {k: v for k, v in gs_dict.items()}
gs_dict_sorted = dict(
    sorted(gs_dict.items(), key=lambda item: abs(item[1]) ** 2, reverse=True)
)

gs_subdict = dict(islice(gs_dict_sorted.items(), num_bs_opt))
bs_matrix, _ = counts_to_arrays(gs_subdict)
betas = np.array(list(gs_subdict.values()))

gs_subdict_2 = dict(islice(gs_dict.items(), num_bs_opt_2))
bs_matrix_all, _ = counts_to_arrays(gs_subdict_2)
betas_stage2 = np.array(list(gs_subdict_2.values()))

#gs_raw = data["ground_state_vector"]
#bs_matrix_all_reconstructed = data["basis_bitstrings"]
#betas_stage2 = np.zeros(bs_matrix_all_reconstructed.shape[0], dtype=complex)
#betas_stage2[:] = gs_raw[:, 0]
#betas_stage2[:] /= np.sqrt(np.sum(np.abs(betas_stage2) ** 2))

# Compute probabilities |beta|^2
#probs = np.abs(betas_stage2) ** 2

# Get indices sorted descending by probability
#sorted_indices = np.argsort(probs)[::-1]

# Take top-k
#top_indices = sorted_indices[:num_bs_opt]

# Subset basis and amplitudes
#bs_matrix = bs_matrix_all_reconstructed[top_indices]
#betas = betas_stage2[top_indices]

log_memory(logger, "after bs_matrix")
res_stage1, res_stage2, count_num_stage1, count_num_stage2 = gradient_two_stage(
    H.num_qubits,
    params,
    numparams,
    bs_matrix,
    betas,
    bs_matrix_all,
    betas_stage2,
    eta=eta,
    eps_grad=1e-12,
    eps_cost=1e-12,
    count_num_stage1=count_num_stage1,
    count_num_stage2=count_num_stage2,
    gradcount=200,
    qc=qc_dual,
    write_fn_stage1=f"{path1}",
    write_fn_stage2=f"{path2}",
)


with open(f"{path1}/res.pkl", "wb") as f:
    pickle.dump(res_stage1, f)
with open(f"{path2}/res.pkl", "wb") as f:
    pickle.dump(res_stage2, f)
print("Two-stage optimization finished.")
print(f"Results saved to {path1} and {path2}")

# -----------------------------
# Post-processing
# -----------------------------

print("printing...")
param_optimized = res_stage2[count_num_stage2]["params"]
betas_optimized = res_stage2[count_num_stage2]["betas"]
optimized_circuit = build_variational_circuit(
    H.num_qubits, param_optimized, ansatz_edges
)

H_tilde_optimized = obp_modified_hamiltonian(H, optimized_circuit)

new_energies, new_eigenstates = solve_qubit(
    bs_matrix_all, H_tilde_optimized, **{"k": 1, "which": "SA"}
)
e_after_opt = new_energies[0]

new_betas = new_eigenstates[:, np.argmin(new_energies)]

# Normalize the ground state amplitudes
new_betas /= np.sqrt(np.sum(np.abs(new_betas) ** 2))

cf = cost(param_optimized, H, bs_matrix_all, new_betas, ansatz_edges, qc_dual)
res_stage2["global_energy_after_opt"] = cf
res_stage2["final_global_energy"] = e_after_opt
res_stage2["bs_matrix"] = bs_matrix_all
res_stage2["betas"] = new_betas
with open(f"{path2}/res.pkl", "wb") as f:
    pickle.dump(res_stage2, f)
print(f"Energy after 2nd diagonalisation = {e_after_opt}")
print(f"Difference before: {skqd_e - e0}")
print(f"Difference after: {e_after_opt - e0}")
print(f"Improvement: {skqd_e - e_after_opt}")

with open(f"{path1}/res.pkl", "rb") as f:
    results1 = pickle.load(f)
cost_100 = [results1[i]["cost"] for i in range(len(results1) - 1)]
cost_hist = [results1[i]["final_global_energy"] for i in range(len(results1) - 1)]
with open(f"{path2}/res.pkl", "rb") as f:
    results2 = pickle.load(f)
cost_full = [results2[i]["cost"] for i in range(len(results2) - 4)]

# need to add one more cost
# Example safety: convert to numpy arrays
cost_100 = np.asarray(cost_100)
cost_hist = np.asarray(cost_hist)
cost_full = np.asarray(cost_full)

# x positions
x_hist = np.arange(len(cost_hist))  # where cost_hist lives
x_full = np.arange(
    len(cost_hist), len(cost_hist) + len(cost_full)
)  # appended after hist
x_100 = np.arange(len(cost_100))  # independent x for cost_100

# Combined for the full timeline (if needed elsewhere)
cost_history = np.concatenate([cost_hist, cost_full])
iterations = np.arange(len(cost_history))

# Configure Matplotlib for LaTeX and nice fonts
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 120,
    }
)

figure_path = f"figures/{idea}/"
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

# Create the plot
fig, ax = plt.subplots(figsize=(7, 5))
# cost_100: separate color and x-range (length mismatch is acceptable)
ax.plot(
    x_100,
    cost_100,
    color="#1f77b4",
    linestyle="-",
    marker="o",
    label="optimized over 100",
)

# cost_hist: first segment of the timeline
ax.plot(
    x_hist, cost_hist, color="#ff7f0e", linestyle="-", marker="o", label="global energy"
)

# cost_full: plotted immediately after cost_hist
ax.plot(
    x_full,
    cost_full,
    color="#2ca02c",
    linestyle="-",
    marker="o",
    label="optimized over all",
)

# Labels and title
ax.set_xlabel(r"Iteration")
ax.set_ylabel(r"Cost (energy)")

ax.grid(True, linestyle="--", alpha=0.6)

# plt.legend()
plt.tight_layout()
plt.savefig(f"{figure_path}/fig_1_n_{n_spins}_jz_{jz}.png")
plt.close()

plt.hlines(
    e0, iterations[0], iterations[-1], linestyle="--", color="tab:blue", label="DMRG"
)
plt.hlines(
    skqd_e,
    iterations[0],
    iterations[-1],
    linestyle="--",
    color="tab:orange",
    label="SKQD before opt",
)
plt.hlines(
    e_after_opt,
    iterations[0],
    iterations[-1],
    linestyle="--",
    color="tab:green",
    label="SKQD after opt",
)

plt.ylabel(r"Cost (energy)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
plt.savefig(f"{figure_path}/energy_comparison_n_{n_spins}_jz_{jz}.png")
plt.close()
time_total = time.time() - time_0
print(f"Total time taken: {time_total} seconds")
