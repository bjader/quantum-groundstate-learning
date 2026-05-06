"""
Submits a SKQD job evolving under the XXZ model and sampling in the normal computational basis
"""

import logging
import os
import time
from pathlib import Path

import dill as pickle
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit_ibm_runtime import QiskitRuntimeService

from qaml.graph.graph_utils import read_edges_txt
from qaml.skqd import SKQDSampler
from qaml.skqd.utils import shots_format

logging.basicConfig()
logger = logging.getLogger("qaml")
logger.setLevel(logging.DEBUG)

# Define fixed parameters (not part of optimization)

# Try to find cmaps directory in common locations
edges_root_dirs = ["../../cmaps", "./cmaps", "../cmaps"]
edges_fn = "heavy_hex_edges_d_7.txt"
for edge_root_dir in edges_root_dirs:
    edges_path = os.path.join(edge_root_dir, edges_fn)
    if os.path.exists(edges_path):
        break

edges, n = read_edges_txt(edges_path)
cmap = CouplingMap(edges)

# Prep `Neel` state as the reference state for evolution
qc_state_prep = QuantumCircuit(n)
g = nx.Graph()
g.add_edges_from(edges)
color = nx.algorithms.bipartite.color(g)
to_flip = [q for q, c in color.items() if c == 0]
qc_state_prep.x(to_flip)

# Qiskit backend
service = QiskitRuntimeService(
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/7e45c9f84c1c4c9da2eb18281f47d21d:7d0a36e8-7e15-4e36-9440-bdd567ce67fa::")
backend = service.backend("ibm_boston")
error_mitigation = None

shots_per_dim = 100_000

# Set fixed parameters for quantum Krylov algorithm
krylov_dim = 11  # size of krylov subspace
num_trotter_steps = 1

bopt_range = "0.00_0.05"
opt_dt_dir = f"../../data/spins_115/skqd_sim/ts_1_kd_{krylov_dim}_shots_10k_dt_{bopt_range}_backend_ITN_chi_8/"

# Save directories for job id and job
save_fn = f"ts_{num_trotter_steps:d}_kd_{krylov_dim:d}_shots_{shots_format(shots_per_dim)}_{backend.name}_{int(time.time())}"
save_dir_job_id = os.path.join("job_ids", f"spins_{n:d}", "skqd", save_fn)

Path(save_dir_job_id).mkdir(parents=True, exist_ok=True)
save_dir_job = os.path.join("jobs", f"spins_{n:d}", "skqd",
                            save_fn)
Path(save_dir_job).mkdir(parents=True, exist_ok=True)

jxx = (
    1.0 / 4
)  # Since we are using Pauli operators, this is the same as jxx=1.0 for spin operators

jzs = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]
for jz in jzs:
    H_op = generate_xyz_hamiltonian(cmap, coupling_constants=(jxx, jxx, jz / 4.0))

    next_dir = "test" if jz <= 1.4 else "train"
    opt_dt_fn = f"XXZ_2d_jz_{jz}.pkl"
    with open(os.path.join(opt_dt_dir, next_dir, opt_dt_fn), "rb") as f:
        dt = pickle.load(f)["params"]["dt"]
    logger.info(f"Jz: {jz:.1f}\nOptimal Time dt: {dt}")

    skqd_sampler = SKQDSampler(
        n_qubits=n,
        krylov_dim=krylov_dim,
        dt=dt,
        num_trotter_steps=num_trotter_steps,
        qc_state_prep=qc_state_prep,
        H_op=H_op,
        backend=backend,
        shots=shots_per_dim,
        cmap=cmap,
        colouring="nx",
        error_mitigation=error_mitigation,
    )

    job = skqd_sampler.submit_job()

    with open(os.path.join(save_dir_job_id, f"job_id_jz_{round(jz, 4):.1f}.txt"), "w") as f:
        f.write(job.job_id())

    with open(os.path.join(save_dir_job, f"job_jz_{round(jz, 4):.1f}.pkl"), "wb") as f:
        pickle.dump(job, f)
