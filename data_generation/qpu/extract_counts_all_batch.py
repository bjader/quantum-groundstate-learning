"""
This script extracts counts_all from SKQD QPU jobs and saves them to disk.
Similar to how job_ids are saved, but for the raw counts data.
"""

import logging
import os
from pathlib import Path

import dill as pickle
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit_ibm_runtime import QiskitRuntimeService

from qaml.graph.graph_utils import read_edges_txt
from qaml.skqd import SKQDPostprocessor

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("qaml").setLevel(logging.INFO)
logging.getLogger("skqd").setLevel(logging.INFO)

# Define fixed parameters

# Try to find cmaps directory in common locations
edges_root_dirs = ["../../cmaps", "./cmaps", "../cmaps"]
edges_fn = "heavy_hex_edges_d_7.txt"
for edge_root_dir in edges_root_dirs:
    edges_path = os.path.join(edge_root_dir, edges_fn)
    if os.path.exists(edges_path):
        break

edges, n = read_edges_txt(edges_path)
cmap = CouplingMap(edges)

g = nx.Graph()
g.add_edges_from(edges)
color = nx.algorithms.bipartite.color(g)
to_flip = [q for q, c in color.items() if c == 0]

# Qiskit backend
service = QiskitRuntimeService(
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/7e45c9f84c1c4c9da2eb18281f47d21d:7d0a36e8-7e15-4e36-9440-bdd567ce67fa::")

scipy_kwargs = {"k": 2, "which": "SA"}

job_id_load_dir = "ts_1_kd_11_shots_100k_ibm_boston_1773150437"

krylov_dim = 11
recovery_method = "post_select"

# Load directory for job ids
job_id_path = os.path.join("job_ids", f"spins_{n:d}", "skqd", job_id_load_dir)

# Save directory for counts_all (under qpu/ similar to job_ids)
save_dir = os.path.join(
    "counts_all",
    f"spins_{n:d}",
    "skqd",
    job_id_load_dir,
)
Path(save_dir).mkdir(parents=True, exist_ok=True)

jxx = 1.0

jzs = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]

logger.info("=" * 80)
logger.info("Extracting counts_all from QPU jobs")
logger.info("=" * 80)

for jz in jzs:
    logger.info(f"\nProcessing Jz={jz:.1f}")
    logger.info("-" * 40)
    
    # Load job ID
    job_id_file = os.path.join(job_id_path, f"job_id_jz_{jz:.1f}.txt")
    try:
        with open(job_id_file, 'r') as f:
            job_id = f.readline().strip()
    except FileNotFoundError:
        logger.info(f"Job for Jz={jz:.1f} not found")
        continue
    
    logger.info(f"Job ID: {job_id}")
    
    # Retrieve job from service
    job = service.job(job_id)
    
    # Create postprocessor to extract counts
    H_op = generate_xyz_hamiltonian(cmap, coupling_constants=(jxx / 4, jxx / 4, jz / 4))
    skqd_postprocessor = SKQDPostprocessor(
        job=job,
        krylov_dim=krylov_dim,
        H_op=H_op,
        backend_name="ibm_kingston",
        scipy_kwargs=scipy_kwargs,
        num_ones=len(to_flip),
        config_recovery_method=recovery_method
    )
    
    # Get counts_all
    counts_all = skqd_postprocessor._get_counts_all()
    
    logger.info(f"Extracted counts_all with {len(counts_all)} Krylov dimensions")
    for i, counts in enumerate(counts_all):
        logger.info(f"  Krylov dim {i}: {len(counts)} bitstrings")
    
    # Save counts_all to disk
    save_path = os.path.join(save_dir, f"counts_all_jz_{jz:.1f}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(counts_all, f)
    
    logger.info(f"Saved to: {save_path}")

logger.info("=" * 80)
logger.info("COMPLETE")
logger.info("=" * 80)
