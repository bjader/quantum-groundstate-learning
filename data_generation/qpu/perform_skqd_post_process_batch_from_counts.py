"""
This script diagonalises samples from an SKQD run and produces a Hamiltonian and ground states which
is added to /data/

Modified version that loads counts_all from saved files instead of fetching from IBM Quantum Cloud API.
"""

import logging
import os
from pathlib import Path

import dill as pickle
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian

from qaml.graph.graph_utils import read_edges_txt
from qaml.skqd import SKQDPostprocessor

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("qaml").setLevel(logging.INFO)
logging.getLogger("skqd").setLevel(logging.INFO)

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

g = nx.Graph()
g.add_edges_from(edges)
color = nx.algorithms.bipartite.color(g)
to_flip = [q for q, c in color.items() if c == 0]

scipy_kwargs = {"k": 2, "which": "SA"}

job_id_load_dir = "ts_1_kd_11_shots_100k_ibm_boston_1773150437_1773854302_mixed"

krylov_dim = 11
recovery_method = "random_flip"

# Load directory for saved counts_all
counts_all_path = os.path.join("counts_all", f"spins_{n:d}", "skqd", job_id_load_dir)

# Save directory for the dataset.
save_dir = os.path.join(
    "../../data",
    f"spins_{n:d}",
    "skqd",
    job_id_load_dir,
    f"recovery_{recovery_method}",
)
Path(save_dir).mkdir(parents=True, exist_ok=True)

jxx = 1.0

jzs = [round(jz, 1) for jz in np.arange(1.1, 6.01, 0.1)]

for jz in jzs:
    # Load counts_all from saved file
    counts_all_file = os.path.join(counts_all_path, f"counts_all_jz_{jz:.1f}.pkl")
    
    try:
        with open(counts_all_file, 'rb') as f:
            counts_all = pickle.load(f)
        logger.info(f"Loaded counts_all for Jz={jz:.1f} from {counts_all_file}")
    except FileNotFoundError:
        logger.info(f"Counts_all data for Jz {jz} not found at {counts_all_file}")
        continue
    
    # Create Hamiltonian for this Jz
    H_op = generate_xyz_hamiltonian(cmap, coupling_constants=(jxx / 4, jxx / 4, jz / 4))
    
    # Create SKQDPostprocessor (job=None since we have counts_all)
    skqd_postprocessor = SKQDPostprocessor(
        job=None,
        krylov_dim=krylov_dim,
        H_op=H_op,
        backend_name="ibm_kingston",
        scipy_kwargs=scipy_kwargs,
        num_ones=len(to_flip),
        config_recovery_method=recovery_method
    )
    
    # Run post-processing from loaded counts_all
    ground_state_energy, ground_state_dict, _ = skqd_postprocessor.run_from_counts_all(counts_all)
    
    result = {
        "jz": round(jz, 4),
        "ground_state_energy": ground_state_energy,
        "ground_state_dict": ground_state_dict
    }
    logger.info(f"SKQD energy: {ground_state_energy}")

    with open(os.path.join(save_dir, f"XXZ_2d_jz_{round(jz, 4):.1f}.pkl"), "wb") as f:
        pickle.dump(result, f)

logger.info("=" * 80)
logger.info("COMPLETE")
logger.info("=" * 80)
