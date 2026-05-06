import logging
import time

import numpy as np
from qiskit.transpiler import CouplingMap

logger = logging.getLogger(__name__)

try:
    from itensornetworks_qiskit.graph import map_onto_2d_grid, extract_cx_gates
    from itensornetworks_qiskit.sample import itn_samples_to_counts_dict
    from itensornetworks_qiskit.utils import qiskit_circ_to_itn_circ_2d
    from juliacall import Main as jl

    jl.seval("using ITensorNetworksQiskit")
    jl.seval("using ITensorNetworks: siteinds")
    logger.info("Loading Julia and ITNQ dependencies successful!")
except:
    logger.warning("Julia or ITensorNetworksQiskit environment not properly set up.")

class ITNBackend:
    def __init__(self, chi, qmap=None, proj_rank=1, norm_rank=1):
        self.chi = chi
        self.qmap = qmap
        self.proj_rank = proj_rank
        self.norm_rank = norm_rank

        self.name = "ITensorNetworks"

    def run(self, qc, shots, cmap: CouplingMap = None):
        if self.qmap is None:
            logger.warning("No qmap provided, running graph isomorphism which can take very long")
            edges = cmap.get_edges()
            edges = [list(e) for e in edges]  # Convert to list of lists
            qmap_0_ind = map_onto_2d_grid(edges)
            # Make the 2d coordinate mapping 1-indexed for Julia
            self.qmap = {k + 1: tuple(q + 1 for q in v) for k, v in qmap_0_ind.items()}
            logger.info(f"Calculated qmap: {self.qmap}")

        # convert circuit to required ITN format
        itn_circ = qiskit_circ_to_itn_circ_2d(qc, qmap=self.qmap)

        # build ITN graph from the Qiskit circuit
        cx_gates = extract_cx_gates(itn_circ)
        g = jl.build_graph_from_gates(jl.seval(cx_gates))
        s = jl.siteinds("S=1/2", g)

        # run simulation
        # extract output MPS and belief propagation cache (bpc)
        start = time.time()
        psi, bpc, errors = jl.tn_from_circuit(itn_circ, self.chi, s)
        logger.info(f"Estimated final state fidelity: {np.prod(1 - np.array(errors))}")
        logger.info(f"Time to generate tensor network state: {time.time() - start}")
        start = time.time()
        logger.info(f"Sampling {shots} shots from tensor network state")
        itn_samples = jl.sample_psi(psi, shots, self.proj_rank, self.norm_rank)
        logger.info(f"Time to sample to {shots} samples: {time.time() - start}")
        counts = itn_samples_to_counts_dict(itn_samples, self.qmap)

        return counts


def itn_run_skqd(qc, index, backend, shots, cmap):
    logger.info(f"Running circuit {index + 1}")
    return backend.run(qc, shots, cmap)