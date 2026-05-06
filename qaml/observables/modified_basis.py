from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import cpu_count
from typing import List, Sequence, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.qubit import project_operator_to_subspace

from qaml.skqd.basis_optimise import obp_modified_hamiltonian


def expectation_value_modified_basis(bs_matrix, betas, obs: SparsePauliOp,
                                     optimized_circuit: QuantumCircuit):

    zz_tilde = obp_modified_hamiltonian(obs, optimized_circuit)

    zz_proj = project_operator_to_subspace(bs_matrix, zz_tilde)

    psi = betas.astype(complex)
    Hpsi = zz_proj @ psi
    val = np.vdot(psi, Hpsi)

    return val.real


def _expectation_worker(obs, bs_matrix, betas, opt_qc):
    return expectation_value_modified_basis(bs_matrix, betas, obs, opt_qc)


def expectation_value_modified_basis_parallel(
    bs_matrix,
    betas,
    observables: Sequence[Any],
    opt_qc: Any,
    max_workers: int | None = None,
    chunksize: int = 8,
    show_progress: bool = True,
) -> List[complex]:
    """
    Compute expectation values in parallel for a list/sequence of observables.
    """

    if max_workers is None:
        max_workers = cpu_count()

    can_measure = hasattr(observables, "__len__")
    total = len(observables) if can_measure else None

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fn = partial(_expectation_worker, bs_matrix=bs_matrix, betas=betas, opt_qc=opt_qc)
        mapped = ex.map(fn, observables, chunksize=chunksize)  # preserves order

        if show_progress and can_measure:
            results: List[complex] = []
            for k, val in enumerate(mapped, start=1):
                print(f"Computing observable {k} of {total}")
                results.append(val)
        else:
            results = list(mapped)

    return results
