import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_sqd.qubit import project_operator_to_subspace

from qaml.skqd.basis_optimise import obp_modified_hamiltonian


def shots_format(num):
    num = float(num)
    for unit in ["", "k", "m"]:
        if abs(num) < 1000:
            return f"{num:g}{unit}"
        num /= 1000

    return f"{num:.0E}"

def energy_and_variance_in_obp_subspace(
    H: SparsePauliOp,
    circuit,
    bitstrings: np.ndarray,
    betas: np.ndarray
):
    psi = betas.astype(complex)
    nrm = np.vdot(psi, psi)
    if not np.isclose(nrm, 1.0):
        psi = psi / np.sqrt(nrm)

    # Energy of the basis-optimised solution
    H_tilde = obp_modified_hamiltonian(H, circuit)
    H_proj_OBP = project_operator_to_subspace(bitstrings, H_tilde)
    Hpsi_OBP = H_proj_OBP @ psi
    E_vopt = np.vdot(psi, Hpsi_OBP)
    E_vopt = float(np.real_if_close(E_vopt))

    # Variance with respect to the "true" Hamiltonian
    H_proj_true = project_operator_to_subspace(bitstrings, H)
    Hpsi_true = H_proj_true @ psi
    E_true = np.vdot(psi, Hpsi_true)
    H2_true = np.vdot(Hpsi_true, Hpsi_true)
    E_true = float(np.real_if_close(E_true))
    H2_true = float(np.real_if_close(H2_true))
    var_true = H2_true - E_true**2
    var_true = float(np.real_if_close(var_true))

    return E_vopt, var_true
