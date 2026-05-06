"""
Optimisation using operator backpropagation (OBP) to simulate the circuit and parameter shift
rule to calculate gradients.
"""
import logging
import pickle
from pathlib import Path
from functools import partial

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_obp import backpropagate
from qiskit_addon_sqd.qubit import project_operator_to_subspace
from qiskit_addon_utils.slicing import slice_by_depth

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_variational_circuit(n_qubits: int, params: np.ndarray, edges) -> QuantumCircuit:
    """
    Construct a circuit where for each edge we add a parameterised 2q block which resolves to
    the identity for θ=0.
    :return:
    """
    qc = QuantumCircuit(n_qubits)
    for i, edge in enumerate(edges):
        qc.cx(edge[0], edge[1])
        qc.ry(params[i], edge[0])
        qc.cx(edge[0], edge[1])

    return qc


def build_left_subcircuit(n_qubits: int, params: np.ndarray, edges, i: int) -> QuantumCircuit:
    """
    Build the left subcircuit L.
    """
    qc = QuantumCircuit(n_qubits)
    # full blocks strictly before i
    for j in range(0, i):
        e = edges[j]
        qc.cx(e[0], e[1])
        qc.ry(params[j], e[0])
        qc.cx(e[0], e[1])
    return qc


def build_right_subcircuit(n_qubits: int, params: np.ndarray, edges, i: int) -> QuantumCircuit:
    """
    Build the right subcircuit RM.
    """
    qc = QuantumCircuit(n_qubits)
    for j in range(i, len(params)):
        e = edges[j]
        qc.cx(e[0], e[1])
        qc.ry(params[j], e[0])
        qc.cx(e[0], e[1])
    return qc


def obp_modified_hamiltonian(H: SparsePauliOp, circuit: QuantumCircuit) -> SparsePauliOp:
    """
    Compute H_tilde = U H U^\dagger using OBP backpropagation.
    """
    if len(circuit.data) == 0:
        return H
    slices = slice_by_depth(circuit, max_slice_depth=1)
    H_tilde, remaining_slices, metadata = backpropagate(H, slices)

    if remaining_slices:
        logger.info(f"[OBP] Warning: {len(remaining_slices)} slices not backpropagated.")

    return H_tilde


def _int_conversion_from_bts_matrix_vmap(bitstring_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a matrix of bitstrings (booleans) into their integer values.

    Each row is interpreted as a binary string. For example:
        [[False, False],
         [False, True],
         [True,  False],
         [True,  True]]
    becomes [0, 1, 2, 3].
    """
    # Ensure boolean -> int (False->0, True->1)
    bits = bitstring_matrix.astype(np.uint8)

    # Precompute powers of two with MSB on the left
    # e.g. for 2 qubits -> [2**1, 2**0] = [2, 1]
    n_qubits = bits.shape[1]
    powers = (2**np.arange(n_qubits - 1, -1, -1))

    # Row wise dot product gives integer for each bitstring
    return bits @ powers


# remember the input to the optimization is actually the ground state obtained from SKQD. Coefficients and bitstrings
def sort_bitstrings_and_betas(bitstring_matrix: np.ndarray, betas: np.ndarray):
    """
    Sort bitstrings by their integer value and reorder betas to match.
    """
    bsmat_asints = _int_conversion_from_bts_matrix_vmap(bitstring_matrix)
    _, indices = np.unique(bsmat_asints, return_index=True)

    bitstring_sorted = bitstring_matrix[indices, :]
    betas_sorted = betas[indices]

    return bitstring_sorted, betas_sorted


def cost(
    params: np.ndarray,
    H: SparsePauliOp,
    bitstring_matrix: np.ndarray,
    betas: np.ndarray,
    edges,
    variational_circuit: QuantumCircuit = None,
    verbose: bool = False,
) -> float:
    """
    Compute C(theta) = <psi| H_tilde(theta) |psi>
    where:
        |psi> = sum_j beta_j |b_j>
        H_tilde(theta) = U(theta) H U(theta)^\dagger
        |b_j> are computational basis states
        betas do NOT depend on theta

    Arguments:
        params: flat vector, length = 3*n_qubits
        H: original SparsePauliOp Hamiltonian
        bitstring_matrix: shape (L, n_qubits), dtype=bool
        betas: shape (L,), complex amplitudes
    """

    n_qubits = H.num_qubits

    # 1. Build circuit U(theta)
    if variational_circuit is None:
        circuit = build_variational_circuit(n_qubits, params, edges)
    else:
        circuit = variational_circuit
    # 2. Compute H_tilde(theta)
    H_tilde = obp_modified_hamiltonian(H, circuit)

    # 3. Sort bitstrings and match betas
    # THIS SEEMS TO TOTALLY BREAK THE INITIAL ENERGY FOR N=115
    # bitstring_matrix, betas = sort_bitstrings_and_betas(bitstring_matrix, betas)

    # 4. Project H_tilde into the subspace spanned by these bitstrings
    H_proj = project_operator_to_subspace(bitstring_matrix, H_tilde)
    # H_proj is an L x L SciPy sparse matrix

    # 5. Compute <psi| H_proj |psi>
    psi = betas.astype(complex)
    Hpsi = H_proj @ psi
    cost_val = np.vdot(psi, Hpsi)  # ψ† \tilde{H} ψ)
    try:
        return float(np.real_if_close(cost_val))
    except TypeError:
        logger.info(f"Cost {cost_val} had large complex part but discarding that anyway")
        return float(np.real(cost_val))
    

def _pauli_string_mul(p1: str, p2: str):
    """
    Multiply two Pauli strings p1 * p2 (both length n, qubit-0 is leftmost char).
    Returns (phase, result_string) where phase is one of {1, -1, 1j, -1j}.
    This uses the single-qubit multiplication table:
        I*P = P, P*I = P
        X*X = I, Y*Y = I, Z*Z = I
        X*Y =  i Z, Y*X = -i Z
        Y*Z =  i X, Z*Y = -i X
        Z*X =  i Y, X*Z = -i Y
    """
    assert len(p1) == len(p2), "Pauli strings must be same length"
    phase = 1+0j
    out = []
    for a, b in zip(p1, p2):
        if a == 'I':
            out.append(b)
            continue
        if b == 'I':
            out.append(a)
            continue
        if a == b:
            out.append('I')
            continue
        # Now a != b and neither is I. Handle the 6 remaining cases.
        if a == 'X' and b == 'Y':
            out.append('Z'); phase *= 1j
        elif a == 'Y' and b == 'X':
            out.append('Z'); phase *= -1j
        elif a == 'Y' and b == 'Z':
            out.append('X'); phase *= 1j
        elif a == 'Z' and b == 'Y':
            out.append('X'); phase *= -1j
        elif a == 'Z' and b == 'X':
            out.append('Y'); phase *= 1j
        elif a == 'X' and b == 'Z':
            out.append('Y'); phase *= -1j
        else:
            raise ValueError(f"Unhandled Pauli multiply: {a} * {b}")
    return phase, ''.join(out)


def _commutator_sparse_pauliop(single_pauli_label: str, spop) -> 'SparsePauliOp':
    """
    Compute the commutator [P_single, spop] where P_single is a single Pauli string
    (e.g. 'IIYI...') and spop is a qiskit.quantum_info.SparsePauliOp.

    Returns a new SparsePauliOp representing the commutator.
    This is done purely at the Pauli-string level and accumulates coefficients.
    """
    # Use to_list() to get (label, coeff) pairs from the SparsePauliOp
    terms = spop.to_list()

    coeffs = {}
    for label, c in terms:
        # p1 = single_pauli_label * label
        ph1, lab1 = _pauli_string_mul(single_pauli_label, label)
        # p2 = label * single_pauli_label
        ph2, lab2 = _pauli_string_mul(label, single_pauli_label)

        # contribution: c * (ph1 * |lab1> - ph2 * |lab2>)
        # accumulate in dict for labels
        val1 = c * ph1
        val2 = c * ph2

        coeffs[lab1] = coeffs.get(lab1, 0+0j) + val1
        coeffs[lab2] = coeffs.get(lab2, 0+0j) - val2

    # remove near-zero entries
    out_list = []
    for lab, coeff in coeffs.items():
        if abs(coeff) > 1e-12:
            out_list.append((lab, complex(coeff)))

    if not out_list:
        # return zero operator
        return SparsePauliOp.from_list([('I'*spop.num_qubits, 0.0 )])
    return SparsePauliOp.from_list(out_list)


def calc_grads_analytic(
    params: np.ndarray,
    H: SparsePauliOp,
    bitstring_matrix: np.ndarray,
    betas: np.ndarray,
    edges,
) -> np.ndarray:
    """
    Calculates the gradients of the parameters analytically.

    For each parameter i:
      1. Build left-circuit L using parameters[:i] and edges[:i] and compute L^† H L via OBP.
      2. Form the commutator [Y_qi, L^† H L] at the Pauli-string level.
      3. Build the right-circuit Rpart that contains the rotation at i and all gates to its right
         (i.e. parameters[i:] and edges[i:]) and OBP-backpropagate the commutator through that
         right-circuit to obtain dHtilde/dtheta in the full circuit frame.
      4. Project the resulting operator into the sampled subspace and evaluate the Rayleigh form
         with the SKQD coefficients.

    This implementation avoids two projections per parameter.
    """

    n_qubits = H.num_qubits
    num_params = len(params)

    psi = betas.astype(complex)
    grads = np.zeros(num_params, dtype=float)

    for i in range(num_params):
        # 1) compute L^\dag H L by backpropagating H through the left-subcircuit
        left_circ = build_left_subcircuit(n_qubits, params, edges, i)
        LdHL = obp_modified_hamiltonian(H, left_circ)  # SparsePauliOp
        
        # 2) build local generator and compute local commutator (assuming Ansatz with disjoint CNOTs)
        # Note: local generator must be modified for more complex CNOT networks.
        control_index = edges[i][0]
        target_index = edges[i][1]
        label = ['I'] * n_qubits
        label[n_qubits -1 - control_index] = 'Y'
        label[n_qubits -1 - target_index] = 'X'
        yx_label = ''.join(label)
        comm_local = _commutator_sparse_pauliop(yx_label, LdHL)

        # 3) backpropagate the commutator through the right-subcircuit
        right_circ = build_right_subcircuit(n_qubits, params, edges, i)

        # dHtilde/dtheta = (i/2) * (right_circ)^\dagger * comm_local * (right_circ)
        dHtilde = obp_modified_hamiltonian(comm_local, right_circ)
        dHtilde = (0.5j) * dHtilde

        # 4) project and evaluate
        dHproj = project_operator_to_subspace(bitstring_matrix, dHtilde)
        val = np.vdot(psi, dHproj @ psi)
        grads[i] = float(np.real_if_close(val))

    return grads


def calc_grads_finite_diff(
        params: np.ndarray,
        H: SparsePauliOp,
        bitstring_matrix: np.ndarray,
        betas: np.ndarray,
        edges,
        ):
    num_params = len(params)
    grads = np.empty(num_params, dtype=float)
    for i in range(num_params):
        
        # Shifted by + pi/2
        shifted = params.copy()
        shifted[i] += np.pi / 2
        fp = cost(shifted, H, bitstring_matrix, betas, edges)
        
        # Shifted by - pi/2
        shifted = params.copy()
        shifted[i] -= np.pi / 2
        fm = cost(shifted, H, bitstring_matrix, betas, edges)
        grads[i] = (fp - fm) / 2.0

    return grads

def calc_grads_factory(
        method: str,
        H: SparsePauliOp,
        bitstring_matrix: np.ndarray,
        betas: np.ndarray,
        edges):
    
    if method == "finite_diff":
        func = calc_grads_finite_diff
    elif method == "analytic":
        func = calc_grads_analytic
    else:
        raise NotImplementedError
        
    return partial(func, 
                       H=H, 
                       bitstring_matrix=bitstring_matrix,
                       betas=betas,
                       edges=edges)

def gradient(
    num_qubits,
    fun1,
    calc_grads: callable,
    params,
    numparams,
    eta,
    eps_grad,
    count_num,
    gradcount,
    checkpoint_dir=None,
    costglobal=None,
    stop_on_global_increase=False,
    global_tol=0.0,
):
    """
    num_qubits : number of qubits in ansatz
    fun1 : cost(params)
    calc_grads (callable): Method for the gradient calculations.
    params : initial parameter vector
    numparams : len(params)
    eta : learning rate
    eps_grad : gradient-norm stopping threshold
    count_num : max iterations
    gradcount : how many times to allow vanishing gradients
    checkpoint_dir: directory to save checkpoints, should be a unique experiment name
    costglobal: Boolean if to calculate and log/save the cost over all bitstrings each iter
    stop_on_global_increase: Exit optimisation early if the global cost goes up
    global_tol: By what value should the global cost rise before we exit early
    """

    def _write_checkpoint(step, stopped=False):
        if not checkpoint_dir:
            return
        results["best_iter"] = best_iter
        results["best_params"] = best_params
        results["best_global_cost"] = best_global
        if stopped:
            results["stopped_on_global_increase"] = True
        write_dir = Path(f"saved_opt_res/{checkpoint_dir}")
        write_dir.mkdir(parents=True, exist_ok=True)
        with open(f"{write_dir}/{step}.pkl", "wb") as f:
            pickle.dump(results, f)

    gcount = 0
    count = 0
    param_current = params.copy()

    cost_value = fun1(num_qubits, param_current, numparams)
    global_value = costglobal(num_qubits, param_current, numparams) if costglobal else None

    best_iter = 0
    best_params = param_current.copy()
    best_global = global_value

    results = {
        0: {"cost": cost_value, "params": param_current.copy(), "global_cost": global_value}
    }
    logger.info(f"{count}, cost:{cost_value}, global cost:{global_value}")
    _write_checkpoint(0)

    while count < count_num and gcount < gradcount:
        prev_params = param_current.copy()
        prev_global = global_value

        grad_vec = calc_grads(param_current)
        grad_norm_sq = float(np.dot(grad_vec, grad_vec))
        if grad_norm_sq <= eps_grad:
            gcount += 1

        param_proposed = param_current - eta * grad_vec
        cost_new = fun1(num_qubits, param_proposed, numparams)

        if cost_value - cost_new >= eta * grad_norm_sq:
            eta *= 2.0
            param_current = param_proposed
        elif cost_value - cost_new < (eta / 2.0) * grad_norm_sq:
            eta /= 2.0
            param_current = param_current - eta * grad_vec
        else:
            param_current = param_proposed

        cost_value = fun1(num_qubits, param_current, numparams)
        global_value = costglobal(num_qubits, param_current, numparams) if costglobal else None

        count += 1
        results[count] = {"cost": cost_value, "params": param_current.copy(),
                          "global_cost": global_value}
        logger.info(f"{count}, cost:{cost_value}, global cost:{global_value}")

        if stop_on_global_increase and prev_global is not None and global_value is not None:
            if global_value > prev_global + global_tol:
                best_iter = count - 1
                best_params = prev_params
                best_global = prev_global
                _write_checkpoint(count, stopped=True)
                break

        # Update best_params based on global cost if available
        if global_value is not None and (
            best_global is None or global_value < best_global - global_tol):
            best_global = global_value
            best_params = param_current.copy()
            best_iter = count
        elif global_value is None:
            # When costglobal is None, always track the latest iteration as best
            best_params = param_current.copy()
            best_iter = count

        _write_checkpoint(count)

    results["best_iter"] = best_iter
    results["best_params"] = best_params
    results["best_global_cost"] = best_global
    if checkpoint_dir:
        _write_checkpoint(count)
    return results

def gradient_cpu(
    num_qubits,
    fun1,
    fun2,
    params,
    numparams,
    eta,
    eps_grad,
    eps_cost,
    count_num,
    gradcount,
    qc=None,
    pool=None,
):
    """
    num_qubits: number of qubits in ansatz
    func1: Cost function
    func2: mutliprocessing cost function worker for parameter shift
    params: initial parameter vector
    numparams: len(params)
    eta: learning rate
    eps_grad: gradient-norm stopping threshold
    eps_cost: cost stopping threshold
    count_num: max iterations
    gradcount: how many times to allow vanishing gradients
    qc: optional QuantumCircuit to pass to cost functions
    pool: multiprocessing pool for parallel gradient computation
    """

    if pool is None:
        raise ValueError(
            "A multiprocessing pool must be provided for parallel gradient."
        )

    gcount = 0
    count = 0

    param_current = params.copy()
    cost_value = fun1(num_qubits, param_current, numparams, qc)

    results = {}
    results[count] = {"cost": cost_value, "params": param_current.copy()}
    print(count, cost_value)

    while count < count_num and gcount < gradcount:
        # Parallel gradient computation
        tasks_p = [
            (num_qubits, param_current, numparams, i, +1) for i in range(numparams)
        ]
        tasks_m = [
            (num_qubits, param_current, numparams, i, -1) for i in range(numparams)
        ]

        vals_p = pool.map(fun2, tasks_p)
        vals_m = pool.map(fun2, tasks_m)

        grad_vec = (np.array(vals_p) - np.array(vals_m)) / 2.0
        grad_norm_sq = np.linalg.norm(grad_vec) ** 2

        if grad_norm_sq <= eps_grad:
            gcount += 1

        # parameter update
        param_proposed = param_current - eta * grad_vec
        cost_new = fun1(num_qubits, param_proposed, numparams, qc)

        # Adaptive learning rate
        if cost_value - cost_new >= eta * grad_norm_sq:
            eta *= 2
            param_current = param_proposed
        elif cost_value - cost_new < (eta / 2) * grad_norm_sq:
            eta /= 2
            param_current = param_current - eta * grad_vec
        else:
            param_current = param_proposed

        cost_value = fun1(num_qubits, param_current, numparams, qc)

        count += 1
        results[count] = {"cost": cost_value, "params": param_current.copy()}
        print(count, cost_value)

    return results

def gradient_freya(
    num_qubits,
    fun1,
    fun2,
    fun3,
    params,
    numparams,
    eta,
    eps_grad,
    eps_cost,
    count_num,
    gradcount,
    qc=None,
    write_fn=None,
):
    """
    num_qubits : number of qubits in ansatz
    fun1 : cost(params)
    fun2 : cost(params + pi/2 shift on index i)
    fun3 : cost(params - pi/2 shift on index i)
    params : initial parameter vector
    numparams : len(params)
    eta : learning rate
    eps_grad : gradient-norm stopping threshold
    eps_cost : cost stopping threshold
    count_num : max iterations
    gradcount : how many times to allow vanishing gradients
    qc : optional QuantumCircuit to pass to cost functions
    """

    # Start
    gcount = 0
    count = 0

    param_current = params.copy()
    cost_value = fun1(num_qubits, param_current, numparams, qc)

    results = {}
    results[count] = {"cost": cost_value, "params": param_current.copy()}
    print(count, cost_value)

    # Gradient descent loop
    while count < count_num and gcount < gradcount:
        # 1. Compute gradient vector via parameter-shift rule
        grad_vec = np.zeros(numparams)
        for i in range(numparams):
            fp = fun2(num_qubits, param_current, numparams, i, qc)  # +pi/2
            fm = fun3(num_qubits, param_current, numparams, i, qc)  # -pi/2
            grad_vec[i] = (fp - fm) / 2  # shift rule

        grad_norm_sq = np.linalg.norm(grad_vec) ** 2

        if grad_norm_sq <= eps_grad:
            gcount += 1

        # 2. Proposed parameter update
        param_proposed = param_current - eta * grad_vec

        # 3. Evaluate cost at proposed point
        cost_new = fun1(num_qubits, param_proposed, numparams, qc)

        # 4. Adaptive learning rate logic
        if cost_value - cost_new >= eta * grad_norm_sq:
            eta *= 2
            param_current = param_proposed
        elif cost_value - cost_new < (eta / 2) * grad_norm_sq:
            eta /= 2
            param_current = param_current - eta * grad_vec
        else:
            param_current = param_proposed

        # 5. Update cost and save result
        cost_value = fun1(num_qubits, param_current, numparams, qc)

        count += 1
        results[count] = {"cost": cost_value, "params": param_current.copy()}
        print(count, cost_value)

        if write_fn:
            exp_dir = write_fn.replace(".pkl", "")
            write_dir = Path(f"{exp_dir}")
            write_dir.mkdir(parents=True, exist_ok=True)
            with open(f"{write_dir}/{count}.pkl", "wb") as f:
                pickle.dump(results, f)

    return results