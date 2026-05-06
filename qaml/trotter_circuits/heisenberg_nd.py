import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


def generate_pauli_op(coupling_map, n_qubits):
    pauli_labels = []
    coeffs = []

    for i, j in coupling_map.get_edges():
        jx, jy, jz = [1, 1, 1]

        xx = ['I'] * n_qubits
        xx[i] = 'X'
        xx[j] = 'X'
        pauli_labels.append(''.join(xx))
        coeffs.append(jx)

        yy = ['I'] * n_qubits
        yy[i] = 'Y'
        yy[j] = 'Y'
        pauli_labels.append(''.join(yy))
        coeffs.append(jy)

        zz = ['I'] * n_qubits
        zz[i] = 'Z'
        zz[j] = 'Z'
        pauli_labels.append(''.join(zz))
        coeffs.append(jz)

    return SparsePauliOp(pauli_labels, coeffs)


def trotter_circuit_qiskit_colouring(n_qubits, cmap, time: float, reps: int = 1,
                                     preserve_order=False) -> QuantumCircuit:
    """Example usage
    backend = FakeSherbrooke()
    cm = CouplingMap(backend.configuration().coupling_map)
    n_qubits = backend.configuration().n_qubits
    circuit = trotter_evolution_circuit(n_qubits, cm, time=1.0, reps=2)
    """

    H = generate_pauli_op(cmap, n_qubits)

    evo_gate = PauliEvolutionGate(
        H,
        time=time,
        synthesis=SuzukiTrotter(order=1, reps=reps, preserve_order=preserve_order)
    )

    qc = QuantumCircuit(n_qubits)
    qc.append(evo_gate, range(n_qubits))
    return qc


def trotter_block(qc: QuantumCircuit, i: int, j: int,
                  dt: float, jx: float, jy: float, jz: float):
    alphas = np.asarray(
        [np.pi / 2 - 0.5 * jz * dt, 0.5 * jx * dt - np.pi / 2, np.pi / 2 - 0.5 * jy * dt])

    qc.rz(-np.pi / 2, j)
    qc.cx(j, i)
    qc.rz(alphas[0], i)
    qc.ry(alphas[1], j)
    qc.cx(i, j)
    qc.ry(alphas[2], j)
    qc.cx(j, i)
    qc.rz(np.pi / 2, i)


def trotter_step_coloured(coupling_map: CouplingMap, dt: float, jz: float):
    edges = coupling_map.get_edges()
    n = coupling_map.size()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    L = nx.line_graph(G)
    edge_to_color = nx.coloring.greedy_color(L, strategy='smallest_last')

    color_classes = {}
    for (u, v), col in edge_to_color.items():
        color_classes.setdefault(col, []).append((u, v))

    qc = QuantumCircuit(n)
    jx = jy = 1.0

    for color in sorted(color_classes):
        for (i, j) in color_classes[color]:
            trotter_block(qc, i, j, dt, jx, jy, jz)

    return qc


def trotter_circuit_nx_colouring(n, cmap, dt, reps=1) -> QuantumCircuit:
    qc = QuantumCircuit(n)

    layer = trotter_step_coloured(cmap, dt)
    for _ in range(reps):
        qc.compose(layer, inplace=True)
    return qc


# backend = FakeSherbrooke()
# cm = CouplingMap(backend.configuration().coupling_map)
# n_qubits = backend.configuration().n_qubits
# circuit = trotter_circuit_qiskit_colouring(n_qubits, cm, 1.0, 1, False)
# circuit2 = trotter_circuit_nx_colouring(n_qubits, cm, 1.0, 1)
# print(transpile(circuit, basis_gates=["cx","rx","ry","rz"]).depth(lambda gate: len(gate.qubits) > 1))
# print(transpile(circuit2, basis_gates=["cx","rx","ry","rz"]).depth(lambda gate: len(gate.qubits) > 1))