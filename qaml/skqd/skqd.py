import logging
import multiprocessing
import os
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Collection

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import generate_preset_pass_manager
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.backend import Backend
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import Result
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import CouplingMap
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.qubit import solve_qubit
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qaml.skqd.custom_backends import ITNBackend, itn_run_skqd
from qaml.trotter_circuits.heisenberg_nd import trotter_step_coloured

logger = logging.getLogger(__name__)


class SKQDSampler:
    def __init__(
        self,
        n_qubits: int,
        krylov_dim: int,
        dt: float,
        num_trotter_steps: int,
        qc_state_prep: QuantumCircuit,
        H_op: SparsePauliOp,
        backend: Backend,
        shots: int,
        cmap: CouplingMap = None,
        colouring: str = "qiskit",
        num_processes: int = 1,
        error_mitigation: dict[str, bool] = None,
        parallelise_itn: bool = False,
        u_opt: QuantumCircuit = None,
    ):
        self.n_qubits = n_qubits
        self.krylov_dim = krylov_dim
        self.dt = dt
        self.num_trotter_steps = num_trotter_steps
        self.qc_state_prep = qc_state_prep
        self.H_op = H_op
        self.jz = (H_op.coeffs[2]).real
        self.cmap = cmap
        self.colouring = colouring
        self.backend = backend
        self.shots = shots
        self.num_processes = num_processes
        self.error_mitigation = error_mitigation
        self.parallelise_itn = parallelise_itn
        self.u_opt = u_opt

    def _construct_circuits(self) -> List[QuantumCircuit]:
        """Construct quantum circuits to generate the Krylov states.

        Returns:
            List[QuantumCircuit]: List of circuits to sample from for each Krylov state.
        """
        if self.colouring == "qiskit":
            evol_gate = PauliEvolutionGate(
                self.H_op,
                time=(self.dt / self.num_trotter_steps),
                synthesis=SuzukiTrotter(order=1, reps=self.num_trotter_steps, preserve_order=False),
            )
            qc_evol = QuantumCircuit(self.n_qubits)
            qc_evol.append(evol_gate, range(self.n_qubits))
        elif self.colouring == "nx":
            logger.info("NX colouring only implemented for XXZ model")
            logger.info(f"Using assumed value Jz={self.jz}")
            qc_evol = trotter_step_coloured(self.cmap, self.dt, self.jz)
        else:
            raise NotImplementedError("Colouring options only 'qiskit' or 'nx'")

        circuits = []
        for rep in range(1, self.krylov_dim):
            circ = self.qc_state_prep.copy()

            for _ in range(rep):
                circ.compose(other=qc_evol, inplace=True)

            if self.u_opt is not None:
                circ.compose(other=self.u_opt, inplace=True)

            circ.measure_all()
            circuits.append(circ)

        circuits = transpile(
            circuits,
            basis_gates=["cx", "rx", "ry", "rz"],
            num_processes=self.num_processes,
        )
        logger.info(
            f"Largest circuit with Krylov dim {self.krylov_dim} has CX depth "
            f"{circuits[-1].depth(lambda g: len(g.qubits) > 1)} and num CX "
            f"{circuits[-1].count_ops().get('cx')}"
        )
        return circuits

    def submit_job(self):
        """
        Submits the SKQD circuits to whatever backend is being used. In the case of a simulated
        backend, this will immediately run the simulation to obtain the samples.
        """

        circuits = self._construct_circuits()
        if isinstance(self.backend, ITNBackend):
            if self.parallelise_itn:
                num_cores = int(os.environ.get("NUM_CORES"))  # If on cluster
                logger.warning(f"Number of cores:{num_cores}")
                pool = multiprocessing.Pool(num_cores)
                futures = []
                for i, qc in enumerate(circuits):
                    future = pool.apply_async(itn_run_skqd,
                                              args=(qc, i, self.backend, self.shots, self.cmap))
                    futures.append(future)
                samples = [future.get() for future in futures]
                pool.close()
                pool.join()
            else:
                samples = [itn_run_skqd(qc, i, self.backend, self.shots, self.cmap) for i, qc in
                           enumerate(circuits)]
            return samples
        elif self.backend.name == "aer_simulator_matrix_product_state":
            if self.dt > 0.15:
                logger.warning("Large values of dt can make the Aer MPS simulator silently die!")
            return self.backend.run(circuits, shots=self.shots)
        else:
            pm = generate_preset_pass_manager(
                backend=self.backend, optimization_level=3
            )
            isa_circuits = pm.run(circuits=circuits)

            sampler = Sampler(mode=self.backend)
            # Set up error mitigation for the sampler
            if self.error_mitigation is not None:
                sampler.options.dynamical_decoupling.enable = self.error_mitigation.get("dd", False)
                logger.info(f"Dynamical decoupling: {sampler.options.dynamical_decoupling.enable}")
                sampler.options.twirling.enable_gates = self.error_mitigation.get("pt", False)
                logger.info(f"Twirling: {sampler.options.twirling.enable_gates}")
            return sampler.run(isa_circuits, shots=self.shots)


class SKQDPostprocessor:
    def __init__(
        self,
        job: Result | Collection[Result],
        krylov_dim: int,
        H_op: SparsePauliOp,
        backend_name: str,
        scipy_kwargs: dict,
        num_ones: int,
        config_recovery_method: str
    ):
        self.job = job
        self.krylov_dim = krylov_dim
        self.H_op = H_op
        self.backend_name = backend_name
        self.scipy_kwargs = scipy_kwargs
        self.num_ones = num_ones
        self.config_recovery_method = config_recovery_method
        self._config_recovery = self._config_recovery_factory()

    def _get_counts_all(self):
        if self.backend_name == "aer_simulator_matrix_product_state":
            counts_all = self.job.result().get_counts()
        elif self.backend_name == "ITensorNetworks":
            counts_all = self.job
        else:
            if isinstance(self.job, Collection):
                counts_jobs = []
                for job in self.job:
                    counts_jobs.append([
                        job.result()[k].data.meas.get_counts()
                        for k in range(self.krylov_dim - 1)
                    ])
                counts_all = [k for job in counts_jobs for k in job]
            else:

                counts_all = [
                    self.job.result()[k].data.meas.get_counts()
                    for k in range(self.krylov_dim - 1)
                ]

        return counts_all

    def _calc_counts_cumulative_list(self, counts_all):
        counts_cumulative_list = []
        for i in range(1, self.krylov_dim):
            counter = Counter()
            for d in counts_all[: i + 1]:
                counter.update(d)

            counts = dict(counter)
            counts_cumulative_list.append(counts)

        return counts_cumulative_list

    def _config_recovery_factory(self) -> callable:
        if self.config_recovery_method == "post_select":
            func = self._postselect_counts
        elif self.config_recovery_method == "random_flip":
            func = self._random_flip_recovery
        else:
            raise NotImplementedError

        return func

    def _random_flip_recovery(self, counts: dict) -> str:
        def random_flip_bitstring(bitstring: str, diff: int):
            if diff == 0:
                return bitstring  # already matches

            bits = list(bitstring)

            if diff < 0:
                # Too many 1s → flip |diff| ones to zeros
                ones_positions = [i for i, b in enumerate(bits) if b == "1"]
                flip_positions = random.sample(ones_positions, k=-diff)
                for i in flip_positions:
                    bits[i] = "0"
            else:
                # Too few 1s → flip diff zeros to ones
                zeros_positions = [i for i, b in enumerate(bits) if b == "0"]
                flip_positions = random.sample(zeros_positions, k=diff)
                for i in flip_positions:
                    bits[i] = "1"

            return "".join(bits)

        filtered_in_counts = defaultdict(int)
        num_filtered_out = 0
        for bitstring, freq in counts.items():
            diff = self.num_ones - bitstring.count("1")
            if diff == 0:
                filtered_in_counts[bitstring] = freq
            else:
                bitstring_recovered = random_flip_bitstring(bitstring, diff)
                filtered_in_counts[bitstring_recovered] += freq
                num_filtered_out += freq

        logger.info(f"{num_filtered_out} bitstrings filtered out")
        return filtered_in_counts

    def _postselect_counts(self, counts: dict) -> dict:
        """Filters out bitstrings that do not have specified number (`num_ones`) of `1` bits, e.g. half of the number of spins.

        Args:
            counts (dict): Counts of the bitstrings.

        Returns:
            dict: Filtered Counts
        """
        filtered_in_counts = {}
        num_filtered_out = 0
        for bitstring, freq in counts.items():
            if bitstring.count("1") == self.num_ones:
                filtered_in_counts[bitstring] = freq
            else:
                num_filtered_out += freq

        logger.info(f"{num_filtered_out} bitstrings filtered out")
        return filtered_in_counts

    def _project_and_diagonalize(
        self, counts: Dict[str, int]
    ) -> Tuple[float, np.ndarray]:
        """Project the Hamiltonian into the subspace spanned by the sampled computational basis states and classically diagonalize.

        Args:
            counts (Dict[str, int]): Cumulative counts of sampled bitstrings up to the Krylov dimension.

        Returns:
            Tuple[float, np.ndarray]: Ground state energy and the ground state amplitudes.
        """

        bitstring_matrix, probs = counts_to_arrays(counts=counts)
        logger.info(f"Bitstring matrix shape {bitstring_matrix.shape}")

        eigenvals, eigenstates = solve_qubit(
            bitstring_matrix, self.H_op, **self.scipy_kwargs
        )
        gs_en = np.min(eigenvals)
        ground_state_amp = eigenstates[:, np.argmin(eigenvals)]

        # Normalize the ground state amplitudes
        ground_state_amp /= np.sqrt(np.sum(np.abs(ground_state_amp)**2))

        return gs_en, ground_state_amp

    def run(self):
        counts_all = self._get_counts_all()
        return self.run_from_counts_all(counts_all)
    
    def run_from_counts_all(self, counts_all):
        """Run post-processing from pre-loaded counts_all.
        
        This method is useful when counts_all has been saved to disk and you want to
        avoid fetching the job from the online service.
        
        Args:
            counts_all: List of count dictionaries for each Krylov dimension
            
        Returns:
            Tuple[float, dict, dict]: Ground state energy, ground state dictionary, and counts
        """
        counts_cumulative_list = self._calc_counts_cumulative_list(counts_all)

        # Take the cumulative counts at the last Krylov dimension and perform post selection to filter out bitstrings.
        counts_cumulative = counts_cumulative_list[-1]
        logger.info(f"{len(counts_cumulative)} unique bitstrings sampled")
        counts_cumulative = self._config_recovery(counts_cumulative)
        logger.info(f"{len(counts_cumulative)} unique bitstrings to diagonalise")
        gs_en, ground_state_amp = self._project_and_diagonalize(counts_cumulative)

        # Sort the bitstrings by unsigned integer value.
        sorted_bitstrings = sorted(counts_cumulative.keys(), key=lambda b: int(b, 2))

        # Construct the state dictionary.
        ground_state_dict = {
            k: ground_state_amp[i] for i, k in enumerate(sorted_bitstrings)
        }

        return gs_en, ground_state_dict, counts_cumulative

    def run_all(self):
        counts_all = self._get_counts_all()
        counts_cumulative_list = self._calc_counts_cumulative_list(counts_all)
        # Take the cumulative counts for each Krylov dimension and calculate the ground state and its energy.
        ground_state_energies = []
        ground_state_dict_list = []
        for counts_cumulative in counts_cumulative_list:
            counts_cumulative = self._config_recovery(counts_cumulative)
            gs_en, ground_state_amp = self._project_and_diagonalize(counts_cumulative)

            # Sort the bitstrings by unsigned integer value.
            sorted_bitstrings = sorted(
                counts_cumulative.keys(), key=lambda b: int(b, 2)
            )

            # Construct the state dictionary.
            ground_state_dict = {
                k: ground_state_amp[i] for i, k in enumerate(sorted_bitstrings)
            }

            ground_state_energies.append(gs_en)
            ground_state_dict_list.append(ground_state_dict)

        return ground_state_energies, ground_state_dict_list


class SKQDRunner:
    """Runs Sample-based Krylov Quantum Diagonalisation (https://arxiv.org/abs/2501.09702v2)

    Args:
        n_qubits (int): Number of qubits, e.g. number of spins.
        krylov_dim (int): Number of Krylov states.
        dt (float): Time step between two Krylov states.
        num_trotter_steps (int): Number of Trotter steps.
        qc_state_prep (QuantumCircuit): Quantum circuit to prepare a reference state (initial state).
        H_op (SparsePauliOp): Hamiltonian.
        backend (Backend): Qiskit Backend to sample.
        scipy_kwargs (dict): arguments for classical diagonalisation within the subspace, legacy library Scipy.
        shots (int): Number of shots for each Krylov state.
        num_ones (int): Number of qubits in the one state in a bitstring. (Conserved quantity)
    """

    def __init__(
        self,
        n_qubits: int,
        krylov_dim: int,
        dt: float,
        num_trotter_steps: int,
        qc_state_prep: QuantumCircuit,
        H_op: SparsePauliOp,
        backend: Backend,
        scipy_kwargs: dict,
        shots: int,
        num_ones: int,
        cmap: CouplingMap = None,
        colouring: str = "qiskit",
        num_processes: int = 1,
        error_mitigation: dict[str, bool] = None,
        config_recovery_method: str = 'post_select',
        parallelise_itn: bool = False,
    ):
        """Initialize the SKQDRunner."""
        self.n_qubits = n_qubits
        self.krylov_dim = krylov_dim
        self.dt = dt
        self.num_trotter_steps = num_trotter_steps
        self.qc_state_prep = qc_state_prep
        self.H_op = H_op
        self.backend = backend
        self.scipy_kwargs = scipy_kwargs
        self.shots = shots
        self.num_ones = num_ones
        self.cmap = cmap
        self.colouring = colouring
        self.num_processes = num_processes
        self.error_mitigation = error_mitigation
        self.config_recovery_method = config_recovery_method
        self.parallelise_itn = parallelise_itn

        self.sampler = SKQDSampler(
            n_qubits,
            krylov_dim,
            dt,
            num_trotter_steps,
            qc_state_prep,
            H_op,
            backend,
            shots,
            cmap,
            colouring,
            num_processes,
            error_mitigation,
            parallelise_itn,
        )

    def run(self):
        job = self.sampler.submit_job()
        self.postprocessor = SKQDPostprocessor(
            job, self.krylov_dim, self.H_op, self.backend.name, self.scipy_kwargs, self.num_ones,
            self.config_recovery_method
        )
        return self.postprocessor.run()

    def run_all(self):
        job = self.sampler.submit_job()
        self.postprocessor = SKQDPostprocessor(
            job, self.krylov_dim, self.H_op, self.backend.name, self.scipy_kwargs, self.num_ones,
            self.config_recovery_method
        )
        return self.postprocessor.run_all()


def project_and_diagonalize(
    counts: Dict[str, int], H_op, scipy_kwargs):
    """Project the Hamiltonian into the subspace spanned by the sampled computational basis states and classically diagonalize.

    Args:
        counts (Dict[str, int]): Cumulative counts of sampled bitstrings up to the Krylov dimension.

    Returns:
        Tuple[float, np.ndarray]: Ground state energy and the ground state amplitudes.
    """

    bitstring_matrix, probs = counts_to_arrays(counts=counts)
    logger.info(f"Bitstring matrix shape {bitstring_matrix.shape}")

    eigenvals, eigenstates = solve_qubit(
        bitstring_matrix, H_op, **scipy_kwargs
    )
    gs_en = np.min(eigenvals)
    ground_state_amp = eigenstates[:, np.argmin(eigenvals)]

    # Normalize the ground state amplitudes
    ground_state_amp /= np.sqrt(np.sum(np.abs(ground_state_amp)**2))

    return gs_en, ground_state_amp
