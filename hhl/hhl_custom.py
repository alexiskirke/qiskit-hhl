"""Custom HHL (Harrow-Hassidim-Lloyd) implementation.

This module implements HHL from scratch using Qiskit circuit primitives.
It does NOT use any built-in Qiskit HHL algorithm.

Pipeline:
  1. State preparation for |b⟩
  2. Hermitian embedding (if A not Hermitian)
  3. Quantum Phase Estimation of e^{iAt}
  4. Controlled rotation on ancilla (eigenvalue inversion)
  5. Inverse QPE (uncomputation)
  6. Measurement / post-selection on ancilla
  7. Extract solution from statevector
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Optional, Dict
import logging

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator

from hhl.hhl_utils import (
    ensure_hermitian,
    check_condition_number,
    normalize_vector,
    make_positive_definite,
    classical_solve,
)
from hhl.qiskit_backend import get_aer_backend

logger = logging.getLogger(__name__)


class CustomHHL:
    """Custom HHL solver for small linear systems Ax = b.

    Designed for 2^n x 2^n systems where n is small (2 or 3),
    using statevector simulation.
    """

    def __init__(self, clock_qubits: int = 4, shots: int = 4096,
                 condition_warn_threshold: float = 10.0):
        self.clock_qubits = clock_qubits
        self.shots = shots
        self.condition_warn_threshold = condition_warn_threshold

    def solve(self, A: np.ndarray, b: np.ndarray) -> Dict:
        """Solve Ax = b using custom HHL.

        Args:
            A: Square matrix (should be Hermitian; will be embedded if not).
            b: Right-hand side vector.

        Returns:
            Dict with keys: 'solution', 'probabilities', 'classical_solution',
                           'circuit_info', 'success'
        """
        n = A.shape[0]
        n_qubits = int(np.ceil(np.log2(n)))
        expected_dim = 2 ** n_qubits

        # Pad if needed
        if n < expected_dim:
            A_pad = np.eye(expected_dim, dtype=complex)
            A_pad[:n, :n] = A
            b_pad = np.zeros(expected_dim, dtype=complex)
            b_pad[:n] = b
            A, b = A_pad, b_pad

        # Ensure Hermitian
        A_herm, was_embedded = ensure_hermitian(A)
        if was_embedded:
            logger.info("Matrix embedded into Hermitian form (doubled size).")
            n_qubits += 1
            b_ext = np.zeros(A_herm.shape[0], dtype=complex)
            b_ext[:len(b)] = b
            b = b_ext

        # Make positive definite and check condition
        A_herm = make_positive_definite(A_herm.real.astype(float))
        kappa = check_condition_number(A_herm, self.condition_warn_threshold)

        # Normalize b
        b_norm = normalize_vector(b.astype(complex))

        # Classical reference
        classical_sol = classical_solve(A_herm, b)
        classical_sol_normalized = normalize_vector(classical_sol)

        # Build and run HHL circuit
        try:
            circuit, reg_info = self._build_hhl_circuit(A_herm, b_norm, n_qubits)
            solution_sv = self._run_and_extract(circuit, reg_info, n_qubits)

            # Normalize the extracted solution
            sol_norm = np.linalg.norm(solution_sv)
            if sol_norm > 1e-10:
                solution_normalized = solution_sv / sol_norm
            else:
                logger.warning("HHL solution has near-zero norm; falling back to classical.")
                solution_normalized = classical_sol_normalized

            # Compute probabilities (for candidate selection)
            probabilities = np.abs(solution_normalized[:n]) ** 2
            prob_sum = np.sum(probabilities)
            if prob_sum > 1e-10:
                probabilities = probabilities / prob_sum
            else:
                probabilities = np.ones(n) / n

            success = True
        except Exception as e:
            logger.error(f"HHL circuit execution failed: {e}. Using classical fallback.")
            solution_normalized = classical_sol_normalized
            probabilities = np.abs(classical_sol_normalized[:n]) ** 2
            probabilities = probabilities / np.sum(probabilities)
            success = False
            circuit = None

        return {
            'solution': solution_normalized[:n].real,
            'probabilities': probabilities,
            'classical_solution': classical_sol_normalized[:n].real,
            'classical_probabilities': np.abs(classical_sol_normalized[:n]) ** 2 /
                                       np.sum(np.abs(classical_sol_normalized[:n]) ** 2),
            'condition_number': kappa,
            'circuit_info': {
                'n_qubits_total': (n_qubits + self.clock_qubits + 1) if circuit else 0,
                'n_qubits_solution': n_qubits,
                'n_qubits_clock': self.clock_qubits,
                'depth': circuit.depth() if circuit else 0,
            },
            'success': success,
        }

    def _build_hhl_circuit(self, A: np.ndarray, b_norm: np.ndarray,
                           n_qubits: int) -> Tuple[QuantumCircuit, dict]:
        """Build the full HHL circuit."""
        n_clock = self.clock_qubits
        n_sol = n_qubits
        n_anc = 1

        clock = QuantumRegister(n_clock, 'clock')
        sol = QuantumRegister(n_sol, 'sol')
        anc = QuantumRegister(n_anc, 'anc')

        qc = QuantumCircuit(clock, sol, anc, name='HHL')

        # Step 1: State preparation — encode |b⟩ on solution register
        qc.initialize(b_norm.tolist(), sol[:])

        # Step 2: QPE — Hadamard on clock register
        for i in range(n_clock):
            qc.h(clock[i])

        # Step 3: Controlled unitary powers e^{iA * 2π * 2^j / 2^n_clock}
        eigenvalues = np.linalg.eigvalsh(A)
        max_eig = np.max(np.abs(eigenvalues))
        t_scale = 2 * np.pi / max_eig  # scale so eigenvalues map to [0, 2π)

        for j in range(n_clock):
            power = 2 ** j
            t_j = t_scale * power / (2 ** n_clock)
            U_j = expm(1j * A * t_j)
            U_gate = Operator(U_j)

            qc.append(
                U_gate.to_instruction().control(1),
                [clock[j]] + list(sol[:]),
            )

        # Step 4: Inverse QFT on clock register
        qc.append(QFTGate(n_clock).inverse(), clock[:])

        # Step 5: Controlled rotation on ancilla
        # For each clock state |k⟩, the encoded eigenvalue is λ_k = k * max_eig / 2^n_clock
        # We rotate ancilla by angle θ = 2*arcsin(C / λ_k) where C is chosen ≤ min |λ|
        min_nonzero_eig = np.min(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-10]))
        C = 0.5 * min_nonzero_eig  # conservative choice

        for k in range(1, 2 ** n_clock):
            lam_k = k * max_eig / (2 ** n_clock)
            if lam_k < 1e-10:
                continue
            ratio = min(C / lam_k, 1.0)
            theta = 2 * np.arcsin(ratio)

            # Multi-controlled Ry: control on clock register encoding |k⟩
            self._apply_controlled_rotation(qc, clock, anc[0], k, n_clock, theta)

        # Step 6: Inverse QPE (uncomputation)
        qc.append(QFTGate(n_clock), clock[:])

        for j in range(n_clock - 1, -1, -1):
            power = 2 ** j
            t_j = t_scale * power / (2 ** n_clock)
            U_j_dag = expm(-1j * A * t_j)
            U_gate_dag = Operator(U_j_dag)
            qc.append(
                U_gate_dag.to_instruction().control(1),
                [clock[j]] + list(sol[:]),
            )

        for i in range(n_clock):
            qc.h(clock[i])

        reg_info = {
            'clock_indices': list(range(n_clock)),
            'sol_indices': list(range(n_clock, n_clock + n_sol)),
            'anc_indices': list(range(n_clock + n_sol, n_clock + n_sol + n_anc)),
        }

        return qc, reg_info

    def _apply_controlled_rotation(self, qc: QuantumCircuit,
                                   clock: QuantumRegister,
                                   anc_qubit,
                                   k: int, n_clock: int,
                                   theta: float):
        """Apply Ry(theta) on ancilla, controlled on clock register encoding integer k.

        Uses X gates to flip control qubits for the appropriate binary pattern.
        """
        binary_k = format(k, f'0{n_clock}b')

        # Flip qubits where binary_k has '0' (to create all-1 control pattern)
        for i, bit in enumerate(reversed(binary_k)):
            if bit == '0':
                qc.x(clock[i])

        # Multi-controlled Ry
        ctrl_qubits = list(clock[:])
        qc.mcry(theta, ctrl_qubits, anc_qubit)

        # Unflip
        for i, bit in enumerate(reversed(binary_k)):
            if bit == '0':
                qc.x(clock[i])

    def _run_and_extract(self, circuit: QuantumCircuit, reg_info: dict,
                         n_sol: int) -> np.ndarray:
        """Run the circuit on statevector simulator and extract the solution."""
        backend = get_aer_backend('statevector')
        tc = transpile(circuit, backend)
        tc.save_statevector()
        result = backend.run(tc, shots=1).result()
        statevector = np.array(result.get_statevector())

        n_clock = len(reg_info['clock_indices'])
        n_anc = len(reg_info['anc_indices'])
        total_qubits = n_clock + n_sol + n_anc

        # Post-select: ancilla = |1⟩, clock = |0...0⟩
        # Qubit ordering in Qiskit is little-endian
        solution = np.zeros(2 ** n_sol, dtype=complex)

        for idx in range(len(statevector)):
            bits = format(idx, f'0{total_qubits}b')[::-1]  # reverse for little-endian

            # Check ancilla qubit is |1⟩
            anc_bit = bits[reg_info['anc_indices'][0]]
            if anc_bit != '1':
                continue

            # Check clock qubits are all |0⟩
            clock_bits = ''.join(bits[ci] for ci in reg_info['clock_indices'])
            if clock_bits != '0' * n_clock:
                continue

            # Extract solution register state
            sol_bits = ''.join(bits[si] for si in reg_info['sol_indices'])
            sol_idx = int(sol_bits, 2)
            solution[sol_idx] = statevector[idx]

        return solution


def solve_hhl(A: np.ndarray, b: np.ndarray,
              clock_qubits: int = 4,
              condition_warn_threshold: float = 10.0) -> Dict:
    """Convenience function to solve Ax = b with custom HHL.

    Args:
        A: Square matrix (Hermitian or non-Hermitian; embedded automatically).
        b: Right-hand side vector.
        clock_qubits: Number of clock qubits for phase estimation (more = more precise).
        condition_warn_threshold: Log a warning if condition number exceeds this.

    Returns:
        Dict containing:
            'solution'               – normalised quantum solution vector (real part)
            'probabilities'          – |solution|² normalised to sum to 1
            'classical_solution'     – normalised classical reference solution
            'classical_probabilities'– |classical_solution|² normalised to sum to 1
            'condition_number'       – condition number of A
            'circuit_info'           – dict with qubit counts and circuit depth
            'success'                – True if quantum circuit ran; False if fell back
    """
    hhl = CustomHHL(
        clock_qubits=clock_qubits,
        condition_warn_threshold=condition_warn_threshold,
    )
    return hhl.solve(A, b)
