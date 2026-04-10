"""Demo: solving linear systems with the custom HHL algorithm.

Runs three examples of increasing size and prints a comparison between
the quantum (HHL) and classical solutions.

Usage::

    python demo.py
"""

import numpy as np
from hhl import solve_hhl


def print_result(label: str, result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Condition number : {result['condition_number']:.3f}")
    print(f"  Circuit success  : {result['success']}")
    print(f"  Total qubits     : {result['circuit_info']['n_qubits_total']}")
    print(f"  Circuit depth    : {result['circuit_info']['depth']}")
    print()
    np.set_printoptions(precision=4, suppress=True)
    print(f"  HHL probabilities      : {result['probabilities']}")
    print(f"  Classical probabilities: {result['classical_probabilities']}")
    print()
    top_q = int(np.argmax(result['probabilities']))
    top_c = int(np.argmax(result['classical_probabilities']))
    print(f"  Top HHL index      : {top_q}")
    print(f"  Top classical index: {top_c}")
    agree = "AGREE" if top_q == top_c else "DIFFER"
    print(f"  Top-1 ranking      : {agree}")


# ---------------------------------------------------------------------------
# Example 1: 2x2 diagonal system  (well-conditioned)
# ---------------------------------------------------------------------------
print("\nExample 1 — 2×2 diagonal system")
A1 = np.array([[2.0, 0.0],
               [0.0, 1.0]])
b1 = np.array([1.0, 1.0])
# Classical solution: x = [0.5, 1.0]  =>  x[1] dominates
result1 = solve_hhl(A1, b1, clock_qubits=3)
print_result("2×2 diagonal  (A=diag(2,1), b=[1,1])", result1)


# ---------------------------------------------------------------------------
# Example 2: 2x2 dense Hermitian system
# ---------------------------------------------------------------------------
print("\nExample 2 — 2×2 dense Hermitian system")
A2 = np.array([[3.0, 1.0],
               [1.0, 3.0]])
b2 = np.array([1.0, 0.0])
# Classical solution: x = [3/8, -1/8]  =>  x[0] dominates
result2 = solve_hhl(A2, b2, clock_qubits=4)
print_result("2×2 dense Hermitian  (A=[[3,1],[1,3]], b=[1,0])", result2)


# ---------------------------------------------------------------------------
# Example 3: 4x4 random positive-definite system
# ---------------------------------------------------------------------------
print("\nExample 3 — 4×4 random positive-definite system")
np.random.seed(7)
M = np.random.randn(4, 4)
A3 = M @ M.T + 3.0 * np.eye(4)   # guaranteed positive definite
b3 = np.array([1.0, 0.5, 0.25, 0.1])
result3 = solve_hhl(A3, b3, clock_qubits=4)
print_result("4×4 random PD system", result3)

print("\nDone.")
