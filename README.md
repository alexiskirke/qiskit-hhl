# qiskit-hhl

A custom implementation of the **HHL (Harrow-Hassidim-Lloyd)** quantum linear solver, built from scratch using Qiskit circuit primitives and the Aer statevector simulator.

> Solves **Ax = b** for small systems using quantum phase estimation, controlled rotations, and post-selection — without relying on any built-in Qiskit HHL algorithm.

---

## How it works

The circuit follows the standard HHL pipeline:

1. **State preparation** — encode `|b⟩` on the solution register
2. **Hermitian embedding** — if `A` is not Hermitian, embed it as `[[0, A], [A†, 0]]`
3. **Quantum Phase Estimation (QPE)** — estimate eigenvalues of `e^{iAt}`
4. **Controlled rotation** — rotate an ancilla qubit by `2·arcsin(C/λ)` for each eigenvalue `λ`
5. **Inverse QPE** — uncompute the clock register
6. **Post-selection** — keep only states where ancilla = `|1⟩`, clock = `|0…0⟩`
7. **Solution extraction** — read off the solution vector from the statevector

---

## Installation

```bash
git clone https://github.com/alexiskirke/qiskit-hhl.git
cd qiskit-hhl
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

```python
import numpy as np
from hhl import solve_hhl

A = np.array([[3.0, 1.0],
              [1.0, 3.0]])
b = np.array([1.0, 0.0])

result = solve_hhl(A, b, clock_qubits=4)

print(result['probabilities'])           # quantum solution as probability distribution
print(result['classical_probabilities']) # classical reference
print(result['condition_number'])
print(result['success'])                 # False if circuit failed and classical was used
```

The `CustomHHL` class is also available for repeated use:

```python
from hhl import CustomHHL

solver = CustomHHL(clock_qubits=4)
result = solver.solve(A, b)
```

### Return value

| Key | Description |
|-----|-------------|
| `solution` | Normalised quantum solution vector (real part) |
| `probabilities` | `\|solution\|²` normalised to sum to 1 |
| `classical_solution` | Normalised classical reference solution |
| `classical_probabilities` | `\|classical_solution\|²` normalised to sum to 1 |
| `condition_number` | Condition number of `A` |
| `circuit_info` | Dict with `n_qubits_total`, `n_qubits_solution`, `n_qubits_clock`, `depth` |
| `success` | `True` if quantum circuit ran; `False` if fell back to classical |

---

## Demo

```bash
python demo.py
```

Runs three worked examples (2×2 diagonal, 2×2 dense Hermitian, 4×4 random positive-definite) and prints a quantum vs classical comparison for each.

---

## Tests

```bash
python -m pytest tests/ -v
```

16 tests covering utility functions and the full solver.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clock_qubits` | `4` | QPE clock register size — more qubits = finer eigenvalue resolution |
| `condition_warn_threshold` | `10.0` | Log a warning if `κ(A)` exceeds this value |

**Practical limits:** the circuit complexity grows exponentially with system size. This implementation is designed for small systems (`n ≤ 8`) on a statevector simulator.

---

## Requirements

- `qiskit >= 1.0`
- `qiskit-aer >= 0.13`
- `numpy >= 1.24`
- `scipy >= 1.10`
- `pytest >= 7.0` (tests only)
