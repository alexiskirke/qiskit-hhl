"""Qiskit simulator backend setup."""

import logging

logger = logging.getLogger(__name__)


def get_aer_backend(method: str = 'statevector'):
    """Return an Aer simulator backend.

    Args:
        method: 'statevector' for statevector simulation,
                'qasm' for shot-based measurement simulation.
    """
    from qiskit_aer import AerSimulator

    if method == 'statevector':
        backend = AerSimulator(method='statevector')
    else:
        backend = AerSimulator()
    return backend


def run_circuit(circuit, backend=None, shots=4096, method='statevector'):
    """Run a circuit and return the result."""
    from qiskit import transpile

    if backend is None:
        backend = get_aer_backend(method)

    tc = transpile(circuit, backend)
    if method == 'statevector':
        tc.save_statevector()
        result = backend.run(tc, shots=1).result()
    else:
        result = backend.run(tc, shots=shots).result()
    return result
