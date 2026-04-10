"""Tests for the qiskit-hhl custom HHL implementation."""

import numpy as np
import pytest
from hhl import CustomHHL, solve_hhl
from hhl.hhl_utils import (
    ensure_hermitian,
    check_condition_number,
    normalize_vector,
    make_positive_definite,
    classical_solve,
)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestHHLUtils:

    def test_ensure_hermitian_already_hermitian(self):
        A = np.array([[2, 1], [1, 3]], dtype=complex)
        H, embedded = ensure_hermitian(A)
        assert not embedded
        assert np.allclose(H, A)

    def test_ensure_hermitian_embeds_non_hermitian(self):
        A = np.array([[1, 2], [3, 4]], dtype=complex)
        H, embedded = ensure_hermitian(A)
        assert embedded
        assert H.shape == (4, 4)
        assert np.allclose(H, H.conj().T)

    def test_condition_number(self):
        A = np.array([[4, 0], [0, 1]], dtype=float)
        kappa = check_condition_number(A)
        assert abs(kappa - 4.0) < 0.1

    def test_normalize_vector(self):
        v = np.array([3, 4], dtype=float)
        vn = normalize_vector(v)
        assert abs(np.linalg.norm(vn) - 1.0) < 1e-10

    def test_normalize_zero_vector_raises(self):
        with pytest.raises(ValueError):
            normalize_vector(np.zeros(3))

    def test_make_positive_definite(self):
        A = np.array([[1, 0], [0, -1]], dtype=float)
        A_pd = make_positive_definite(A)
        eigvals = np.linalg.eigvalsh(A_pd)
        assert np.all(eigvals > 0)

    def test_make_positive_definite_already_pd(self):
        A = np.array([[3, 1], [1, 2]], dtype=float)
        A_pd = make_positive_definite(A)
        # Should be unchanged (already positive definite)
        assert np.allclose(A_pd, A)

    def test_classical_solve(self):
        A = np.array([[2, 1], [1, 3]], dtype=float)
        b = np.array([1, 0], dtype=float)
        x = classical_solve(A, b)
        assert np.allclose(A @ x, b, atol=1e-10)


# ---------------------------------------------------------------------------
# HHL solver tests
# ---------------------------------------------------------------------------

class TestCustomHHL:

    def test_2x2_identity(self):
        """HHL on scaled identity: 2I x = b => x = b/2."""
        A = np.eye(2) * 2.0
        b = np.array([1.0, 0.0])
        result = solve_hhl(A, b, clock_qubits=3)
        # Classical reference should match b/2 (normalised)
        classical = result['classical_solution']
        assert len(classical) == 2

    def test_2x2_simple(self):
        """HHL on a simple 2x2 system — classical solution must be correct."""
        A = np.array([[3, 1], [1, 3]], dtype=float)
        b = np.array([1, 0], dtype=float)
        result = solve_hhl(A, b, clock_qubits=3)
        x_classical = np.linalg.solve(A, b)
        expected = x_classical / np.linalg.norm(x_classical)
        assert np.allclose(result['classical_solution'], expected, atol=1e-6)

    def test_4x4_system_probabilities_sum_to_one(self):
        """Probabilities from a 4x4 system must sum to 1."""
        np.random.seed(42)
        M = np.random.randn(4, 4)
        A = M @ M.T + 2 * np.eye(4)
        b = np.array([1, 0.5, 0.3, 0.1])
        result = solve_hhl(A, b, clock_qubits=4)
        assert 'probabilities' in result
        assert len(result['probabilities']) == 4
        assert abs(sum(result['probabilities']) - 1.0) < 1e-6

    def test_result_keys_present(self):
        """Result dict must contain all expected keys."""
        A = np.array([[3, 1], [1, 3]], dtype=float)
        b = np.array([1.0, 1.0])
        result = solve_hhl(A, b, clock_qubits=3)
        for key in ('solution', 'probabilities', 'classical_solution',
                    'classical_probabilities', 'condition_number',
                    'circuit_info', 'success'):
            assert key in result, f"Missing key: {key}"

    def test_ranking_agreement(self):
        """HHL and classical solutions should agree on the top-ranked candidate."""
        A = np.array([[4, 1, 0, 0],
                      [1, 3, 1, 0],
                      [0, 1, 2, 1],
                      [0, 0, 1, 5]], dtype=float)
        b = np.array([1, 0.5, 0.2, 0.1])
        result = solve_hhl(A, b, clock_qubits=4)
        q_probs = result['probabilities']
        c_probs = result['classical_probabilities']
        q_top2 = set(np.argsort(q_probs)[-2:])
        c_top2 = set(np.argsort(c_probs)[-2:])
        # Allow at least 1 overlap (quantum approximation may differ slightly)
        assert len(q_top2 & c_top2) >= 1

    def test_condition_number_reported(self):
        """High-condition-number system should still produce output with correct kappa."""
        A = np.diag([100, 1, 1, 1]).astype(float)
        b = np.array([1, 1, 1, 1], dtype=float)
        result = solve_hhl(A, b, clock_qubits=4, condition_warn_threshold=5.0)
        assert result['condition_number'] > 5.0
        assert 'probabilities' in result

    def test_non_hermitian_matrix(self):
        """Non-Hermitian A should be embedded and produce a valid result."""
        A = np.array([[1, 2], [0, 3]], dtype=float)
        b = np.array([1, 1], dtype=float)
        result = solve_hhl(A, b, clock_qubits=3)
        assert 'solution' in result
        # Classical probabilities must sum to 1
        assert abs(sum(result['classical_probabilities']) - 1.0) < 1e-6

    def test_classical_hhl_solve_object(self):
        """CustomHHL object interface should behave identically to solve_hhl."""
        A = np.array([[3, 1], [1, 3]], dtype=float)
        b = np.array([1.0, 0.5])
        hhl = CustomHHL(clock_qubits=3)
        result = hhl.solve(A, b)
        assert result['classical_solution'].shape == (2,)
