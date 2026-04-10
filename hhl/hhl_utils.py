"""HHL utility functions: Hermitian embedding, condition checks, state prep."""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def ensure_hermitian(A: np.ndarray) -> Tuple[np.ndarray, bool]:
    """If A is not Hermitian, embed it as [[0, A], [A†, 0]].

    Returns (H, was_embedded) where H is Hermitian.
    """
    if np.allclose(A, A.conj().T, atol=1e-10):
        return A.astype(complex), False

    n = A.shape[0]
    H = np.zeros((2 * n, 2 * n), dtype=complex)
    H[:n, n:] = A
    H[n:, :n] = A.conj().T
    return H, True


def check_condition_number(A: np.ndarray, warn_threshold: float = 10.0) -> float:
    """Compute and log the condition number of A."""
    eigenvalues = np.linalg.eigvalsh(A) if np.allclose(A, A.conj().T) else np.linalg.eigvals(A)
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        logger.warning("Matrix has no significant eigenvalues!")
        return float('inf')
    kappa = np.max(eigenvalues) / np.min(eigenvalues)
    if kappa > warn_threshold:
        logger.warning(f"Condition number κ={kappa:.2f} exceeds threshold {warn_threshold}. "
                       "HHL accuracy may degrade.")
    return kappa


def normalize_vector(b: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length for quantum state preparation."""
    norm = np.linalg.norm(b)
    if norm < 1e-15:
        raise ValueError("Cannot normalize zero vector.")
    return b / norm


def make_positive_definite(A: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """Shift eigenvalues to make matrix positive definite if needed."""
    if not np.allclose(A, A.conj().T, atol=1e-10):
        A = (A + A.conj().T) / 2

    eigvals = np.linalg.eigvalsh(A)
    min_eig = np.min(eigvals)
    if min_eig <= 0:
        needed_shift = abs(min_eig) + 0.1 + shift
        A = A + needed_shift * np.eye(A.shape[0])
        logger.info(f"Shifted eigenvalues by {needed_shift:.4f} to ensure positive definiteness.")
    return A


def compute_eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition for Hermitian A."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return eigenvalues, eigenvectors


def classical_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Classical reference solution for Ax = b."""
    return np.linalg.solve(A, b)


def rescale_matrix_for_hhl(A: np.ndarray, t: float = None) -> Tuple[np.ndarray, float]:
    """Rescale A so eigenvalues map nicely to phase estimation register.

    For HHL, we need eigenvalues λ such that λ*t/(2π) fits in [0, 1).
    Returns (A_scaled, t_param) where A_scaled * t_param has eigenvalues in [0, 2π).
    """
    eigenvalues = np.linalg.eigvalsh(A) if np.allclose(A, A.conj().T) else np.abs(np.linalg.eigvals(A))
    max_eig = np.max(np.abs(eigenvalues))

    if t is None:
        t = 2 * np.pi / max_eig if max_eig > 1e-10 else 2 * np.pi

    return A, t
