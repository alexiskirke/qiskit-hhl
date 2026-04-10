"""qiskit-hhl: Custom HHL (Harrow-Hassidim-Lloyd) linear solver using Qiskit.

Public API::

    from hhl import solve_hhl, CustomHHL
"""

from hhl.hhl_custom import CustomHHL, solve_hhl
from hhl.hhl_utils import (
    ensure_hermitian,
    check_condition_number,
    normalize_vector,
    make_positive_definite,
    classical_solve,
    rescale_matrix_for_hhl,
)

__all__ = [
    "CustomHHL",
    "solve_hhl",
    "ensure_hermitian",
    "check_condition_number",
    "normalize_vector",
    "make_positive_definite",
    "classical_solve",
    "rescale_matrix_for_hhl",
]
