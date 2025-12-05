"""Orthonormal polynomials."""
import numpy as np
from functools import cache


@cache
def _jrc(
    a: int, n: int
) -> tuple[float, float, float]:
    """Get the Jacobi recurrence relation coefficients.

    Args:
        a: The parameter a
        n: The parameter n

    Returns:
        The Jacobi coefficients
    """
    return (
        (a + 2 * n + 1) * (a + 2 * n + 2) / 2 / (n + 1) / (a + n + 1),
        a * a * (a + 2 * n + 1) / 2 / (n + 1) / (a + n + 1) / (a + 2 * n),
        n * (a + n) * (a + 2 * n + 2) / (n + 1) / (a + n + 1) / (a + 2 * n),
    )


def _index(p: int, q: int) -> int:
    """Get the flat index of a point (p, q) on a triangle."""
    return (p + q + 1) * (p + q) // 2 + q


def tabulate(points: np.ndarray[float], degree: int) -> np.ndarray[float]:
    """Tabulate orthogonal polynomials on a triangle.

    Args:
        points: points to evaluate polynomials at. This should be a two-dimensional array:
                    the entry [i][j] of this should be the jth coordinate of the ith point
        degree: polynomial degree

    Returns:
        Values of orthogonal polynomials on a triangle. This function returns a
        two-dimensional array: the entry [i][j] of this is the value of the ith
        basis function at the jth point
    """
    table = np.zeros(((degree + 1) * (degree + 2) // 2, points.shape[0]))
    table[_index(0, 0)] = 1

    x = points[:, 0]
    y = points[:, 1]

    for p in range(1, degree + 1):
        a = (2 * p - 1) / p

        table[_index(0, p)] = (2 * x + y - 1) * table[_index(0, p - 1)] * a

        if p > 1:
            table[_index(0, p)] -= (1 - y) ** 2 * table[_index(0, p - 2)] * (a - 1)

    for p in range(degree):
        table[_index(1, p)] = table[_index(0, p)] * (y * (3 + 2 * p) - 1)

        for q in range(1, degree - p):
            a1, a2, a3 = _jrc(2 * p + 1, q)
            table[_index(q + 1, p)] = table[_index(q, p)] * ((2 * y - 1) * a1 + a2)
            table[_index(q + 1, p)] -= table[_index(q-1, p)] * a3

    for p in range(degree + 1):
        for q in range(degree + 1 - p):
            table[_index(q, p)] *= ((2 * p + 1) * (p + q + 1)) ** 0.5

    return table
