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


def _jacobi_polynomial(points: np.ndarray[float], n: int, a: int) -> np.ndarray[float]:
    """Tabulate Jacobi polynomial.

    Args:
        points: The points on the interval [0,1] to evaluate the polynomial at
        n: Polynomial degree
        a: The parameter a
    """
    if n == 0:
        return np.ones(points.shape[0])
    if n == 1:
        return (a + 1) + (a + 2) * (points - 1) / 2
    i, j, k = _jrc(a, n - 1)
    return (i * points + j) * _jacobi_polynomial(points, n - 1, a) - k * _jacobi_polynomial(points, n - 2, a)


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

    return np.array([
        _jacobi_polynomial(2 * points[:, 0] / (1 - points[:, 1]) - 1, p, 0)
        * (1 - points[:, 1]) ** p
        * _jacobi_polynomial(2 * points[:, 1] - 1, q, 2 * p + 1)
        * ((p + 0.5) * (p + q + 1) * 2) ** 0.5
        for p in range(degree + 1)
        for q in range(degree - p + 1)
    ])
