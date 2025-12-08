"""Lagrange element on a triangle."""

import numpy as np
import numpy.typing as npt
from simplefem.polynomials import tabulate


class LagrangeElementTriangle:
    """A Lagrange element on a triangle."""

    def __init__(self, degree):
        """Initialise."""
        self.degree = degree
        self.evaluation_points = np.array(
            [
                [(2 * j + i) / degree - 1, i / degree]
                for i in range(degree + 1)
                for j in range(degree + 1 - i)
            ]
        )

        self.coeffs = np.linalg.inv(tabulate(self.evaluation_points, degree))

    def evaluate(self, basis_function_index: int, point: npt.NDArray[np.float64]) -> np.float64:
        """Evaluate a basis function at a point.

        Args:
            basis_function_index: The index of the basis function to evaluate.
            point: The points to evaluate at.

        Returns:
            The value of the basis function at the point
        """
        return np.dot(self.coeffs[basis_function_index], tabulate(point.reshape(1, 2), self.degree))


def lagrange_element(degree: int) -> LagrangeElementTriangle:
    """Create a Lagrange element on a triangle.

    Args:
        degree: The polynomial degree

    Returns:
        A Lagrange element on a triangle
    """
    assert degree > 0
    return LagrangeElementTriangle(degree)
