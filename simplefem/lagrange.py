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

    def tabulate(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Tabulate the values of the basis functions at a set of points.

        Args:
            points: The points to evaluate at. This should be a two-dimensional array:
                    the entry [i][j] of this should be the jth coordinate of the ith point

        Returns:
            The value of each basis function at each point. This function returns a
            two-dimensional array: the entry [i][j] of this is the value of the ith
            basis function at the jth point
        """
        return self.coeffs @ tabulate(points, self.degree)


def lagrange_element(degree: int) -> LagrangeElementTriangle:
    """Create a Lagrange element on a triangle.

    Args:
        degree: The polynomial degree

    Returns:
        A Lagrange element on a triangle
    """
    assert degree > 0
    return LagrangeElementTriangle(degree)
