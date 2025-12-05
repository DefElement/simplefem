"""Lagrange element on a triangle."""

import numpy as np
from simplefem.polynomials import tabulate


class LagrangeElementTriangle:
    """A Lagrange element on a triangle."""

    def __init__(self, degree):
        """Initialise."""
        self.degree = degree
        self.evaluation_points = np.array(
            [[j / degree, i / degree] for i in range(degree + 1) for j in range(degree + 1 - i)]
        )

        self.coeffs = np.linalg.inv(tabulate(self.evaluation_points, degree))

    def tabulate(self, points: np.ndarray[float]) -> np.ndarray[float]:
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
