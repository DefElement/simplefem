import numpy as np
from simplefem.polynomials import tabulate
import pytest
import quadraturerules


@pytest.mark.parametrize("degree", range(6))
def test_orthogonal(degree):
    pts, wts = quadraturerules.single_integral_quadrature(
        quadraturerules.QuadratureRule.XiaoGimbutas,
        quadraturerules.Domain.Triangle,
        max(1, 2 * degree)
    )
    pts = pts[:, 1:]
    values = tabulate(pts, degree)

    for i, v_i in enumerate(values):
        for v_j in values[:i]:
            assert np.isclose((v_i * v_j * wts).sum(), 0.0)


@pytest.mark.parametrize("degree", range(6))
def test_orthonormal(degree):
    pts, wts = quadraturerules.single_integral_quadrature(
        quadraturerules.QuadratureRule.XiaoGimbutas,
        quadraturerules.Domain.Triangle,
        max(1, 2 * degree)
    )
    pts = pts[:, 1:]
    values = tabulate(pts, degree)

    for v in values:
        assert np.isclose((v * v * wts).sum(), 1.0)


@pytest.mark.parametrize("degree", range(6))
def test_not_nan_at_vertices(degree):
    pts = np.array([[0, 0], [1, 0], [0, 1]])

    values = tabulate(pts, degree)

    for row in values:
        for v in row:
            assert not np.isnan(v)
