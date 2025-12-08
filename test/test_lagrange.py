import numpy as np
from simplefem import lagrange_element

N = 4


def test_lagrange1_element():
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])

    e = lagrange_element(1)
    values = e.tabulate(points)

    x = points[:, 0]
    y = points[:, 1]

    assert np.allclose(values[0], (1 - x - y) / 2)
    assert np.allclose(values[1], (1 + x - y) / 2)
    assert np.allclose(values[2], y)


def test_lagrange2_element():
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])

    e = lagrange_element(2)
    values = e.tabulate(points)

    x = points[:, 0]
    y = points[:, 1]

    assert np.allclose(values[0], (y + x - 1) * (y + x) / 2)
    assert np.allclose(values[1], (1 + x - y) * (1 - x - y))
    assert np.allclose(values[2], (y - x - 1) * (y - x) / 2)
    assert np.allclose(values[3], 2 * y * (1 - x - y))
    assert np.allclose(values[4], 2 * y * (1 + x - y))
    assert np.allclose(values[5], y * (2 * y - 1))


def test_vertices():
    e = lagrange_element(1)
    assert np.allclose(e.evaluation_points[0], [-1, 0])
    assert np.allclose(e.evaluation_points[1], [1, 0])
    assert np.allclose(e.evaluation_points[2], [0, 1])
