import numpy as np
from simplefem import lagrange_element

N = 4


def test_lagrange1_element():
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])

    e = lagrange_element(1)
    values = e.tabulate(points)

    x = points[:, 0]
    y = points[:, 1]

    assert np.allclose(values[0], 1 - x - y)
    assert np.allclose(values[1], x)
    assert np.allclose(values[2], y)


def test_lagrange2_element():
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])

    e = lagrange_element(2)
    values = e.tabulate(points)

    x = points[:, 0]
    y = points[:, 1]

    assert np.allclose(values[0], (1 - x - y) * (1 - 2 * x - 2 * y))
    assert np.allclose(values[1], 4 * x * (1 - x - y))
    assert np.allclose(values[2], x * (2 * x - 1))
    assert np.allclose(values[3], 4 * y * (1 - x - y))
    assert np.allclose(values[4], 4 * x * y)
    assert np.allclose(values[5], y * (2 * y - 1))
