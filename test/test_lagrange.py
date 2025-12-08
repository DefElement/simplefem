import numpy as np
from simplefem import lagrange_element

N = 4


def test_lagrange1_element():
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])

    e = lagrange_element(1)

    for p in points:
        x, y = p

        assert np.isclose(e.evaluate(0, p), (1 - x - y) / 2)
        assert np.isclose(e.evaluate(1, p), (1 + x - y) / 2)
        assert np.isclose(e.evaluate(2, p), y)


def test_lagrange2_element():
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])

    e = lagrange_element(2)

    for p in points:
        x, y = p

        assert np.isclose(e.evaluate(0, p), (y + x - 1) * (y + x) / 2)
        assert np.isclose(e.evaluate(1, p), (1 + x - y) * (1 - x - y))
        assert np.isclose(e.evaluate(2, p), (y - x - 1) * (y - x) / 2)
        assert np.isclose(e.evaluate(3, p), 2 * y * (1 - x - y))
        assert np.isclose(e.evaluate(4, p), 2 * y * (1 + x - y))
        assert np.isclose(e.evaluate(5, p), y * (2 * y - 1))


def test_vertices():
    e = lagrange_element(1)
    assert np.allclose(e.evaluation_points[0], [-1, 0])
    assert np.allclose(e.evaluation_points[1], [1, 0])
    assert np.allclose(e.evaluation_points[2], [0, 1])
