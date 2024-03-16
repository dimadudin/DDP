import numpy as np


def solve_differential_equation(
    x: np.ndarray, k: np.ndarray, q: np.ndarray, f: np.ndarray
) -> np.ndarray:
    node_num = len(x)
    a = np.zeros(node_num)
    c = np.zeros(node_num)
    b = np.zeros(node_num)
    d = np.zeros(node_num)

    a[0] = 0
    c[0] = 1
    b[0] = 0
    d[0] = f[0]

    for j in range(1, node_num - 1):
        half_h = 0.5 * (x[j + 1] - x[j - 1])
        a[j] = (0.5 * (k[j] + k[j - 1])) / (half_h * (x[j] - x[j - 1]))
        b[j] = (0.5 * (k[j + 1] + k[j])) / (half_h * (x[j + 1] - x[j]))
        c[j] = q[j] - a[j] - b[j]
        d[j] = f[j]

    a[-1] = 0
    c[-1] = 1
    b[-1] = 0
    d[-1] = f[-1]

    return solve_tridiagonal_system(a, c, b, d)


def solve_tridiagonal_system(
    a: np.ndarray, c: np.ndarray, b: np.ndarray, d: np.ndarray
) -> np.ndarray:
    n = len(c)
    b[0] /= c[0]
    d[0] /= c[0]
    for i in range(1, n):
        fac = c[i] - a[i] * b[i - 1]
        b[i] /= fac
        d[i] = (d[i] - a[i] * d[i - 1]) / fac
    y = [0] * n
    y[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        y[i] = d[i] - b[i] * y[i + 1]
    return np.array(y)
