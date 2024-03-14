import matplotlib.pyplot as plt
import numpy as np
from numpy import exp


def solve_ddp(
    tol: float,
    x: np.ndarray,
    psi_0: np.ndarray,
    phi_n_0: np.ndarray,
    phi_p_0: np.ndarray,
    impurity: np.ndarray,
    m_n: np.ndarray,
    m_p: np.ndarray,
    t_n: np.float_,
    t_p: np.float_,
    n_1: np.float_,
    p_1: np.float_,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_num = len(x)

    prev_psi = psi_0.copy()
    prev_phi_n = phi_n_0.copy()
    prev_phi_p = phi_p_0.copy()

    new_psi = np.zeros(node_num)
    new_phi_n = np.zeros(node_num)
    new_phi_p = np.zeros(node_num)

    error = float("inf")
    print(error)
    i = 1
    while error > tol:
        plot(x, prev_phi_n, "phi_n", f"res/phi_n_{i}.png")
        plot(x, prev_phi_p, "phi_p", f"res/phi_p_{i}.png")
        plot(x, prev_psi, "psi", f"res/psi_{i}.png")

        new_psi, err_psi = solve_poisson(x, prev_psi, prev_phi_n, prev_phi_p, impurity)

        S = 1 / (
            t_n * (prev_phi_p * np.exp(-new_psi) + p_1)
            + t_p * (prev_phi_n * np.exp(new_psi) + n_1)
        )

        new_phi_n, err_phi_n = solve_continuity_n(
            x, prev_phi_n, new_psi, m_n, prev_phi_p, S
        )

        new_phi_p, err_phi_p = solve_continuity_p(
            x, prev_phi_p, new_psi, m_p, new_phi_n, S
        )

        error = np.max([err_psi, err_phi_n, err_phi_p])
        prev_psi, prev_phi_n, prev_phi_p = new_psi, new_phi_n, new_phi_p

        i += 1

    return new_psi, new_phi_n, new_phi_p


def solve_continuity_n(
    x: np.ndarray,
    phi_n_0: np.ndarray,
    psi: np.ndarray,
    m_n: np.ndarray,
    phi_p: np.ndarray,
    S: np.ndarray,
) -> tuple[np.ndarray, np.float_]:
    node_num = len(x)
    k = np.zeros(node_num)
    q = np.zeros(node_num)
    f = np.zeros(node_num)
    for j in range(node_num):
        k[j] = m_n[j] * exp(psi[j])
        q[j] = -(phi_p[j] * S[j])
        f[j] = -S[j]
    new_phi_n = solve_differential_equation(x, k, q, f)
    err = np.linalg.norm(new_phi_n - phi_n_0)
    return new_phi_n, err


def solve_continuity_p(
    x: np.ndarray,
    phi_p_0: np.ndarray,
    psi: np.ndarray,
    m_p: np.ndarray,
    phi_n: np.ndarray,
    S: np.ndarray,
) -> tuple[np.ndarray, np.float_]:
    node_num = len(x)
    k = np.ones(node_num)
    q = np.zeros(node_num)
    f = np.zeros(node_num)
    for j in range(node_num):
        k[j] = m_p[j] * exp(-psi[j])
        q[j] = -(phi_n[j] * S[j])
        f[j] = -S[j]
    new_phi_p = solve_differential_equation(x, k, q, f)
    err = np.linalg.norm(new_phi_p - phi_p_0)
    return new_phi_p, err


def solve_poisson(
    x: np.ndarray,
    psi_0: np.ndarray,
    phi_n: np.ndarray,
    phi_p: np.ndarray,
    impurity: np.ndarray,
) -> tuple[np.ndarray, np.float_]:
    node_num = len(x)
    k = np.ones(node_num)

    q = np.zeros(node_num)
    for j in range(node_num):
        q[j] = -(phi_n[j] * exp(psi_0[j]) + phi_p[j] * exp(-psi_0[j]))

    f = np.zeros(node_num)
    f[0] = 0.0
    f[-1] = 0.0
    for j in range(1, node_num - 1):
        half_h = 0.5 * (x[j + 1] - x[j - 1])
        f[j] = (
            phi_n[j] * exp(psi_0[j])
            - phi_p[j] * exp(-psi_0[j])
            - (1 / half_h)
            * (
                (psi_0[j + 1] - psi_0[j]) / (x[j + 1] - x[j])
                - (psi_0[j] - psi_0[j - 1]) / (x[j] - x[j - 1])
            )
            - impurity[j]
        )

    sigma = solve_differential_equation(x, k, q, f)
    for j in range(node_num):
        if abs(sigma[j]) > 3.7:
            sigma[j] = np.sign(sigma[j]) * np.log(abs(sigma[j]))
        elif abs(sigma[j]) > 1:
            sigma[j] = np.sign(sigma[j]) * (abs(sigma[j]) ** 0.2)

    new_psi = psi_0 + sigma
    err = np.linalg.norm(new_psi - psi_0)

    return new_psi, err


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
        c[j] = -(a[j] + b[j] + q[j])
        d[j] = -f[j]

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


def plot(x, y, y_label, fname):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(fname)
    plt.close()
