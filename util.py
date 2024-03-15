import matplotlib.pyplot as plt
import numpy as np
from consts import t_n, t_p, n_1, p_1, l_g, phi_t, M, n_i, q_e, m_n, m_p


def solve_ddp(
    tol: np.float_,
    x: np.ndarray,
    psi_0: np.ndarray,
    phi_n_0: np.ndarray,
    phi_p_0: np.ndarray,
    impurity: np.ndarray,
    m_n: np.ndarray,
    m_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_num = len(x)

    prev_psi = psi_0.copy()
    prev_phi_n = phi_n_0.copy()
    prev_phi_p = phi_p_0.copy()

    new_psi = np.zeros(node_num)
    new_phi_n = np.zeros(node_num)
    new_phi_p = np.zeros(node_num)

    error = float("inf")
    i = 1
    while error > tol:
        new_psi, err_psi = solve_poisson(x, prev_psi, prev_phi_n, prev_phi_p, impurity)

        new_phi_n, err_phi_n = solve_continuity_n(
            tol, x, prev_phi_n, new_psi, m_n, prev_phi_p
        )

        new_phi_p, err_phi_p = solve_continuity_p(
            tol, x, prev_phi_p, new_psi, m_p, new_phi_n
        )

        error = np.max([err_psi, err_phi_n, err_phi_p])
        prev_psi, prev_phi_n, prev_phi_p = new_psi, new_phi_n, new_phi_p

        i += 1

    return new_psi, new_phi_n, new_phi_p


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
        q[j] = -(phi_n[j] * np.exp(psi_0[j]) + phi_p[j] * np.exp(-psi_0[j]))

    f = np.zeros(node_num)
    f[0] = 0.0
    f[-1] = 0.0
    for j in range(1, node_num - 1):
        half_h = 0.5 * (x[j + 1] - x[j - 1])
        f[j] = (
            phi_n[j] * np.exp(psi_0[j])
            - phi_p[j] * np.exp(-psi_0[j])
            - (1 / half_h)
            * (
                (psi_0[j + 1] - psi_0[j]) / (x[j + 1] - x[j])
                - (psi_0[j] - psi_0[j - 1]) / (x[j] - x[j - 1])
            )
            - impurity[j]
        )

    sigma = solve_differential_equation(x, k, q, f)
    for j in range(node_num):
        if abs(sigma[j]) >= 3.7:
            sigma[j] = np.sign(sigma[j]) * np.log(abs(sigma[j]))
        elif abs(sigma[j]) > 1:
            sigma[j] = np.sign(sigma[j]) * (abs(sigma[j]) ** 0.2)

    new_psi = psi_0 + sigma
    err = np.linalg.norm(sigma)
    print(f"poisson {err}")

    return new_psi, err


def get_s(psi: np.ndarray, phi_n: np.ndarray, phi_p: np.ndarray) -> np.ndarray:
    return 1 / (t_n * (phi_p * np.exp(-psi) + p_1) + t_p * (phi_n * np.exp(psi) + n_1))


def solve_continuity_n(
    tol: np.float_,
    x: np.ndarray,
    phi_n_0: np.ndarray,
    psi: np.ndarray,
    m_n: np.ndarray,
    phi_p: np.ndarray,
) -> tuple[np.ndarray, np.float_]:
    node_num = len(x)
    k = np.zeros(node_num)
    q = np.zeros(node_num)
    f = np.zeros(node_num)
    f[0] = phi_n_0[0]
    f[-1] = phi_n_0[-1]

    prev_phi_n = np.copy(phi_n_0)
    new_phi_n = np.zeros(node_num)

    err = np.float_("inf")
    while err > tol:
        for j in range(1, node_num - 1):
            s = get_s(psi, prev_phi_n, phi_p)
            k[j] = m_n[j] * np.exp(psi[j])
            q[j] = -(phi_p[j] * s[j])
            f[j] = -s[j]
        new_phi_n = solve_differential_equation(x, k, q, f)
        err = np.linalg.norm(new_phi_n - prev_phi_n)
        print(f"n_continuity {err}")
        prev_phi_n = new_phi_n
    return new_phi_n, err


def solve_continuity_p(
    tol: np.float_,
    x: np.ndarray,
    phi_p_0: np.ndarray,
    psi: np.ndarray,
    m_p: np.ndarray,
    phi_n: np.ndarray,
) -> tuple[np.ndarray, np.float_]:
    node_num = len(x)
    k = np.ones(node_num)
    q = np.zeros(node_num)
    f = np.zeros(node_num)
    f[0] = phi_p_0[0]
    f[-1] = phi_p_0[-1]

    prev_phi_p = np.copy(phi_p_0)
    new_phi_p = np.zeros(node_num)

    err = np.float_("inf")
    while err > tol:
        for j in range(1, node_num - 1):
            s = get_s(psi, phi_n, prev_phi_p)
            k[j] = m_p[j] * np.exp(-psi[j])
            q[j] = -(phi_n[j] * s[j])
            f[j] = -s[j]
        new_phi_p = solve_differential_equation(x, k, q, f)
        err = np.linalg.norm(new_phi_p - prev_phi_p)
        print(f"p_continuity {err}")
        prev_phi_p = new_phi_p
    return new_phi_p, err


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


def get_pot_phis(psi, phi_n, phi_p):
    n_psi = phi_t * psi
    n_phi_n = -phi_t * np.log(phi_n)
    n_phi_p = phi_t * np.log(phi_p)
    return n_psi, n_phi_n, n_phi_p


def get_densities(psi, phi_n, phi_p):
    n = n_i * phi_n * np.exp(psi)
    p = n_i * phi_p * np.exp(-psi)
    return n, p


def get_currents(x, psi, phi_n, phi_p):
    node_num = len(x)

    n, p = get_densities(psi, phi_n, phi_p)
    _, n_phi_n, n_phi_p = get_pot_phis(psi, phi_n, phi_p)

    dphi_n = np.zeros(node_num)
    dphi_p = np.zeros(node_num)
    for j in range(node_num - 1):
        h = l_g * (x[j + 1] - x[j])
        dphi_n[j] = (n_phi_n[j + 1] - n_phi_n[j]) / h
        dphi_p[j] = (n_phi_p[j + 1] - n_phi_p[j]) / h
    h = l_g * (x[node_num - 1] - x[node_num - 2])
    dphi_n[node_num - 1] = (n_phi_n[node_num - 1] - n_phi_n[node_num - 2]) / h
    dphi_p[node_num - 1] = (n_phi_p[node_num - 1] - n_phi_p[node_num - 2]) / h

    j_n = np.zeros(node_num)
    j_p = np.zeros(node_num)
    j_n = -q_e * m_n * n * dphi_n
    j_p = -q_e * m_p * p * dphi_p

    J = np.zeros(node_num)
    J = j_n + j_p

    return j_n, j_p, J


def plot_potential(x, psi, phi_n, phi_p, fname):
    n_psi, n_phi_n, n_phi_p = get_pot_phis(psi, phi_n, phi_p)
    plt.figure()
    plt.axvline(x=1e4 * l_g * M, color="black", linestyle="dashed")
    plt.plot(1e4 * l_g * x, n_psi, label=r"$\psi$")
    plt.plot(1e4 * l_g * x, n_phi_n, label=r"$\phi_n$")
    plt.plot(1e4 * l_g * x, n_phi_p, label=r"$\phi_p$")
    plt.xlabel("x")
    plt.grid(False)
    plt.legend()
    plt.savefig(fname)
    plt.close()


def plot_densities(x, psi, phi_n, phi_p, fname):
    n, p = get_densities(psi, phi_n, phi_p)
    plt.figure()
    plt.axvline(x=1e4 * l_g * M, color="black", linestyle="dashed")
    plt.plot(1e4 * l_g * x, n, label=r"$n$")
    plt.plot(1e4 * l_g * x, p, label=r"$p$")
    plt.xlabel("x")
    plt.grid(False)
    plt.legend()
    plt.savefig(fname)
    plt.close()


def plot_currents(x, psi, phi_n, phi_p, fname):
    j_n, j_p, J = get_currents(x, psi, phi_n, phi_p)
    plt.figure()
    plt.axvline(x=1e4 * l_g * M, color="black", linestyle="dashed")
    plt.plot(1e4 * l_g * x, j_n, label=r"$j_n$")
    plt.plot(1e4 * l_g * x, j_p, label=r"$j_p$")
    # plt.plot(1e4 * l_g * x, J, label=r"$J$")
    plt.xlabel("x")
    plt.grid(False)
    plt.legend()
    plt.savefig(fname)
    plt.close()


def plot_log_densities(x, psi, phi_n, phi_p, fname):
    n, p = get_densities(psi, phi_n, phi_p)
    plt.figure()
    plt.axvline(x=1e4 * l_g * M, color="black", linestyle="dashed")
    plt.plot(1e4 * l_g * x, np.log(n), label=r"$log(n)$")
    plt.plot(1e4 * l_g * x, np.log(p), label=r"$log(p)$")
    plt.xlabel("x")
    plt.grid(False)
    plt.legend()
    plt.savefig(fname)
    plt.close()
