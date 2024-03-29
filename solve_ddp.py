import numpy as np
from consts import t_n, t_p, n_1, p_1, m_n, m_p
from util import solve_differential_equation


def solve_ddp(
    tol: np.float_,
    x: np.ndarray,
    psi_0: np.ndarray,
    phi_n_0: np.ndarray,
    phi_p_0: np.ndarray,
    impurity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_num = len(x)

    prev_psi = psi_0.copy()
    prev_phi_n = phi_n_0.copy()
    prev_phi_p = phi_p_0.copy()

    new_psi = np.zeros(node_num)
    new_phi_n = np.zeros(node_num)
    new_phi_p = np.zeros(node_num)

    error = float("inf")
    while error > tol:
        new_psi, err_psi = solve_poisson(x, prev_psi, prev_phi_n, prev_phi_p, impurity)

        new_phi_n, err_phi_n = solve_continuity_n(x, new_psi, prev_phi_n, prev_phi_p)

        new_phi_p, err_phi_p = solve_continuity_p(x, new_psi, new_phi_n, prev_phi_p)

        error = np.max([err_psi, err_phi_n, err_phi_p])
        prev_psi, prev_phi_n, prev_phi_p = new_psi, new_phi_n, new_phi_p

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
    x: np.ndarray,
    psi: np.ndarray,
    phi_n_0: np.ndarray,
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

    s = get_s(psi, prev_phi_n, phi_p)
    for j in range(1, node_num - 1):
        k[j] = m_n[j] * np.exp(psi[j])
        q[j] = -(phi_p[j] * s[j])
        f[j] = -s[j]
    new_phi_n = solve_differential_equation(x, k, q, f)
    err = np.linalg.norm(new_phi_n - prev_phi_n)
    print(f"n_continuity {err}")
    prev_phi_n = new_phi_n
    return new_phi_n, err


def solve_continuity_p(
    x: np.ndarray,
    psi: np.ndarray,
    phi_n: np.ndarray,
    phi_p_0: np.ndarray,
) -> tuple[np.ndarray, np.float_]:
    node_num = len(x)
    k = np.ones(node_num)
    q = np.zeros(node_num)
    f = np.zeros(node_num)
    f[0] = phi_p_0[0]
    f[-1] = phi_p_0[-1]

    prev_phi_p = np.copy(phi_p_0)
    new_phi_p = np.zeros(node_num)

    s = get_s(psi, phi_n, prev_phi_p)
    for j in range(1, node_num - 1):
        k[j] = m_p[j] * np.exp(-psi[j])
        q[j] = -(phi_n[j] * s[j])
        f[j] = -s[j]
    new_phi_p = solve_differential_equation(x, k, q, f)
    err = np.linalg.norm(new_phi_p - prev_phi_p)
    print(f"p_continuity {err}")
    return new_phi_p, err
