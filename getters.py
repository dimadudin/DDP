import numpy as np
from consts import phi_t, n_i, l_g, q_e, m_n, m_p, M


def get_M():
    return 1e4 * l_g * M


def get_xi(x):
    xi = np.zeros(len(x))
    xi = 1e4 * l_g * x
    return xi


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

    dphi_n = np.zeros(node_num)
    dphi_p = np.zeros(node_num)
    for j in range(node_num - 1):
        h = l_g * (x[j + 1] - x[j])
        dphi_n[j] = (phi_n[j + 1] - phi_n[j]) / h
        dphi_p[j] = (phi_p[j + 1] - phi_p[j]) / h
    h = l_g * (x[node_num - 1] - x[node_num - 2])
    dphi_n[node_num - 1] = (phi_n[node_num - 1] - phi_n[node_num - 2]) / h
    dphi_p[node_num - 1] = (phi_p[node_num - 1] - phi_p[node_num - 2]) / h

    j_n = np.zeros(node_num)
    j_p = np.zeros(node_num)
    j_n = q_e * m_n * phi_t * n_i * np.exp(psi) * dphi_n
    j_p = -q_e * m_p * phi_t * n_i * np.exp(-psi) * dphi_p

    J = np.zeros(node_num)
    J = j_n + j_p

    return j_n, j_p, J
