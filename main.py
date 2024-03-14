from consts import (
    x,
    m_n,
    m_p,
    t_n,
    t_p,
    n_1,
    p_1,
    impurity_func,
    phi_n,
    phi_p,
    psi,
    phi_t,
    l_g,
    n_i,
)
from util import solve_ddp, plot
import numpy as np

psi = np.array(psi)
phi_n = np.array(phi_n)
phi_p = np.array(phi_p)
impurity_func = np.array(impurity_func)
m_n = np.array(m_n)
m_p = np.array(m_p)
t_n = np.float_(t_n)
t_p = np.float_(t_p)
n_1 = np.float_(n_1)
p_1 = np.float_(p_1)

new_psi, new_phi_n, new_phi_p = solve_ddp(
    1e-5, x, psi, phi_n, phi_p, impurity_func, m_n, m_p
)

plot(l_g * x, n_i * new_phi_n * np.exp(new_psi), "n", "res/n_e.png")
plot(l_g * x, n_i * new_phi_p * np.exp(-new_psi), "p", "res/p_e.png")
plot(l_g * x, -phi_t * np.log(new_phi_n), "phi_n", "res/phi_n_e.png")
plot(l_g * x, phi_t * np.log(new_phi_p), "phi_p", "res/phi_p_e.png")
plot(l_g * x, phi_t * new_psi, "psi", "res/psi_e.png")
