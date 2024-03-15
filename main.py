import numpy as np

from consts import (
    impurity_func,
    m_n,
    m_p,
    n_1,
    p_1,
    phi_n,
    phi_p,
    psi,
    t_n,
    t_p,
    x,
)
from util import (
    plot_currents,
    plot_densities,
    plot_potential,
    solve_ddp,
    plot_log_densities,
)

t_n = np.float_(t_n)
t_p = np.float_(t_p)
n_1 = np.float_(n_1)
p_1 = np.float_(p_1)

new_psi, new_phi_n, new_phi_p = solve_ddp(
    np.float_(1e-5), x, psi, phi_n, phi_p, impurity_func, m_n, m_p
)

plot_densities(x, new_psi, new_phi_n, new_phi_p, "res/np_e.png")
plot_log_densities(x, new_psi, new_phi_n, new_phi_p, "res/log_np_e.png")
plot_currents(x, new_psi, new_phi_n, new_phi_p, "res/j_np_e.png")
plot_potential(x, new_psi, new_phi_n, new_phi_p, "res/potential_e.png")
