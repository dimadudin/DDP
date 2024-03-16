import numpy as np

from consts import impurity_func, phi_n, phi_p, psi, x
from plotting import plot_currents, plot_densities, plot_log_densities, plot_potential
from solve_ddp import solve_ddp

new_psi, new_phi_n, new_phi_p = solve_ddp(
    np.float_(1e-5), x, psi, phi_n, phi_p, impurity_func
)

plot_densities(x, new_psi, new_phi_n, new_phi_p, "res/np_e.png")
plot_log_densities(x, new_psi, new_phi_n, new_phi_p, "res/log_np_e.png")
plot_currents(x, new_psi, new_phi_n, new_phi_p, "res/j_np_e.png")
plot_potential(x, new_psi, new_phi_n, new_phi_p, "res/potential_e.png")
