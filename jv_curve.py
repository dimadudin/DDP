import numpy as np

from consts import get_psi_bc, get_phi_n_bc, get_phi_p_bc
from plotting import plot_currents, plot_densities, plot_log_densities, plot_potential
from solve_ddp import solve_ddp


def jv_curve(tol, x, impurity_func, voltages):
    node_number = len(x)

    prev_psi = np.zeros(node_number)
    prev_psi[0], prev_psi[-1] = get_psi_bc(0.0)
    d_psi = (prev_psi[-1] - prev_psi[0]) / node_number
    for j in range(1, node_number - 1):
        prev_psi[j] = prev_psi[j - 1] + d_psi

    prev_phi_n = np.zeros(node_number)
    prev_phi_n[0], prev_phi_n[-1] = get_phi_n_bc(0.0)
    d_phi_n = (prev_phi_n[-1] - prev_phi_n[0]) / node_number
    for j in range(1, node_number - 1):
        prev_phi_n[j] = prev_phi_n[j - 1] + d_phi_n

    prev_phi_p = np.zeros(node_number)
    prev_phi_p[0], prev_phi_p[-1] = get_phi_p_bc(0.0)
    d_phi_p = (prev_phi_p[-1] - prev_phi_p[0]) / node_number
    for j in range(1, node_number - 1):
        prev_phi_p[j] = prev_phi_p[j - 1] + d_phi_p

    print("iteration 0: Va = 0 V")
    print("")
    new_psi, new_phi_n, new_phi_p = solve_ddp(
        tol, x, prev_psi, prev_phi_n, prev_phi_p, impurity_func
    )

    plot_densities(x, new_psi, new_phi_n, new_phi_p, "res/np0.png", rf"$V_a = {0.0} V$")
    plot_log_densities(
        x, new_psi, new_phi_n, new_phi_p, "res/log_np0.png", rf"$V_a = {0.0} V$"
    )
    plot_currents(
        x, new_psi, new_phi_n, new_phi_p, "res/j_np0.png", rf"$V_a = {0.0} V$"
    )
    plot_potential(
        x, new_psi, new_phi_n, new_phi_p, "res/pot0.png", rf"$V_a = {0.0} V$"
    )

    prev_psi, prev_phi_n, prev_phi_p = new_psi, new_phi_n, new_phi_p

    j = np.zeros(len(voltages))

    i = 1
    for v_a in voltages:
        prev_psi[0], prev_psi[-1] = get_psi_bc(v_a)
        prev_phi_n[0], prev_phi_n[-1] = get_phi_n_bc(v_a)
        prev_phi_p[0], prev_phi_p[-1] = get_phi_p_bc(v_a)

        print("")
        print(f"iteration {i}: Va = {v_a} V")
        print("")
        new_psi, new_phi_n, new_phi_p = solve_ddp(
            tol, x, prev_psi, prev_phi_n, prev_phi_p, impurity_func
        )

        plot_densities(
            x, new_psi, new_phi_n, new_phi_p, f"res/np{i}.png", rf"$V_a = {v_a} V$"
        )
        plot_log_densities(
            x, new_psi, new_phi_n, new_phi_p, f"res/log_np{i}.png", rf"$V_a = {v_a} V$"
        )
        plot_currents(
            x, new_psi, new_phi_n, new_phi_p, f"res/j_np{i}.png", rf"$V_a = {v_a} V$"
        )
        plot_potential(
            x, new_psi, new_phi_n, new_phi_p, f"res/pot{i}.png", rf"$V_a = {v_a} V$"
        )

        prev_psi, prev_phi_n, prev_phi_p = new_psi, new_phi_n, new_phi_p
        i += 1
