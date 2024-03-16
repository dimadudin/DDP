import matplotlib.pyplot as plt
from getters import get_xi, get_pot_phis, get_densities, get_currents, get_M
from numpy import log


def plot_jv(v, j, fname, title):
    plt.figure()
    plt.plot(v, j, label=r"$j$")
    plt.xlabel("v_a")
    plt.grid(False)
    plt.legend()
    plt.title(title)
    plt.savefig(fname)
    plt.close()


def plot_potential(x, psi, phi_n, phi_p, fname, title):
    xi = get_xi(x)
    n_psi, n_phi_n, n_phi_p = get_pot_phis(psi, phi_n, phi_p)

    plt.figure()
    plt.axvline(x=get_M(), color="black", linestyle="dashed")
    plt.plot(xi, n_psi, label=r"$\psi$")
    plt.plot(xi, n_phi_n, label=r"$\phi_n$")
    plt.plot(xi, n_phi_p, label=r"$\phi_p$")
    plt.xlabel("x")
    plt.grid(False)
    plt.legend()
    plt.title(title)
    plt.savefig(fname)
    plt.close()


def plot_densities(x, psi, phi_n, phi_p, fname, title):
    xi = get_xi(x)
    n, p = get_densities(psi, phi_n, phi_p)
    plt.figure()
    plt.axvline(x=get_M(), color="black", linestyle="dashed")
    plt.plot(xi, n, label=r"$n$")
    plt.plot(xi, p, label=r"$p$")
    plt.xlabel("x")
    plt.grid(False)
    plt.legend()
    plt.title(title)
    plt.savefig(fname)
    plt.close()


def plot_currents(x, psi, phi_n, phi_p, fname, title):
    xi = get_xi(x)
    j_n, j_p, J = get_currents(x, psi, phi_n, phi_p)
    plt.figure()
    plt.axvline(x=get_M(), color="black", linestyle="dashed")
    plt.plot(xi, j_n, label=r"$j_n$")
    plt.plot(xi, j_p, label=r"$j_p$")
    plt.plot(xi, J, label=r"$J$")
    plt.xlabel("x")
    plt.grid(False)
    plt.legend()
    plt.title(title)
    plt.savefig(fname)
    plt.close()


def plot_log_densities(x, psi, phi_n, phi_p, fname, title):
    xi = get_xi(x)
    n, p = get_densities(psi, phi_n, phi_p)
    plt.figure()
    plt.axvline(x=get_M(), color="black", linestyle="dashed")
    plt.plot(xi, log(n), label=r"$log(n)$")
    plt.plot(xi, log(p), label=r"$log(p)$")
    plt.xlabel("x")
    plt.grid(False)
    plt.legend()
    plt.title(title)
    plt.savefig(fname)
    plt.close()
