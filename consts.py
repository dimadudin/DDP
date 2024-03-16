from numpy import exp, log, sqrt, linspace, zeros

t = 300
k = 1.38e-23
q_e = 1.6e-19
phi_t = (k * t) / q_e

mu_0 = 1.0e4
d_0 = phi_t * mu_0
n_i = 1.9e10
l_g = 3.00576e-3
t_0 = (l_g**2) / d_0

t_n = 2.0e-9 / t_0
t_p = 1.0e-9 / t_0

e_gap = 1.12

n_c = 3.2e19
n_v = 1.8e19

n_1 = (n_c * exp(-e_gap / (2 * k * t))) / n_i
p_1 = (n_v * exp(-e_gap / (2 * k * t))) / n_i

O = 0.0 / l_g
M = 0.3e-4 / l_g
L = 2.0e-4 / l_g
node_number = 2001

x = linspace(O, L, node_number)

n_d = 1e16
n_a = 1e18

impurity_func = zeros(node_number)
for j in range(node_number):
    if x[j] < (M - O):
        impurity_func[j] = -n_a / n_i
    else:
        impurity_func[j] = n_d / n_i

m_n = zeros(node_number)
for j in range(node_number):
    if x[j] < (M - O):
        m_n[j] = 250 / mu_0
    else:
        m_n[j] = 1100 / mu_0

m_p = zeros(node_number)
for j in range(node_number):
    if x[j] < (M - O):
        m_p[j] = 100 / mu_0
    else:
        m_p[j] = 450 / mu_0


def get_psi_bc(v_a):
    psi_o = log(impurity_func[0] / 2 + sqrt((impurity_func[0] / 2) ** 2 + 1)) + (
        v_a / phi_t
    )
    psi_l = log(impurity_func[-1] / 2 + sqrt((impurity_func[-1] / 2) ** 2 + 1))
    return psi_o, psi_l


def get_phi_n_bc(v_a):
    phi_n_o = exp(-(v_a / phi_t))
    phi_n_l = 1.0
    return phi_n_o, phi_n_l


def get_phi_p_bc(v_a):
    phi_p_o = exp(v_a / phi_t)
    phi_p_l = 1.0
    return phi_p_o, phi_p_l
