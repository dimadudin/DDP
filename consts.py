from numpy import exp, log, sqrt, linspace

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

n_1 = n_c * exp(-e_gap / (2 * k * t))
p_1 = n_v * exp(-e_gap / (2 * k * t))

O = 0.0 / l_g
M = 0.3e-4 / l_g
L = 2.0e-4 / l_g
node_number = 2001

x = linspace(O, L, node_number)

n_d = 1e16
n_a = 1e18

n = [0.0] * node_number
for j in range(node_number):
    if x[j] < (M - O):
        n[j] = 3.61e2
    else:
        n[j] = n_d / n_i

p = [0.0] * node_number
for j in range(node_number):
    if x[j] < (M - O):
        p[j] = n_a / n_i
    else:
        p[j] = 3.61e4

impurity_func = [0.0] * node_number
for j in range(node_number):
    impurity_func[j] = n[j] - p[j]

m_n = [0.0] * node_number
for j in range(node_number):
    if x[j] < (M - O):
        m_n[j] = 250 / mu_0
    else:
        m_n[j] = 1100 / mu_0

m_p = [0.0] * node_number
for j in range(node_number):
    if x[j] < (M - O):
        m_p[j] = 100 / mu_0
    else:
        m_p[j] = 450 / mu_0

voltage = 0.0 / phi_t

psi = [0.0] * node_number
psi[0] = log(impurity_func[0] / 2 + sqrt((impurity_func[0] / 2) ** 2 + 1)) + voltage
psi[-1] = log(impurity_func[-1] / 2 + sqrt((impurity_func[-1] / 2) ** 2 + 1))
d_psi = (psi[-1] - psi[0]) / node_number
for j in range(1, node_number - 1):
    psi[j] = psi[j - 1] + d_psi

phi_n = [0.0] * node_number
phi_n[0] = exp(voltage)
phi_n[-1] = 1.0
for j in range(1, node_number - 1):
    phi_n[j] = n[j] * exp(-psi[j])

phi_p = [0.0] * node_number
phi_p[0] = exp(-voltage)
phi_p[-1] = 1.0
for j in range(1, node_number - 1):
    phi_p[j] = p[j] * exp(psi[j])
