from numpy import log, exp, sqrt


t_0 = 3.49499e-8
t_n = 2 / t_0
t_p = 1 / t_0

mu_0 = 1e4
m_n = 250 / mu_0
m_p = 450 / mu_0

l_g = 3.00576e-3

O = 0.0 / l_g
M = 0.3 / l_g
L = 2.0 / l_g
node_number = 20
dx = (L - O) / node_number

x = [0.0] * node_number
for j in range(1, node_number):
    x[j] = x[j - 1] + dx

n_d = 1e16
n_a = 1e18

n_i = 1.9e10

v_t = 0.02585

n = [0.0] * node_number
for j in range(node_number):
    if x[j] < (M - O):
        n[j] = n_d / n_i
    else:
        n[j] = 0.0

p = [0.0] * node_number
for j in range(node_number):
    if x[j] < (M - O):
        p[j] = 0.0
    else:
        p[j] = n_a / n_i

impurity_func = [0.0] * node_number
for j in range(node_number):
    impurity_func[j] = n[j] - p[j]

voltage = 4 / v_t

psi = [0.0] * node_number
psi[0] = log(impurity_func[0] / 2 + sqrt((impurity_func[0] / 2) ** 2 + 1)) + voltage
psi[-1] = log(impurity_func[-1] / 2 + sqrt((impurity_func[-1] / 2) ** 2 + 1))
d_psi = (psi[-1] - psi[0]) / node_number
for j in range(1, node_number - 1):
    psi[j] = psi[j - 1] + d_psi

phi_n = [0.0] * node_number
phi_n[0] = exp(voltage)
phi_n[-1] = 1
for j in range(1, node_number - 1):
    phi_n[j] = exp(log(n[j]) - psi[j])

phi_p = [0.0] * node_number
phi_p[0] = exp(-voltage)
phi_p[-1] = 1
for j in range(1, node_number - 1):
    phi_p[j] = exp(log(p[j]) + psi[j])
