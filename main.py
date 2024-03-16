import numpy as np
from consts import x, impurity_func
from jv_curve import jv_curve


voltages = np.linspace(0.1, 0.5, 41)
# print(voltages)
tolerance = np.float_(1e-3)
jv_curve(tolerance, x, impurity_func, voltages)
