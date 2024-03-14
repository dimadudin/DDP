from util import solve_tridiagonal_system, solve_differential_equation
import numpy as np
import unittest


def construct_tridiag(m):
    node_num = len(m[0])
    a = np.zeros(node_num)
    c = np.zeros(node_num)
    b = np.zeros(node_num)

    a[0] = 0
    c[0] = m[0][0]
    b[0] = m[0][1]

    for j in range(1, node_num - 1):
        a[j] = m[j][j - 1]
        c[j] = m[j][j]
        b[j] = m[j][j + 1]

    a[-1] = m[-1][-2]
    c[-1] = m[-1][-1]
    b[-1] = 0

    return a, c, b


class Tests(unittest.TestCase):
    def test_tridiag_solve1(self):
        m1 = np.array([[2, 3, 0, 0], [6, 3, 9, 0], [0, 2, 5, 2], [0, 0, 4, 3]])
        rhs1 = np.array([21.0, 69.0, 34.0, 22.0])

        a, c, b = construct_tridiag(m1)
        tmp = np.array([21.0, 69.0, 34.0, 22.0])

        y1 = solve_tridiagonal_system(a, c, b, tmp)
        cmp_rhs1 = np.matmul(m1, y1)

        for j in range(len(rhs1)):
            self.assertEqual(
                rhs1[j],
                cmp_rhs1[j],
            )

    def test_diff_solve(self):
        node_num = 11
        x = np.linspace(0.0, 1.0, node_num)

        def theor_sol(x):
            return np.sin(x) + 2.0 * np.cos(x) + 1.0

        th_sol_arr = np.array([theor_sol(x[i]) for i in range(node_num)])

        k = np.ones(node_num)
        q = np.ones(node_num)
        f = np.ones(node_num)
        f[0] = 3.0
        f[-1] = 2.922076
        comp_sol = solve_differential_equation(x, k, q, f)

        tmp = np.zeros(node_num)
        tmp[0] = f[0]
        tmp[-1] = f[-1]
        for j in range(1, node_num - 1):
            half_h = 0.5 * (x[j + 1] - x[j - 1])
            tmp[j] = (1 / half_h) * (
                (comp_sol[j + 1] - comp_sol[j]) / (x[j + 1] - x[j])
                - (comp_sol[j] - comp_sol[j - 1]) / (x[j] - x[j - 1])
            ) + q[j] * comp_sol[j]

        print(tmp)
        print(f)
        print(comp_sol)
        print(th_sol_arr)

        for j in range(node_num):
            self.assertEqual(
                comp_sol[j],
                th_sol_arr[j],
            )


if __name__ == "__main__":
    unittest.main()
