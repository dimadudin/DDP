from util import solve_tridiagonal_system
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


if __name__ == "__main__":
    unittest.main()
