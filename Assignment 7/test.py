import scipy.signal as sp
import pylab as p
import sympy as sy
from sympy.abc import s,t


def sym_to_sys(sym):
    sym = sy.simplify(sym)
    n, d = sy.fraction(sym)
    n, d = sy.Poly(n, s), sy.Poly(d, s)
    num, den = n.all_coeffs(), d.all_coeffs()
    if len(num) > len(den):
        print("bs")
        exit()



    H = sp.lti(n.all_coeffs(), d.all_coeffs())

    return H

def lowpass(R1, R2, C1, C2, G):
    A = sy.Matrix([[0, 0, 1, -1/G],\
            [-1/(1 + s * R2 * C2), 1, 0, 0],\
            [0, -G, G, 1],\
            [-1/R1 - 1/R2 - s * C1, 1/R2, 0, s * C1]])
    b =  sy.Matrix([0, 0, 0, -1/R1])
    V = A.inv() * b
    return A, b, V

A, b, V = lowpass(10000, 10000, 1e-9, 1e-9, 1.586)
Vo = V[3]

sym_to_sys(Vo)