# EE2703 Assignment 7
# by Saurav Kale, EE19B141

import sympy as sy
import pylab as p
import scipy.signal as sp

s = sy.symbols('s')

# lowpass filter circuit
def lowpass(R1, R2, C1, C2, G, Vi):
    A = sy.Matrix([[0, 0, 1, -1/G], [-1/(1+R2*C2*s), 1, 0, 0], [0, -G, G, 1], [(1/R1 + 1/R2 + C1*s), -1/R2, 0, -C1*s]])
    b = sy.Matrix([0, 0, 0, Vi/R1])

    V = A.inv() * b
    
    return A, b, V

# highpass filter circuit
def highpass(R1, R3, C1, C2, G, Vi):
    A = sy.Matrix([[0, 0, 1, -1/G], [-R3*C2*s/(1+R3*C2*s), 1, 0, 0], [0, -G, G, 1], [(C1*s + C2*s + 1/R1), -C2*s, 0, -1/R1]])
    b = sy.Matrix([0, 0, 0, Vi*s*C1])

    V = A.inv() * b

    return A, b, V

# converts a sympy transfer function to a scipy.signal.lti object
def convertSympyToLTI(H):
    # fraction function: Returns a pair with expression’s numerator and denominator.
    n, d = sy.fraction(H)
    # convert those to polynomials
    polynum = sy.poly(n)
    polyden = sy.poly(d)
    # get arrays for their coefficients
    numCoeff = polynum.all_coeffs()
    denCoeff = polyden.all_coeffs()

    # feed the coefficient arrays into sp.lti to get an lti system object with the transfer funciton H
    H_lti = sp.lti(p.array(numCoeff, dtype=float), p.array(denCoeff, dtype=float))

    return H_lti

def lowpassAnalysis():
    # first of all show the magnitude plot
    A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
    print('G=1000')
    Vo = V[3]
    print(Vo)
    w = p.logspace(0,8,801)
    ss = complex(0, 1)*w
    hf = sy.lambdify(s, Vo, 'numpy')
    v = hf(ss)

    p.loglog(w,abs(v),lw=2)
    p.title("Magnitude plot of lowpass filter")
    p.ylabel("$|H(j\\omega)|\\rightarrow$")
    p.xlabel("$\\omega$(in rad/s)$\\rightarrow$")
    p.grid(True)
    p.show()

    # Q1
    # now we obtain the step response
    t = p.linspace(0, 0.001, 1000)
    StepResponse_SDomain = convertSympyToLTI(Vo * 1/s)
    t, StepResponse = sp.impulse(StepResponse_SDomain, None, t)

    p.plot(t, StepResponse)
    p.title("Step response of lowpass filter")
    p.ylabel("$v_{o}(t)$(in V)$\\rightarrow$")
    p.xlabel("$t$(in s)$\\rightarrow$")
    p.grid(True)
    p.show()

    # Q2
    # response for sum of sinusoids
    t = p.linspace(0,0.01,100000)
    Vinp = p.sin(2e3*p.pi*t)+p.cos(2e6*p.pi*t)
    t, Vout, svec = sp.lsim(convertSympyToLTI(Vo), Vinp, t)

    p.plot(t, Vinp, label='$V_{in}$')
    p.plot(t, Vout, label='$V_{out}$')
    p.title("Response for superposition of sinusoids of frequency $10^3s^{-1}$  and $10^6s^{-1}$")
    p.ylabel("$v(t)$(in V)$\\rightarrow$")
    p.xlabel("$t$(in s)$\\rightarrow$")
    p.grid(True)
    p.legend()
    p.show()

def highpassAnalysis():
    # first of all show the magnitude plot
    A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,1)
    Vo = V[3]
    print(Vo)
    w = p.logspace(0,8,801)
    ss = complex(0, 1)*w
    hf = sy.lambdify(s, Vo, 'numpy')
    v = hf(ss)

    p.loglog(w,abs(v),lw=2)
    p.title("Magnitude plot of highpass filter")
    p.ylabel("$|H(j\\omega)|\\rightarrow$")
    p.xlabel("$\\omega$(in rad/s)$\\rightarrow$")
    p.grid(True)
    p.show()
    
    # Q4
    # output for a decaying sinusoid with
    # freq = 1
    # decay factor = 0.5
    t = p.linspace(0, 5, 1000)
    Vinp = p.exp(-0.5*t) * p.sin(2*p.pi*t)
    t, Vout, svec = sp.lsim(convertSympyToLTI(Vo), Vinp, t)

    p.plot(t, Vinp, label='$V_{in}$')
    p.plot(t, Vout, label='$V_{out}$')
    p.title("Response for decaying sinusoid of frequency $10^3$")
    p.ylabel("$v(t)$(in V)$\\rightarrow$")
    p.xlabel("$t$(in s)$\\rightarrow$")
    p.grid(True)
    p.legend()
    p.show()

    # output for a decaying sinusoid with
    # freq = 1e6
    # decay factor = 0.5
    t = p.linspace(0, 0.0001, 10000)
    Vinp = p.exp(-0.5*t) * p.sin(2e6*p.pi*t)
    t, Vout, svec = sp.lsim(convertSympyToLTI(Vo), Vinp, t)

    p.plot(t, Vinp, label='$V_{in}$')
    p.plot(t, Vout, label='$V_{out}$')
    p.title("Response for decaying sinusoid of frequency $10^6$")
    p.ylabel("$v(t)$(in V)$\\rightarrow$")
    p.xlabel("$t$(in s)$\\rightarrow$")
    p.grid(True)
    p.legend()
    p.show()

    #Q5
    # now we obtain the step response
    t = p.linspace(0, 0.001, 1000)
    StepResponse_SDomain = convertSympyToLTI(Vo * 1/s)
    t, StepResponse = sp.impulse(StepResponse_SDomain, None, t)

    p.plot(t, StepResponse)
    p.title("Step response of lowpass filter")
    p.ylabel("$v_{o}(t)$(in V)$\\rightarrow$")
    p.xlabel("$t$(in s)$\\rightarrow$")
    p.grid(True)
    p.show()

# Q1, Q2
lowpassAnalysis()
# Q3, Q4, Q5
highpassAnalysis()