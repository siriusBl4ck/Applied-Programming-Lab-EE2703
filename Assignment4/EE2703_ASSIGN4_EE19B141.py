# ASSIGN 4: Fourier Approximations
# Saurav Sachin Kale, EE19B141

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate

# class for calculating fourier coefficients
class Fourier:
    # class attributes
    f = None
    u = None
    v = None
    numCoeffs = 51
    # holds the fourier coeffs
    cosineCoeffs = []
    sineCoeffs = []
    Fseries = None

    # constructor
    def __init__(self, _f, _numCoeffs = 51):
        self.f = _f
        self.numCoeffs = _numCoeffs
        # define u(x,k) and v(x, k)
        self.u = lambda x, k : _f(x)*np.cos(x*k)
        self.v = lambda x, k : _f(x)*np.sin(x*k)
        self.cosineCoeffs.clear()
        self.sineCoeffs.clear()

    # calculates cosine coefficients (a_i)
    def calcUk(self):
        print("Cosine")
        # a0 to a25
        for k in range((self.numCoeffs + 1) // 2):
            fourierCoeff = (1/np.pi) * float(integrate.quad(self.u, 0, 2*np.pi, args=(k))[0])
            #print(fourierCoeff)
            self.cosineCoeffs.append(fourierCoeff)
        print(self.cosineCoeffs)

    # calculates sine coefficients (b_i)
    def calcVk(self):
        print("Sine")
        # b1 to b25
        for k in range(1, (self.numCoeffs + 1) // 2):
            fourierCoeff = (1/np.pi) * float(integrate.quad(self.v, 0, 2*np.pi, args=(k))[0])
            #print(fourierCoeff)
            self.sineCoeffs.append(fourierCoeff)
        print(self.sineCoeffs)
    
    # driver function to calculate all the fourier coeffs
    def calcCTFS(self):
        self.calcUk()
        self.calcVk()
    
    # return the coeffs separately
    def getCoeffsSeparate(self):
        return self.cosineCoeffs.copy(), self.sineCoeffs.copy()
    
    # put them together in the form a0, a1, b1, a2, b2, so on and return
    def getCoeffsTogether(self):
        result = [self.cosineCoeffs[0]/2]
        for i in range(len(self.sineCoeffs)):
            result.append(self.cosineCoeffs[i + 1])
            result.append(self.sineCoeffs[i])
        return result
    
    def getNum(self):
        return self.numCoeffs

# define cos(cos(x)) and exp(x) which 
# takes a scalar or a vector and compute the value of the function at x and returns a scalar or vector
def coscos(x):
    return np.cos(np.cos(x))

def Exp(x):
    return np.exp(x)

# plots the original function cos(cos(x)) and expected fourier plot
def plotCoscos():
    #interval from -2pi to 4pi
    x = np.linspace(-2*np.pi, 4*np.pi, 10000)
    f = coscos(x)
    f_periodic = coscos(x % (2*np.pi))
    plt.figure(1)
    plt.plot(x, f, label="cos(cos(x))")
    plt.title("Q1. cos(cos(x)) vs. x and the expected synthesis of fourier series")
    plt.xlabel("$x\\rightarrow$")
    plt.ylabel("$cos(cos(x))\\rightarrow$")
    plt.grid(True)
    plt.plot(x, f_periodic, label="expected generated plot by fourier series")
    plt.legend()
    plt.show()

# plots the original function cos(cos(x)) and expected fourier plot
def plotExp():
    #interval from -2pi to 4pi
    x = np.linspace(-2*np.pi, 4*np.pi, 10000)
    f = Exp(x)
    f_periodic = Exp(x % (2*np.pi))
    plt.figure(2)
    plt.semilogy(x, f, label="$e^x$")
    plt.grid(True)
    plt.title("Q1. $e^x$ vs. x and the expected synthesis of fourier series")
    plt.xlabel("$x\\rightarrow$")
    plt.ylabel("$e^x\\rightarrow$")
    plt.semilogy(x, f_periodic, label="expected generated plot by fourier series")
    plt.legend()
    plt.show()

# calculate the fourier coeffs using direct integration for exp(x) and plot their absolute values in semilogy and loglog scale
def plotExpFourier():
    #find fourier coeffs
    f = lambda x : Exp(x)
    #initialize fourier object
    ctfs = Fourier(f, 51)

    #calc coeffs
    ctfs.calcCTFS()

    # in form a0, a1, b1, a2, b2, so on
    mixedFourier = [abs(i) for i in ctfs.getCoeffsTogether()]

    #plot semilogy
    plt.figure(3)
    plt.stem(range(len(mixedFourier)), mixedFourier, markerfmt='ro', linefmt='r')
    plt.title("Q3. Magnitude of fourier coefficients of $e^x$ (semilogy)")
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Magnitude of fourier coefficients$\\rightarrow$")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    #plot loglog
    plt.figure(4)
    plt.stem(range(len(mixedFourier)), mixedFourier, markerfmt='ro', linefmt='r')
    plt.title("Q3. Magnitude of fourier coefficients of $e^x$ (loglog)")
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Magnitude of fourier coefficients$\\rightarrow$")
    plt.yscale('log')
    plt.grid(True)
    plt.xscale('log')
    plt.show()

    return ctfs.getCoeffsTogether()

# calculate the fourier coeffs using direct integration for cos(cos(x)) and plot their absolute values in semilogy and loglog scale
def plotCosCosFourier():
    #find fourier coeffs
    g = lambda x : coscos(x)
    #initialize fourier object
    ctfs2 = Fourier(g, 51)

    #calc coeffs
    ctfs2.calcCTFS()

    # in form a0, a1, b1, a2, b2, so on
    mixedFourier = [abs(i) for i in ctfs2.getCoeffsTogether()]

    #plot in semilogy
    plt.figure(5)
    plt.stem(range(len(mixedFourier)), mixedFourier, markerfmt='ro', linefmt='r')
    plt.title("Q3. Magnitude of fourier coefficients of $cos(cos(x))$ (semilogy)")
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Magnitude of fourier coefficients$\\rightarrow$")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    #plot in loglog
    plt.figure(6)
    plt.stem(range(len(mixedFourier)), mixedFourier, markerfmt='ro', linefmt='r')
    plt.title("Q3. Magnitude of fourier coefficients of $cos(cos(x))$ (loglog)")
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Magnitude of fourier coefficients$\\rightarrow$")
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    
    return ctfs2.getCoeffsTogether()

# finds c in Ac = b using method of least squares
def solveForC(f):
    x = np.linspace(0, 2*np.pi, 401)
    x = x[:-1]
    b = f(x)
    A = np.zeros((400, 51))

    #generate the A matrix
    A[:, 0] = 1

    for k in range(1, 26):
        A[:, 2*k - 1] = np.cos(k*x)
        A[:, 2*k] = np.sin(k*x)
    
    # get C
    c1 = scipy.linalg.lstsq(A, b)[0]
    return [A, c1]

# calculate the fourier coeffs using method of least squares for cos(cos(x)) and plot their absolute values in semilogy and loglog scale
def plotCosCosLstSqFourier():
    g = lambda x : coscos(x)
    ctfs2 = Fourier(g, 51)

    ctfs2.calcCTFS()

    mixedFourier = [abs(i) for i in ctfs2.getCoeffsTogether()]

    # get C
    solverResult = solveForC(coscos)
    C = [abs(i) for i in solverResult[1]]
    A = solverResult[0]
    print(C)

    #plot semilogy
    plt.figure(7)
    plt.stem(range(len(mixedFourier)), mixedFourier, markerfmt='r.', linefmt='r', label="Coefficients obtained using direct integration")
    plt.title("Q5. Magnitude of fourier coefficients of $cos(cos(x))$ using least squares method (semilogy)")
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Magnitude of fourier coefficients$\\rightarrow$")
    plt.scatter(range(len(C)), C, c='green', marker='o', label="Coefficients obtained using least squared method")
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    #plot loglog
    plt.figure(8)
    plt.stem(range(len(mixedFourier)), mixedFourier, markerfmt='r.', linefmt='r', label="Coefficients obtained using direct integration")
    plt.title("Q5. Magnitude of fourier coefficients of $cos(cos(x))$ using least squares method (loglog)")
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Magnitude of fourier coefficients$\\rightarrow$")
    plt.scatter(range(len(mixedFourier)), C, c='green', marker='o', label="Coefficients obtained using least squared method")
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    return A,solverResult[1]

def plotExpLstSqFourier():
    f = lambda x : Exp(x)
    ctfs = Fourier(f, 51)

    ctfs.calcCTFS()

    mixedFourier = [abs(i) for i in ctfs.getCoeffsTogether()]

    #get C
    solverResult = solveForC(Exp)
    C = [abs(i) for i in solverResult[1]]
    A = solverResult[0]

    #plot semilogy
    plt.figure(9)
    plt.stem(range(len(mixedFourier)), mixedFourier, markerfmt='r.', linefmt='r', label="Coefficients obtained using direct integration")
    plt.title("Q5. Magnitude of fourier coefficients of $e^x$ using least squares method (semilogy)")
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Magnitude of fourier coefficients$\\rightarrow$")
    plt.scatter(range(len(mixedFourier)), C, c='green', marker='o', label="Coefficients obtained using least squared method")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    #plot loglog
    plt.figure(10)
    plt.stem(range(len(mixedFourier)), mixedFourier, markerfmt='r.', linefmt='r', label="Coefficients obtained using direct integration")
    plt.title("Q5. Magnitude of fourier coefficients of $e^x$ using least squares method (loglog)")
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Magnitude of fourier coefficients$\\rightarrow$")
    plt.scatter(range(len(mixedFourier)), C, c='green', marker='o', label="Coefficients obtained using least squared method")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.show()

    return A, solverResult[1]

#find the absolute error between direct integration and least squares calculations for FS coeffs
def findError(f, c, funcName):
    #get deviation
    deviation = [abs(i) for i in (f - c)]
    #plot deviation
    plt.figure(11)
    plt.stem(range(len(deviation)), deviation, markerfmt='r.', linefmt='r')

    #find max
    m = 0
    for i in range(len(deviation)):
        if deviation[m] < deviation[i]:
            m = i
    
    #mark the maxmimum
    plt.plot(m, deviation[m], 'bo')
    plt.title("Deviation in coefficients calculated using least squares and direct integration for " + funcName)
    plt.xlabel("$n\\rightarrow$")
    plt.ylabel("Deviation in coefficients$\\rightarrow$")
    plt.grid(True)
    plt.annotate("Max deviation = " + str(deviation[m]), (m, deviation[m]))
    plt.show()

#cos(cos(x)) superimposed with its fourier series
def plotCoscosWithAc(c):
    #plot original plots
    x = np.linspace(-2*np.pi, 4*np.pi, 10000)
    f = coscos(x)
    f_periodic = coscos(x % (2*np.pi))
    plt.figure(12)
    plt.plot(x, f, label="$cos(cos(x))$")
    plt.grid(True)
    plt.plot(x, f_periodic, label="expected generated plot by fourier series")

    plt.title("Q7. $cos(cos(x))$ vs. x, the expected and real synthesis of fourier series")
    plt.xlabel("$x\\rightarrow$")
    plt.ylabel("$e^x\\rightarrow$")
    
    A = np.zeros((10000, 51))
    A[:, 0] = 1

    for k in range(1, 26):
        A[:, 2*k - 1] = np.cos(k*x)
        A[:, 2*k] = np.sin(k*x)

    # calculate A*c
    y = np.dot(A, c)
    #plot it
    plt.scatter(x, y, c='green', label="generated fourier series from coefficients obtained from least squares method")
    plt.yscale('log')
    plt.legend()
    plt.show()

#exp(x) superimposed with its fourier series
def plotExpWithAc(c):
    #plot original plots
    x = np.linspace(-2*np.pi, 4*np.pi, 10000)
    f = Exp(x)
    f_periodic = Exp(x % (2*np.pi))
    plt.figure(13)
    plt.semilogy(x, f, label="$e^x$")
    plt.grid(True)
    plt.semilogy(x, f_periodic, label="expected generated plot by fourier series")

    plt.title("Q7. $e^x$ vs. x, the expected and real synthesis of fourier series")
    plt.xlabel("$x\\rightarrow$")
    plt.ylabel("$e^x\\rightarrow$")

    A = np.zeros((10000, 51))
    A[:, 0] = 1

    for k in range(1, 26):
        A[:, 2*k - 1] = np.cos(k*x)
        A[:, 2*k] = np.sin(k*x)

    #calculate A*c
    y = np.dot(A, c)
    #plot it
    plt.scatter(x, y, c='green', label="generated fourier series from coefficients obtained from least squares method")
    plt.yscale('log')
    plt.legend()

    plt.show()

#Q1
plotExp()
plotCoscos()
#Q3
fourier_exp = np.array(plotExpFourier())
fourier_coscos = np.array(plotCosCosFourier())
#Q5
A_exp, c_exp = plotExpLstSqFourier()
A_coscos, c_coscos = plotCosCosLstSqFourier()
#Q6
findError(fourier_exp, np.array(c_exp), "$e^x$")
findError(fourier_coscos, np.array(c_coscos), "$cos(cos(x))$")
#Q7
plotExpWithAc(np.array(c_exp))
plotCoscosWithAc(np.array(c_coscos))