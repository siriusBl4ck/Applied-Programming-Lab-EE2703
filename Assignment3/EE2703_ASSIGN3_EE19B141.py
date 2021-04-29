import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as sp

# exact values of A and B
A = 1.05
B = -0.105

#loads and parses the data from "fitting.dat"
def loadAndParse(filepath):
    try:
        data = np.loadtxt("fitting.dat")
    except:
        print("ERROR: failed to parse data")
        exit()
    #first column contains the x axis
    x = data[:, 0]
    #remaining columns contain y = f(t) values
    y = data[:, 1:]
    return x, y

#this function simply plots the input x and y
def plotNoise(x, y):
    plt.plot(x, y)
    sigma = np.logspace(-1, -3, 9)
    plt.title("f(t) with differing amount of noise")
    plt.grid(True)
    plt.xlabel("t", size=15)
    plt.ylabel("f(t) + noise", size=15)
    plt.legend(sigma)
    plt.show()

#calculating g(t) = A*J2(t) + B*t
def g(t, A, B):
    return A*sp.jn(2, t) + B*t

#plotting G(t) along with the noisy data
def plotG(t):
    plt.figure(0)
    plt.title('Original Plot')
    #simply x and y to get the noise
    plt.plot(x, y)
    sigma = np.logspace(-1, -3, 9)
    plt.title("Q4. data to be fitted with theory")
    plt.grid(True)
    plt.xlabel("$t\\rightarrow$", size=15)
    plt.ylabel("$f(t) + noise\\rightarrow$", size=15)
    # plotting x, g(x, A, B) gives the ideal curve
    plt.plot(x,g(x, A, B),'k', label="True Value", linewidth=3)
    labels = np.append(sigma, "True Value")
    for i in range(len(labels) - 1):
        labels[i] = "$\sigma" + "_" + str(i + 1) + "$" + "=" + str(round(float(labels[i]), 4))
    plt.legend(labels)
    plt.show()

#this plots the errorbars
def dataWithErrorBarsForFirstCol(x, y):
    y_ideal = g(x, A, B)
    #standard deviation from the data points and ideal values
    stdev = np.std(y[:, 0] - y_ideal)
    plt.figure(1)
    plt.grid(True)
    plt.plot(x, y_ideal)
    # plot the standard deviation using error bars
    plt.errorbar(x[::5], y[::5, 0], stdev, fmt='ro')
    plt.title("Q5. Data points for σ = 0.10 along with exact function")
    plt.ylabel("Errorbars",size=15)
    plt.xlabel("$t\\rightarrow$",size=15)
    labels = np.array(["f(t)", "errorbar"])
    plt.legend(labels)
    plt.show()

#construct the matrix M
def constructM(x):
    # ideal vals
    g_vals = g(x, A, B)
    g_array = np.array(g_vals)
    #M  concists of J2 and t
    M = np.c_[sp.jn(2, x), x]
    #p contains A and B
    p = np.array([A, B])
    #print(M)
    #multiplation of M and p
    mul = np.dot(M, p)
    #mul = M*p
    print("Verifying Mp - G = 0")
    print(mul - g_array)
    #shows equality of G and M*p via a plot
    plt.figure(2)
    plt.plot(x, g_array, label="g(t, A, B)", color='r', linewidth=3)
    plt.plot(x, mul, '*', label="product M*p values")
    plt.title("Q6. Verifying the equality of M*p and g(t, A, B)")
    plt.xlabel("$t\\rightarrow$",size=15)
    plt.ylabel("$g(t) or M*p\\rightarrow$",size=15)
    plt.legend()
    plt.grid(True)
    plt.show()
    #print(g_array)
    return M

# finding the error term for ith column of data points
def findE(x, y, m, i = 0, showPlot = True):
    g_vals = g(x, A, B)
    #initialize the bounds
    A0 = np.linspace(0, 2, 21)
    B0 = np.linspace(-0.2, 0, 21)
    #print(A0)
    #print(B0)
    #initialize E
    E = np.zeros((len(A0), len(B0)))
    f = y[:, i]
    #calculate E for the ith column of data points
    for i in range(len(A0)):
        for j in range(len(B0)):
            E[i, j] = ((f - g(x, A0[i], B0[j]))**2).mean(axis=0)
    xx, yy = np.meshgrid(A0, B0)
    #set contour levels
    levels = np.linspace(0.025, 0.5, 20)
    #find the least squares solution
    p = findLeastSqSoln(m, g_vals)
    
    if (showPlot):
        plt.figure
        cs = plt.contour(xx, yy, E, levels)
        plt.clabel(cs, levels)
        plt.plot(p[0][0], p[0][1], color="blue", marker="o")
        plt.title("Q8: contour plot of $\epsilon_{ij}$")
        plt.xlabel("$A\\rightarrow$",size=15)
        plt.ylabel("$B\\rightarrow$",size=15)
        plt.annotate("Exact value" + str(round(p[0][0], 3)) + "," + str(round(p[0][1], 3)), (p[0][0], p[0][1] - 0.02))
        plt.show()
    print("Results from least squares method")
    return p

#find least square solution using scipy.linalg.lstsq
def findLeastSqSoln(M, G):
    p = scipy.linalg.lstsq(M, G)
    #print(p)
    return p

#finding errors in A and B
def plotErrors(x, y, m):
    errorA = []
    errorB = []
    #populate the errors
    for i in range(9):
        AB = findLeastSqSoln(m, y[:, i])[0]
        #print(AB)
        errorA.append(abs(AB[0] - A))
        errorB.append(abs(AB[1] - B))
    #standard deviation
    sigma = np.logspace(-1, -3, 9)

    #plot error in A and error in B vs standard deviation
    plt.plot(sigma, errorA, linestyle="--", marker="o", label="A error")
    plt.plot(sigma, errorB, linestyle="--", marker="o", label="B error")
    plt.grid(True)
    plt.title("Q10. Variation of error with noise")
    plt.xlabel("$\sigma_n\\rightarrow$",size=15)
    plt.ylabel("$MS Error\\rightarrow$",size=15)
    plt.legend()
    plt.show()
    #plot error in A and error in B vs standard deviation in loglog plot
    plt.figure()
    plt.loglog(sigma, errorA, 'ro', label="A error")
    plt.loglog(sigma, errorB, 'ro', label="B error")
    plt.stem(sigma, errorA, 'ro', label="A error")
    plt.grid(True)
    plt.stem(sigma, errorB, 'ro', label="B error")
    plt.errorbar(sigma,errorA,errorA,fmt='ro',uplims = True, lolims = True, marker='o',ecolor='r',label='errorbars',markerfacecolor='red')
    plt.errorbar(sigma,errorB,errorB,fmt='ro',uplims = True, lolims = True, marker='o',ecolor='b',label='errorbars',markerfacecolor='green')
    plt.title("Q11. Variation of error with noise (loglog)")
    plt.xlabel("$\sigma_n\\rightarrow$",size=15)
    plt.ylabel("$MS Error\\rightarrow$",size=15)
    plt.legend()
    plt.show()

#load and parse file
print("ASSIGN3_EE19B141")
print("----------------")
x, y = loadAndParse("fitting.dat")
#plot G(x)
plotG(x)
#Error bars
dataWithErrorBarsForFirstCol(x, y)
#construct M
m = constructM(x)
#contour plot
print(findE(x, y, m, 0)[0])
#error in A and B plots
plotErrors(x, y, m)