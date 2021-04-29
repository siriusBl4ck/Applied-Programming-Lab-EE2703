# EE2703: Applied Programming lab
# Assignment 5: The resistor problem
# Saurav Sachin Kale, EE19B141

# instructions for usage:
# command to run the code:
# python3 EE2703_ASSIGN5_EE19B141 Nx Ny Radius Niter
# Nx Ny Radius Niter are optional command line arguments

from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.special as sp

# set defaults
Nx = 25 # size along x
Ny = 25 # size along y
Radius = 8 # radius of central lead
Niter = 1500 # number of iterations to perform

try:
    Nx = int(argv[1])
    Ny = int(argv[2])
    Radius = int(argv[3])
    Niter = int(argv[4])
except:
    print("Few params have taken their default values")
    # do nothing.
    # an error here simply means the user wishes to leave
    # that param at default so nothing to do

print("Input params:")
print("Nx:", Nx, "\nNy:", Ny, "\nradius:", Radius, "\nNiter:", Niter)

# init x and y,
x = list(range(-(Nx//2), (Nx)//2 + 1))
y = list(range(-(Ny//2), (Ny)//2 + 1))

if Nx % 2 == 0:
    x.pop()
if Ny % 2 == 0:
    y.pop()

x = np.array(x)
y = np.array(y)

#print(x)
#print(y)

Y, X = meshgrid(y, x)

#extraction of points in the center
ii = np.where(Y*Y + X*X <= (Radius)**2)

def plotPhi():
    # init phi as zeroes of Ny rows Ny cols
    phi = np.zeros((Nx, Ny))

    # set potential here = 1.0V
    phi[ii] = 1.0
    #print("Newphi")
    #print(phi)
    # plot the initial assumed distribution of phi
    plt.title("Contour plot of initial potential distribution before iterating (in V)")
    plt.xlabel("$x\\rightarrow$")
    plt.ylabel("$y\\rightarrow$")
    levels = np.linspace(0, 1, 50)
    cs = plt.contourf(Y, X, phi, levels, cmap=plt.cm.get_cmap("hot"))
    plt.scatter([y[j] for j in ii[1]], [x[j] for j in ii[0]], c='r', label='Vo = 1V')
    plt.colorbar(cs)
    plt.legend()
    plt.grid(True)
    plt.show()

    errors = []

    # iterate Niter times, every iteration we update the elements in working area with the average of its nieghbors
    for k in range(Niter):
        oldphi = phi.copy()
        # update the working area
        phi[1:-1, 1:-1] = 0.25*(oldphi[2:, 1:-1] + oldphi[:-2, 1:-1] + oldphi[1:-1, 2:] + oldphi[1:-1, :-2])
        #boundary conditions
        #left edge
        phi[1:-1,0]=phi[1:-1,1]
        #right edge
        phi[1:-1,-1]=phi[1:-1,-2]
        #bottom edge
        phi[0, 1:-1]=0
        #top edge
        phi[-1, 1:-1]=phi[-2, 1:-1]
        # 4 corners get updated with the average of their neighboring values
        phi[0, 0] = 0.5*(phi[0, 1] + phi[1, 0])
        phi[0, -1] = 0.5*(phi[0, -2] + phi[1, -1])
        phi[-1, 0] = 0.5*(phi[-1, 1] + phi[-2, 0])
        phi[-1, -1] = 0.5*(phi[-1, -2] + phi[-2, -1])
        #restore the electrode portion
        phi[ii]=1.0
        errors.append(abs(phi-oldphi).max())

    # plot the final contour plot of potential after iterating
    plt.title("Contour plot of final potential distribution after iterating (in V)")
    plt.xlabel("$x\\rightarrow$")
    plt.ylabel("$y\\rightarrow$")
    levels = np.linspace(0, 1, 50)
    cs = plt.contourf(Y, X, phi, levels, cmap=plt.cm.get_cmap("hot"))
    plt.colorbar(cs)
    #electrode position with red dots
    plt.scatter([y[j] for j in ii[1]], [x[j] for j in ii[0]], c='r', label='Vo = 1V')
    plt.legend()
    plt.grid(True)
    plt.show()

    return phi, errors

def plotErrors(errors):
    x = np.array(range(0, Niter))
    #every 50th point after 500
    x_datapoints = np.array(range(549, Niter, 50))
    #error values corresponding 
    errorValsAfter500 = [errors[i] for i in x_datapoints]

    #semilogy plot
    plt.title("Semilogy plot of maximum absolute error vs. number of iterations")
    plt.xlabel("Number of iterations$\\rightarrow$")
    plt.ylabel("Maximum absolute error$\\rightarrow$")
    plt.semilogy(x, errors, label="Original error curve")
    plt.grid(True)

    #fitted curve using all iterations
    p2 = curveFit(x, errors)[0]
    A2 = np.exp(p2[0])
    B2 = p2[1]
    plt.plot(x, A2*np.exp(B2*x), c='orange', label="curve fit for all iterations")
    
    #max error
    iters = np.arange(100, 1501, 50)
    maxErrors = abs((-A2/B2)*np.exp(B2*(iters+0.5)))

    #fitted curve using only iterations > 500
    p1 = curveFit(x_datapoints, errorValsAfter500)[0]
    A1 = np.exp(p1[0])
    B1 = p1[1]
    plt.plot(x_datapoints, A1*np.exp(B1*x_datapoints), c='green', label="curve fit for iterations > 500")

    #plot every 50th point
    x50Sample = np.array(range(0, Niter, 50))
    plt.scatter(x50Sample, [errors[i] for i in x50Sample], c='red', label="Samples every 50 iterations")
    
    plt.legend()

    plt.show()

    #loglog plot
    plt.title("loglog plot of maximum absolute error vs. number of iterations")
    plt.xlabel("Number of iterations$\\rightarrow$")
    plt.ylabel("Maximum absolute error$\\rightarrow$")
    plt.loglog(x, errors)
    plt.scatter(x50Sample, [errors[i] for i in x50Sample], c='red', label="Samples every 50 iterations")
    plt.legend()
    plt.grid(True)
    plt.show()

    #plot the maxErrors
    plt.title("Maximum error vs. number of iterations")
    plt.xlabel("Number of iterations$\\rightarrow$")
    plt.ylabel("Maximum Error$\\rightarrow$")
    plt.loglog(iters, maxErrors)
    plt.grid(True)
    plt.show()

def curveFit(x_datapoints, Y):
    #constructing M
    M = np.c_[np.ones(len(x_datapoints)), x_datapoints]
    #print(M)
    #print(G)
    #find least squares solution
    p = scipy.linalg.lstsq(M, np.log(Y))
    print("Least squares solution")
    print(p[0])
    return p

def plot3d(phi):
    #plot 3d surface plot
    fig1 = figure(4)
    plt.title("Surface plot of the potential")
    ax = fig1.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, phi.T, rstride=1, cstride=1, cmap=plt.cm.jet)
    plt.colorbar(surf,shrink=0.5)
    ax.set_title("3D potential plot (in V)")
    ax.set_xlabel("x") 
    ax.set_ylabel("y")
    ax.set_zlabel("$\\phi$")
    plt.show()

def vectorPlot(phi):
    # J is defined as half the difference between adjacent phi values
    Jy = 0.5*(phi[1:-1,0:-2]-phi[1:-1,2:])
    Jx = 0.5*(-phi[2:, 1:-1]+phi[0:-2,1:-1])

    #plot quiver graph
    plt.title("Vector Plot of the current density J")
    plt.xlabel("$x\\rightarrow$")
    plt.ylabel("$y\\rightarrow$")
    plt.quiver(Y[1:-1, 1:-1], X[1:-1, 1:-1], Jy, Jx, scale=4, label="$\\vec{J}$")
    plt.scatter([y[j] for j in ii[1]], [x[j] for j in ii[0]], c='r', label='Vo = 1V')
    plt.legend()
    plt.show()
    return Jx, Jy

def heatAndTemp(Jx, Jy):
    #init temp at room temp (300K)
    temp = 300*np.ones((Nx, Ny))
    # set temps in electrode area and bottom edge = 300K
    temp[0, 1:-1]=300
    temp[ii] = 300

    #iterate Niter times with new laplace equation
    for k in range(Niter):
        oldtemp = temp.copy()
        #updation of working area
        temp[1:-1, 1:-1] = 0.25*(oldtemp[2:, 1:-1] + oldtemp[:-2, 1:-1] + oldtemp[1:-1, 2:] + oldtemp[1:-1, :-2] + (Jx**2 + Jy**2))
        #boundary conditions
        #left side
        temp[1:-1,0]=temp[1:-1,1]
        #right side
        temp[1:-1,-1]=temp[1:-1,-2]
        #bottom side
        temp[0, 1:-1]=300
        #top side
        temp[-1, 1:-1]=temp[-2, 1:-1]
        # 4 corners get updated with the average of their neighboring values
        temp[0, 0] = 0.5*(temp[0, 1] + temp[1, 0])
        temp[0, -1] = 0.5*(temp[0, -2] + temp[1, -1])
        temp[-1, 0] = 0.5*(temp[-1, 1] + temp[-2, 0])
        temp[-1, -1] = 0.5*(temp[-1, -2] + temp[-2, -1])
        #restore the electrode portion
        temp[ii]=300

    #plot the contour plot of temperature across the plate
    plt.title("Contour Plot of the temperature of the plate (in K)")
    plt.xlabel("$x\\rightarrow$")
    plt.ylabel("$y\\rightarrow$")
    levels = np.linspace(temp.min(), temp.max(), 50)
    cs = plt.contourf(Y, X, temp, levels, cmap=plt.cm.get_cmap("magma"))
    plt.colorbar(cs)
    #electrode position with red dots
    plt.scatter([y[j] for j in ii[1]], [x[j] for j in ii[0]], c='r', label='Vo = 1V')
    plt.legend()
    plt.grid(True)
    plt.show()

phi, errors = plotPhi()
plotErrors(errors)
plot3d(phi)
Jx, Jy = vectorPlot(phi)
heatAndTemp(Jx, Jy)