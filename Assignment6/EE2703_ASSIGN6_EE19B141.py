# EE2703 Assignment 6
# by Saurav Sachin Kale, EE19B141

# Usage instructions:
# python3 EE2703_ASSIGN6_EE19B141.py <n> <M> <nk> <u0> <p> <Msig> <accurateUpdate>
# terms <..> are optional and to be written as just numbers without the angle brackets. So a valid command could be
# python3 EE2703_ASSIGN6_EE19B141.py 100 5 500 7 0.5 2 True
# note that accurateUpdate is "False" for false and "True" for true

import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import pylab as plab
#initialize the values to default
n = 100
M = 5
nk = 500
u0 = 5
p = 0.25
Msig = 2
accurateUpdate = False

#grab values which are available as command line arguments
try:
    n = int(argv[1])
    M = int(argv[2])
    nk = int(argv[3])
    u0 = int(argv[4])
    p = float(argv[5])
    Msig = float(argv[6])
    print(argv[7])
    if (argv[7] == "true" or argv[7] == "True"):
        accurateUpdate = True
    else:
        accurateUpdate = False
except:
    print("Few params have taken their default values")
    # do nothing.
    # an error here simply means the user wishes to leave
    # that param at default so nothing to do

#print for showing the user
print("n:",n)
print("M:",M)
print("nk:",nk)
print("u0:",u0)
print("p:",p)
print("Msig:",Msig)
print("accurateUpdate:",accurateUpdate)

#initialize the position, speed and change in position arrays
xx = np.zeros(n*M)
u = np.zeros(n*M)
dx = np.zeros(n*M)

#initialize the intensity, position and speed arrays
I = []
X = []
V = []

#find all positions where electrons are present
ii = np.where(xx > 0)[0]
allIndices = set(range(n*M))

#loop
for k in range(nk):
    #update their positions, change in positions and speeds
    dx[ii] = u[ii] + 0.5
    xx[ii] = xx[ii] + dx[ii]
    u[ii] = u[ii] + 1

    #set the position and speed of those electrons which have reached the anode to be zero
    reachedAnode = np.where(xx > n)[0]
    xx[reachedAnode] = 0
    u[reachedAnode] = 0

    #find electrons which collided
    kk = np.where(u >= u0)[0]
    ll = np.where(np.random.rand(len(kk)) <= p)[0]
    kl = kk[ll]

    # update distance (both accurate and inaccurate)
    if (not accurateUpdate):
        xx[kl] = xx[kl] - dx[kl]*np.random.rand()
        u[kl] = 0
    else:
        dt = np.random.rand(len(kl))
        xx[kl] = xx[kl] - dx[kl] + ((u[kl] - 1) * dt + 0.5* dt * dt) + 0.5*(1 - dt)**2
        u[kl]=1-dt

    #add the emission data to intensity
    I.extend(xx[kl].tolist())
    #find the required empty slots
    m = int(plab.randn()*Msig + M)
    #find the available empty slots
    emptySlots = list(allIndices - set(ii))

    # if we need more slots than are available just take available slots, else pick the first m slots
    if m > len(emptySlots):
        xx[emptySlots] = 1
        u[emptySlots] = 0
    else:
        xx[emptySlots[:m]] = 1
        u[emptySlots[:m]] = 0
    
    # find the position and speed of electrons and add to X and V
    ii = np.where(xx > 0)[0]
    X.extend(xx[ii].tolist())
    V.extend(u[ii].tolist())

#histogram for light intensity
def pltIntensity():
    plt.figure(0)
    #draw histogram
    pops, bins, thirdOneWhichWeWontUse = plt.hist(I, bins=np.arange(0, n + 1, 1), edgecolor='black', rwidth=1, color='white')
    xpos = 0.5*(bins[0:-1] + bins[1:])
    #tabulate results
    print("Intensity data:")
    print("xpos    count")
    for i in range(len(pops)):
        print(str(xpos[i]) + "    " + str(int(pops[i])))
    plt.title("Light Intensity")
    plt.xlabel("Position$\\rightarrow$")
    plt.ylabel("Intensity$\\rightarrow$")

#histogram for electron density
def plteDensity():
    plt.figure(1)
    plt.hist(X, bins=np.arange(0, n + 1, 1), edgecolor='black', rwidth=1, color='white')
    plt.title("Electron Density")
    plt.xlabel("Position$\\rightarrow$")
    plt.ylabel("Electron Density$\\rightarrow$")

#phase space diagram
def pltPhaseSpace():
    plt.figure(2)
    plt.plot(X,V,'x')
    plt.title("Electron Phase Space")
    plt.xlabel("Position$\\rightarrow$")
    plt.ylabel("Velocity$\\rightarrow$")
    plt.show()

pltIntensity()
plteDensity()
pltPhaseSpace()