# EE2703 Endsem Assignment
# by Saurav Sachin Kale, EE19B141

import pylab as p
import scipy
import scipy.special as sp

# mu0
mu0 = 8.85418782e-12
# I0
I0 = 4*p.pi/mu0
N = 100
# Dimensions
u = 3
v = 3
w = 1000
a = 10
# k = w / c (not to be confused by i, j, k so im calling it wByC)
wByC = 0.1

# Q2
# dividing the volume into 3 x 3 x 1000
xCoords = p.linspace(-(u - 1)/2, (u - 1)/2, u)
yCoords = p.linspace(-(v - 1)/2, (v - 1)/2, v)
zCoords = p.linspace(1, w, w)

# Q3
# dividing angle into segments to obtain 100 dl segments
phi = p.linspace(0, 2*p.pi, N+1)[:-1]
# initialize magnitude of current I. we can ignore exp(jwt) because at no point in the analysis do we involved time

# Q4
# so the exp(jwt) factor will be present in Bz, and when we take magnitude, it will vanish
I = I0*p.cos(phi)
ringpos_x = a*p.cos(phi)
ringpos_y = a*p.sin(phi)
# dphi = 2pi/100
dphi = phi[1] - phi[0]
# |dl| = radius * dphi
mag_dl = a*dphi

# dl vector = magnitude * direction
dl_x = -mag_dl*p.sin(phi)
dl_y = mag_dl*p.cos(phi)

# current element J = I*dl
JX = I*dl_x
JY = I*dl_y

#plot quiver graph
p.title("Vector Plot of the current elements")
p.xlabel("$x\\rightarrow$")
p.ylabel("$y\\rightarrow$")
p.quiver(ringpos_x, ringpos_y, JX/40, JY/40, label="$current element$")
p.scatter((ringpos_x[1:] + ringpos_x[:-1])/2, (ringpos_y[1:] + ringpos_y[:-1])/2)
p.legend()
p.grid()
p.show()

# Q 5 and 6
# calc function
def calc(l):
    # create meshgrid
    xx, yy, zz = p.meshgrid(xCoords, yCoords, zCoords, indexing='ij')
    # find Rijk vector
    xx = xx - ringpos_x[l]
    yy = yy - ringpos_y[l]
    # find |Rijk|
    Rijk = p.sqrt(xx**2 + yy**2 + zz**2)
    # return x component and y component of A vector
    return (I[l]/I0)*p.exp(complex(0, -1)*wByC*Rijk)*dl_x[l]/Rijk, (I[l]/I0)*p.exp(complex(0, -1)*wByC*Rijk)*dl_y[l]/Rijk

# initialize A_x and A_y
A_x = 0
A_y = 0
# Q7
# populate all the values of A_x and A_y
# we have to use a loop here because for vectorized operations the dimensions
# of the two arrays have to match, which would be cumbersome,
# because we have 100 points on our loop, but 9000 points in our mesh
for l in range(N):
    dA_x, dA_y = calc(l)
    if l == 0:
        A_x = dA_x
        A_y = dA_y
    else:
        A_x += dA_x
        A_y += dA_y

# Q8
# B = del x A
# B(z) = (A_y[deltaX, 0, z] - A_y[-deltaX, 0, z] - A_x[0, deltaY, z] + A[0, -deltaY, z])/2
Bz = (A_y[u//2 + 1, v//2, :] - A_y[u//2 - 1, v//2, :] - A_x[u//2, v//2 + 1, :] + A_x[u//2, v//2 - 1, :])/2

# Q9
# plot loglog
p.loglog(zCoords, abs(Bz))
p.title("Magnitude of Magnetic field $\\vec{B}$ along z axis")
p.ylabel("$|B(z)|$")
p.xlabel("z$\\rightarrow$")
p.grid()
p.show()

# Q10
# curve fitting using lstsq
M = p.c_[p.log10(zCoords), p.ones(w)]
G = p.log10(abs(Bz))
r = scipy.linalg.lstsq(M, G)[0]
c = 10**(r[1])
b = r[0]
print("Least Squares fitting result")
print("c = " + str(c))
print("b = " + str(b))
fit = c*(zCoords**b)
# plot fit vs actual func
p.loglog(zCoords, abs(Bz), label="Original")
p.loglog(zCoords, fit, label="Fit")
p.title("Fitting $cz^b$ to the plot of magnetic field")
p.ylabel("$|B(z)|$")
p.xlabel("z$\\rightarrow$")
p.grid()
p.legend()
p.show()


# ========================= Statics case ========================
print("\nStatics case")
# static current
I = I0*p.cos(phi)
# we need our dAx, dAy = dl_x/Rijk, dl_y/Rijk, so we remove the exponential part by setting the exponent to zero
# so we do that by setting k = w / c to zero
wByC = 0
# current element J = I*dl
JX = I*dl_x
JY = I*dl_y

#plot quiver graph
p.title("Vector Plot of the current elements for statics case")
p.xlabel("$x\\rightarrow$")
p.ylabel("$y\\rightarrow$")
p.quiver(ringpos_x, ringpos_y, JX/40, JY/40, label="$current element$", )
p.scatter((ringpos_x[1:] + ringpos_x[:-1])/2, (ringpos_y[1:] + ringpos_y[:-1])/2)
p.legend()
p.grid()
p.show()

# initialize A_x and A_y
A_x = 0
A_y = 0

# populate all the values of A_x and A_y
for l in range(N):
    dA_x, dA_y = calc(l)
    if l == 0:
        A_x = dA_x
        A_y = dA_y
    else:
        A_x += dA_x
        A_y += dA_y

# B = del x A
Bz = (A_y[u//2 + 1, v//2, :] - A_y[u//2 - 1, v//2, :] - A_x[u//2, v//2 + 1, :] + A_x[u//2, v//2 - 1, :])/2

# plot loglog
p.loglog(zCoords, abs(Bz))
p.title("Magnitude of Magnetic field $\\vec{B}$ along z axis for statics case")
p.ylabel("$|B(z)|$")
p.xlabel("z$\\rightarrow$")
p.grid()
p.show()

# curve fitting using lstsq
M = p.c_[p.log10(zCoords), p.ones(w)]
G = p.log10(abs(Bz))
r = scipy.linalg.lstsq(M, G)[0]
c = 10**(r[1])
b = r[0]
print("Least Squares fitting result")
print("c = " + str(c))
print("b = " + str(b))
fit = c*(zCoords**b)
# plot fit vs actual func
p.loglog(zCoords, abs(Bz), label="Original")
p.loglog(zCoords, fit, label="Fit")
p.title("Fitting $cz^b$ to the plot of magnetic field for statics case")
p.ylabel("$|B(z)|$")
p.xlabel("z$\\rightarrow$")
p.grid()
p.legend()
p.show()

# ========================= constant current case ========================
print("\nConstant Current case")
# constant current
I = I0*p.ones(N)
# we need our dAx, dAy = dl_x/Rijk, dl_y/Rijk, so we remove the exponential part by setting the exponent to zero
# so we do that by setting k = w / c to zero
wByC = 0
# current element J = I*dl
JX = I*dl_x
JY = I*dl_y

#plot quiver graph
p.title("Vector Plot of the current elements for constant current case")
p.xlabel("$x\\rightarrow$")
p.ylabel("$y\\rightarrow$")
p.quiver(ringpos_x, ringpos_y, JX/40, JY/40, label="$current element$")
p.scatter((ringpos_x[1:] + ringpos_x[:-1])/2, (ringpos_y[1:] + ringpos_y[:-1])/2)
p.legend()
p.grid()
p.show()

# initialize A_x and A_y
A_x = 0
A_y = 0

# populate all the values of A_x and A_y
for l in range(N):
    dA_x, dA_y = calc(l)
    if l == 0:
        A_x = dA_x
        A_y = dA_y
    else:
        A_x += dA_x
        A_y += dA_y

# B = del x A
Bz = (A_y[u//2 + 1, v//2, :] - A_y[u//2 - 1, v//2, :] - A_x[u//2, v//2 + 1, :] + A_x[u//2, v//2 - 1, :])/2

# plot loglog
p.loglog(zCoords, abs(Bz))
p.title("Magnitude of Magnetic field $\\vec{B}$ along z axis for constant current case")
p.ylabel("$|B(z)|$")
p.xlabel("z$\\rightarrow$")
p.grid()
p.show()

# curve fitting using lstsq
M = p.c_[p.log10(zCoords), p.ones(w)]
G = p.log10(abs(Bz))
r = scipy.linalg.lstsq(M, G)[0]
c = 10**(r[1])
b = r[0]
print("Least Squares fitting result")
print("c = " + str(c))
print("b = " + str(b))
fit = c*(zCoords**b)
# plot fit vs actual func
p.loglog(zCoords, abs(Bz), label="Original")
p.loglog(zCoords, fit, label="Fit")
p.title("Fitting $cz^b$ to the plot of magnetic field for constant current case")
p.ylabel("$|B(z)|$")
p.xlabel("z$\\rightarrow$")
p.grid()
p.legend()
p.show()