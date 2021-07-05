# EE2703 Assignment 8
# by Saurav Sachin Kale
import pylab as p

# function to generate the fourier terms from
# - a given time domain function
# - a given time interval xinterval
# - number of samples per cycle
def plotFT(signal, xinterval, N, normalizeGauss=False):
    # initialize the points where we will sample the function.
    # size of x is N, since N is number of samples per cycle
    x = p.linspace(xinterval[0], xinterval[1], N + 1)[:-1]
    w = p.linspace(-p.pi*N/(xinterval[1]-xinterval[0]), p.pi*N/(xinterval[1]-xinterval[0]), N + 1)[:-1]
    # sample the function
    y = signal(x)
    # calculate and normalize the frequency spectrum
    Y = p.fftshift(p.fft(y))/N
    
    # gaussian curve follows a different method of normalization since it is the only non
    # periodic function we will consider.
    if normalizeGauss:
        Y = p.fftshift(abs(p.fft(y)))/N
        Y = Y * p.sqrt(2*p.pi)/max(Y)
        actualY =  p.exp(-w**2/2) * p.sqrt(2 * p.pi)
        maxError = max(abs(actualY-Y))
        phase = p.angle(Y)
        return w, abs(Y), phase, maxError

    phase = p.angle(Y)
    # return the magnitude and the phase
    return w, abs(Y), phase

def sin5(x):
    return p.sin(5*x)

# handles sin(5t)
def Q1_1():
    x = p.linspace(0,2*p.pi,128)
    y = p.sin(5*x)
    Y = p.fft(y)
    p.subplot(2,1,1)
    p.plot(abs(Y))
    p.title("Frequency Spectrum of sin(5t) without corrections")
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2,1,2)
    p.plot(p.unwrap(p.angle(Y)))
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.grid(True)
    p.show()

    w, mag, phase = plotFT(sin5, [0, 2*p.pi], 128)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of sin(5t) with corrections")
    p.xlim([-10, 10])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-10, 10])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

def AM(x):
    return (1+0.1*p.cos(x))*p.cos(10*x)

# handles (1+0.1cos(t))cos(10t)
def Q1_2():
    w, mag, phase = plotFT(AM, [0, 2*p.pi], 1024)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $(1+0.1cos(t))cos(10t)$ without corrections")
    p.xlim([-15, 15])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-15, 15])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

    w, mag, phase = plotFT(AM, [-4*p.pi, 4*p.pi], 1024)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $(1+0.1cos(t))cos(10t)$ with corrections")
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.xlim([-15, 15])
    p.grid(True)
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-15, 15])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

def sin3t(x):
    return (p.sin(x))**3

def cos3t(x):
    return (p.cos(x))**3

# handles cos^3(t) and sin^3(t)
def Q2():
    w, mag, phase = plotFT(sin3t, [-4*p.pi, 4*p.pi], 512)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $sin^3(t)$")
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.xlim([-5, 5])
    p.grid(True)
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-5, 5])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

    w, mag, phase = plotFT(cos3t, [-4*p.pi, 4*p.pi], 1024)
    p.subplot(2, 1, 1)
    p.title("Frequency Spectrum of $cos^3(t)$")
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.plot(w, abs(mag))
    p.xlim([-5, 5])
    p.grid(True)
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-5, 5])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

def cosCeption(x):
    return p.cos(20*x+5*p.cos(x))

# handles cos(20t+5cos(t))
def Q3():
    w, mag, phase = plotFT(cosCeption, [-4*p.pi, 4*p.pi], 1024)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $cos(20t+5cos(t))$")
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.xlim([-40, 40])
    p.grid(True)
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-40, 40])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

def exptSqr(x):
    return p.exp(-x*x/2)

# handles exp(-(t^2)/2)
def Q4():
    # vary the time interval as -i*pi to i*pi as i goes from 1 to 10, and print the maximum error each time
    # with respect to the actual calculated answer
    print("Maximum errors for time intervals")
    for i in range(1, 11):
        w, mag, phase, maxError = plotFT(exptSqr, [-i*p.pi, i*p.pi], 1024, normalizeGauss=True)
        print("For time interval of (-"+ str(i)+"pi, " + str(i)+"pi]"+ " Maximum error = " + str(maxError))
    
    # plot the mag and phase plot once the errors are low enough
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $exp(-\\frac{t^2}{2})$")
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.xlim([-10, 10])
    p.grid(True)
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-10, 10])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

# do it, execute them all!
Q1_1()
Q1_2()
Q2()
Q3()
Q4()

# They'll believe anything, Peter :)
