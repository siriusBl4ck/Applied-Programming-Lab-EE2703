from numpy.fft.helper import fftshift
import pylab as p

def plotFT(signal, xinterval, N, doY0=False):
    # initialize the points where we will sample the function.
    # size of x is N, since N is number of samples per cycle
    t = p.linspace(xinterval[0], xinterval[1], N + 1)[:-1]
    w = p.linspace(-p.pi*N/(xinterval[1]-xinterval[0]), p.pi*N/(xinterval[1]-xinterval[0]), N + 1)[:-1]
    # sample the function
    y = signal(t)
    
    # we need to do this for odd signals to ensure the sampled function has purely imaginary coefficients
    if doY0:
        y[0] = 0
    
    y = p.fftshift(y)

    # calculate and normalize the frequency spectrum
    Y = p.fftshift(p.fft(y))/N

    phase = p.angle(Y)
    # return the magnitude and the phase
    return w, abs(Y), phase

# analysis of sin(sqrt(2)*t)

# definition
def sinroot2(x):
    return p.sin(p.sqrt(2)*x)

def Q1_1():
    # sinroot2t without the hamming window
    w, mag, phase = plotFT(sinroot2, [-p.pi, p.pi], 64, doY0=True)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $sin(\\sqrt{2}t)$ without hamming window")
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

    # showing that the periodic approximation of sinsqrt2t is not accurate enough
    x1 = p.linspace(-10, 10, 200)
    x2 = p.linspace(-p.pi, p.pi, 100)
    y1 = sinroot2(x1)
    y2 = sinroot2(x2)
    p.subplot(2, 1, 1)
    p.plot(x1, y1, label='original function')
    p.plot(x2, y2, label='sampled interval')
    p.title("Sampled interval of $sin(\\sqrt{2}t)$")
    p.legend()
    p.xlabel("t")
    p.ylabel("$sin(\\sqrt{2}t)$")
    p.grid(True)

    x3 = p.linspace(p.pi, 3*p.pi, 100)
    x4 = p.linspace(-3*p.pi, -p.pi, 100)
    p.subplot(2, 1, 2)
    p.plot(x2, y2, color='blue')
    p.plot(x3, y2, color='blue')
    p.plot(x4, y2, color='blue')
    p.title("The periodic function approximation for $sin(\\sqrt{2}t)$")
    p.xlabel("t")
    p.ylabel("$sin(\\sqrt{2}t)$")
    p.grid(True)
    p.show()

# definition of ramp function
def ramp(x):
    return x

# plots the mag plot for frequency response of ramp function
def Q1_2():
    w, mag, phase = plotFT(ramp, [-p.pi, p.pi], 64, doY0=True)
    p.semilogx(w, 20*p.log10(abs(mag)))
    p.title("Magnitude response for ramp function")
    p.xlim([1, 10])
    p.ylim([-20, 0])
    p.xticks([1, 2, 5, 10])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$ in dB")
    p.show()

# define a hammming window with given params
def hammingWindow(a, b, n):
    return p.fftshift(a + b*p.cos(2*p.pi*n/(len(n) - 1)))

def sinroot2WithHammingWindow(x):
    return sinroot2(x)*hammingWindow(0.54, 0.46, p.arange(len(x)))

def Q1_3():
    x2 = p.linspace(-p.pi, p.pi, 65)[:-1]
    x3 = p.linspace(p.pi, 3*p.pi, 65)[:-1]
    x4 = p.linspace(-3*p.pi, -p.pi, 65)[:-1]

    # plot of sinroot2t with hamming window
    y = sinroot2WithHammingWindow(x2)
    p.plot(x2, y, color='blue')
    p.plot(x3, y, color='blue')
    p.plot(x4, y, color='blue')
    p.title("$sin(\\sqrt{2}t)$ after applying Hamming Window")
    p.xlabel("$t$")
    p.ylabel("$sin(\\sqrt{2}t)$")
    p.grid(True)
    p.show()

    # Frequency response of sinroot2t with the hamming window, at low resolution
    w, mag, phase = plotFT(sinroot2WithHammingWindow, [-p.pi, p.pi], 64, doY0=True)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $sin(\\sqrt{2}t)$ with hamming window")
    p.xlim([-8, 8])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-8, 8])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

    # Frequency response of sinroot2t with the hamming window, at low resolution
    w, mag, phase = plotFT(sinroot2WithHammingWindow, [-4*p.pi, 4*p.pi], 256, doY0=True)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $sin(\\sqrt{2}t)$ with hamming window and increased resolution")
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

# define function with Wo = 0.86
def cosCube(x):
    return (p.cos(0.86*x))**3

def cosCubeWithHammingWindow(x):
    return cosCube(x)*hammingWindow(0.54, 0.46, p.arange(len(x)))

def Q2():
    # Frequency response of coscube without the hamming window
    w, mag, phase = plotFT(cosCube, [-4*p.pi, 4*p.pi], 512)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $cos^3(0.86t)$ without hamming window")
    p.xlim([-8, 8])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-8, 8])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

    # Frequency response of coscube with the hamming window
    w, mag, phase = plotFT(cosCubeWithHammingWindow, [-4*p.pi, 4*p.pi], 512)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $cos^3(0.86t)$ with hamming window")
    p.xlim([-8, 8])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-8, 8])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

# define params Wo = 0.8, delta = 0.78
def cosw0tPlusDelta(x):
    return p.cos(0.8*x + 0.78)

def cosw0tPlusDeltaWithHammingWindow(x):
    return cosw0tPlusDelta(x)*hammingWindow(0.54, 0.46, p.arange(len(x)))

# estimate the Wo and Delta from the frequency response data
def estimateWoAndDelta(w, mag, phase):
    # find the location near the peaks
    actualMag = p.where(mag > 0.2)
    print(w[actualMag])
    # take weighted average across the peaks
    wWeightedAvg = p.sum((mag[actualMag]**2) * abs(w[actualMag]))/p.sum(mag[actualMag]**2)
    # take simple average of absolute value of the phases at the peaks (got better results from simple avg rather than weighted here)
    phaseEstimate = p.mean(abs(phase[actualMag]))
    print(phase[actualMag])
    print("Estimate for Wo: ", wWeightedAvg)
    print("Estimate for delta: ", phaseEstimate)

def Q3():
    print("Q3")
    # Frequency response of cos(wot + delta) without hamming window
    w, mag, phase = plotFT(cosw0tPlusDelta, [-4*p.pi, 4*p.pi], 512)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    print("No hamming")
    estimateWoAndDelta(w, mag, phase)
    p.title("Frequency Spectrum of $cos(0.8t + 0.78)$ without hamming window")
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

    # Frequency response of cos(wot + delta) with hamming window
    w, mag, phase = plotFT(cosw0tPlusDeltaWithHammingWindow, [-4*p.pi, 4*p.pi], 512)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    print("Hamming")
    estimateWoAndDelta(w, mag, phase)
    p.title("Frequency Spectrum of $cos(0.8t + 0.78)$ with hamming window")
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

# cos(wot + delta) with some noise added to it
def noisyCosW0tPlusDelta(x):
    return p.cos(0.8*x + 0.78) + 0.1*p.randn(len(x))

def noisyCosW0tPlusDeltaWithHammingWindow(x):
    return noisyCosW0tPlusDelta(x)*hammingWindow(0.54, 0.46, p.arange(len(x)))

def Q4():
    print("Q4")
    # Frequency response of cos(wot + delta) + noise without hamming window
    w, mag, phase = plotFT(noisyCosW0tPlusDelta, [-4*p.pi, 4*p.pi], 512)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    print("No hamming")
    estimateWoAndDelta(w, mag, phase)
    p.title("Frequency Spectrum of $cos(0.8t+0.78)$ + gaussian noise without hamming window")
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

    # Frequency response of cos(wot + delta) + noise with hamming window
    w, mag, phase = plotFT(noisyCosW0tPlusDeltaWithHammingWindow, [-4*p.pi, 4*p.pi], 512)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    print("Hamming")
    estimateWoAndDelta(w, mag, phase)
    p.title("Frequency Spectrum of $cos(0.8t + 0.78)$ + gaussian noise with hamming window")
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-4, 4])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

# define the chirp
def chirp(x):
    return p.cos(16*x*(1.5 + x/(2*p.pi)))

def chirpWithHammingWindow(x):
    return chirp(x)*hammingWindow(0.54, 0.46, p.arange(len(x)))

def Q5():
    # Frequency response of chirp without hamming window
    w, mag, phase = plotFT(chirp, [-p.pi, p.pi], 1024)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $cos(16t(1.5+\\frac{t}{2\\pi}))$ without hamming window")
    p.xlim([-100, 100])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-100, 100])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

    # Frequency response of chirp with hamming window
    w, mag, phase = plotFT(chirpWithHammingWindow, [-p.pi, p.pi], 1024)
    p.subplot(2, 1, 1)
    p.plot(w, abs(mag))
    p.title("Frequency Spectrum of $cos(16t(1.5+\\frac{t}{2\\pi}))$ with hamming window")
    p.xlim([-100, 100])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("$|Y|$")
    p.subplot(2, 1, 2)
    p.scatter(w, phase, marker='o', color = '#D9DDDE')
    eligible = p.where(mag > 1e-3)
    p.scatter(w[eligible], phase[eligible], color='#48C042')
    p.xlim([-100, 100])
    p.grid(True)
    p.xlabel("$\\omega$")
    p.ylabel("Phase of $Y$")
    p.show()

def Q6():
    #define the full interval in 1024 samples
    t_full = p.linspace(-p.pi, p.pi, 1025)[:-1]
    # split it into 16x64
    t_broken = p.reshape(t_full, (16, 64))
    # mag and phase arrays for each interval
    mags = []
    phases = []
    # define the w, sampling frequency is still same, despite the breaking up in 16 intervals
    w = p.linspace(-512, 512, 65)[:-1]
    # for each interval we find FFT and append it to mags and phases
    for t in t_broken:
        y = chirp(t)
        y[0] = 0
        y = p.fftshift(y)
        Y = p.fftshift(p.fft(y))/64

        mags.append(abs(Y))
        phases.append(p.angle(Y))
    
    mags = p.array(mags)
    phases = p.array(phases)

    # plot a 3d surface plot using w as x axis, time intervals as y axis and mag and phase as z
    X = w
    Y = p.linspace(-p.pi, p.pi, 17)[:-1]

    X, Y = p.meshgrid(X, Y)
    fig = p.figure(1)
    ax = fig.add_subplot(211, projection='3d')
    surf = ax.plot_surface(X, Y, mags, cmap=p.cm.coolwarm)
    fig.colorbar(surf,shrink=0.5)
    ax.set_title("Surface plot of Magnitude response vs. frequency and time")
    ax.set_xlabel("$\\omega$") 
    ax.set_ylabel("$t$")
    ax.set_zlabel("$|Y|$")
    ax = fig.add_subplot(212, projection='3d')
    surf = ax.plot_surface(X, Y, phases, cmap=p.cm.coolwarm)
    fig.colorbar(surf,shrink=0.5)
    ax.set_title("Surface plot of Phase response vs. frequency and time")
    ax.set_xlabel("$\\omega$") 
    ax.set_ylabel("$t$")
    ax.set_zlabel("$Phase of Y$")
    p.show()

    # after adding a hamming window we plot the frequency time response surface
    mags = []
    phases = []

    w = p.linspace(-32, 32, 65)[:-1]

    for t in t_broken:
        y = chirp(t)*hammingWindow(0.54, 0.46, p.arange(len(t)))
        y[0] = 0
        y = p.fftshift(y)
        Y = p.fftshift(p.fft(y))/64

        mags.append(abs(Y))
        phases.append(p.angle(Y))
    
    mags = p.array(mags)
    phases = p.array(phases)

    X = w
    Y = 64*p.arange(16)

    X, Y = p.meshgrid(X, Y)

    fig = p.figure(1)
    ax = fig.add_subplot(211, projection='3d')
    surf = ax.plot_surface(X, Y, mags, cmap=p.cm.coolwarm)
    fig.colorbar(surf,shrink=0.5)
    ax.set_title("Surface plot of Magnitude response vs. frequency and time")
    ax.set_xlabel("$\\omega$") 
    ax.set_ylabel("$t$")
    ax.set_zlabel("$|Y|$")
    ax = fig.add_subplot(212, projection='3d')
    surf = ax.plot_surface(X, Y, phases, cmap=p.cm.coolwarm)
    fig.colorbar(surf,shrink=0.5)
    ax.set_title("Surface plot of Phase response vs. frequency and time")
    ax.set_xlabel("$\\omega$") 
    ax.set_ylabel("$t$")
    ax.set_zlabel("$Phase of Y$")
    p.show()

Q1_1()
Q1_2()
Q1_3()
Q2()
Q3()
Q4()
Q5()
Q6()