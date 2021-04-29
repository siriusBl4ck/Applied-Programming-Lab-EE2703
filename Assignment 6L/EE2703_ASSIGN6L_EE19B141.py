#EE2703 ASSIGN6L: Laplace Transform
#by Saurav Kale, EE19B141

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

def Q1_2():
    # Q1

    # define the time vector
    t = np.linspace(0, 50, 1000)

    # we write the overall laplace domain expression for X(s)
    X = sp.lti([1, 0.5], np.polymul([1, 0, 2.25], np.polyadd(np.polymul([1, 0.5], [1, 0.5]), [2.25])))

    # get the time domain function
    t, x = sp.impulse(X, None, t)

    # plot the time domain function x
    plt.figure(0)
    plt.title("Solving for x(t)")
    plt.ylabel("$x(t)\\rightarrow$")
    plt.xlabel("$t$ (in sec)$\\rightarrow$")
    plt.grid(True)
    plt.plot(t, x, label="When decay = 0.5")

    # Q2

    # change the definition of X for a = 0.05
    X = sp.lti([1, 0.05], np.polymul([1, 0, 2.25], np.polyadd(np.polymul([1, 0.05], [1, 0.05]), [2.25])))

    # get the time domain function
    t, x = sp.impulse(X, None, t)

    # plot the time domain function x
    plt.plot(t, x, label="When decay = 0.05")
    plt.legend()
    plt.show()

def Q3():
    # define the time vector
    t = np.linspace(0, 50, 1000)

    # the system transfer function is
    H = sp.lti([1], [1, 0, 2.25])

    t, h = sp.impulse(H, None, t)

    plt.figure(1)
    plt.title("Output obtained when input frequency is varied")

    # for frequencies from 1.4 to 1.6 with 0.05 step
    for freq in np.linspace(1.4, 1.6, 5):
        # no need to include the u(t) because our t is already > 0
        f = np.cos(freq*t) * np.exp(-0.05*t)
        t, y, svec = sp.lsim(H, f, t)
        plt.plot(t, y, label="At frequency = " + str(freq))
    
    plt.legend()
    plt.ylabel("$x(t)\\rightarrow$")
    plt.xlabel("$t$  (in sec)$\\rightarrow$")
    plt.grid(True)
    plt.show()

def Q4():
    # define the time vector
    t = np.linspace(0, 20, 1000)

    # find the transfer function for X and Y
    Y = sp.lti([1, 0], np.polyadd(np.polymul([1, 0, 1], [0.5, 0, 1]), [-1]))
    X = sp.lti(np.polymul([1, 0], [0.5, 0, 1]), np.polyadd(np.polymul([1, 0, 1], [0.5, 0, 1]), [-1]))

    # get the time domain function
    t, y = sp.impulse(Y, None, t)
    t, x = sp.impulse(X, None, t)

    plt.figure(2)
    plt.title("Plots for $x(t)$ and $y(t)$ vs t")
    plt.plot(t, x, label="x(t)")
    plt.plot(t, y, label="y(t)")
    plt.legend()
    plt.ylabel("$Output signal\\rightarrow$")
    plt.xlabel("$t$ (in sec)$\\rightarrow$")
    plt.grid(True)
    plt.show()

def Q5():
    #define the transfer function
    H = sp.lti([1], [1e-12, 1e-4, 1])

    # get the bode plot
    w, S, phi = H.bode()

    plt.figure(3)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("Magnitude Plot")
    ax1.set_ylabel("$Magnitude$ (in dB)$\\rightarrow$")
    ax1.set_xlabel("$\\omega\\rightarrow$")
    plt.grid(True)
    ax1.semilogx(w, S)
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title("Phase Plot")
    ax2.set_ylabel("$Phase$ (in degrees)$\\rightarrow$")
    ax2.set_xlabel("$\\omega\\rightarrow$")
    plt.grid(True)
    ax2.semilogx(w, phi)
    plt.show()

def Q6():
    #define the transfer function
    H = sp.lti([1], [1e-12, 1e-4, 1])

    #short term response
    t = np.linspace(0, 30e-6, 10000)

    #define input for short response
    vi = np.cos(1e3*t)-np.cos(1e6*t)

    #perform convolution
    t, vo, svec = sp.lsim(H, vi, t)

    plt.figure(4)
    plt.plot(t, vo)
    plt.title("Output $v_o$ vs $t$ (short term response)")
    plt.ylabel("$v_o(t)\\rightarrow$")
    plt.xlabel("$t$ (in sec)$\\rightarrow$")
    plt.grid(True)

    # define long term time vector
    t = np.linspace(0, 1e-2, 10000)

    #define input for short response
    vi = np.cos(1e3*t)-np.cos(1e6*t)

    #perform convolution
    t, vo, svec = sp.lsim(H, vi, t)

    plt.figure(5)
    plt.title("Output $v_o$ vs $t$ (long term response)")
    plt.plot(t, vo)
    plt.ylabel("$v_o(t\\rightarrow$")
    plt.xlabel("$t$ (in sec)$\\rightarrow$")
    plt.grid(True)

    plt.show()

Q1_2()
Q3()
Q4()
Q5()
Q6()