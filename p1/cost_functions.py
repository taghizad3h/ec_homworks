import numpy as np


def rosen(x:np.array):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def ackely(x:np.array, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    return -np.exp(-b*np.sqrt(1/d*sum(x))) - np.exp(1/d*sum(np.cos(c*x))) + a + np.exp(1)