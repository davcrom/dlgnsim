'''
Helper functions for running simulations with PyLGN.
'''

import numpy as np

from scipy.special import erf

def get_neuron(name, network):
    """
    From the Mobarhan edog_simulations repo on github.
    Returns a specific pylgn.Neuron

    Parameters
    ----------
    name : string
         Name of the neuron

    network : pylgn.Network

    Returns
    -------
    out : pylgn.Neuron

    """
    neuron = [neuron for neuron in network.neurons if type(neuron).__name__ == name]
    if not neuron:
        raise NameError("neuron not found in network", name)
    elif len(neuron) > 1 and name == "Relay":
        raise ValueError("more than one Relay cell found in network")
    return neuron

def ratio_of_gaussians(x, kc, wc, ks, ws):
    """
    From djd.model.py
    Ratio of Gaussians model (Cavanaugh et al. 2002 J Neurophysiol)
        kc: gain center
        wc: width center (sigma)
        ks: gain surround
        ws: width surround (sigma)
    """
    
    Lc = erf(x/wc) ** 2
    Ls = erf(x/ws) ** 2
    y = kc*Lc / (1 + ks*Ls)
    return y
    
def himmelblau(p):
    """
    Himmelblau's function for testing optimization algorithms.
    """
    
    x, y = p
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b

class Newton(object):
    """
    A toy example of newton's method for root finding in a 1D function.
    """
    
    def __init__(self, f, df, x0, max_iter=200, eps=np.finfo('float').eps):
        
        self.f = f
        self.df = df
        self.x0 = x0
        self.max_iter = max_iter
        self.eps = eps
        
    def iterate(self):
        
        xn = self.x0
        for n in range(0, self.max_iter):
            f_xn = self.f(xn)
            if abs(f_xn) < self.eps:
                self.x = xn
                self.convergence = True
                print("Convergence at %d iterations." % n)
                return
            df_xn = self.df(xn)
            try:
                xn = xn - f_xn / df_xn
            except ZeroDivisionError:
                self.x = xn
                self.convergence = False
                print("Zero derivative, exiting iterator at %d iterations" % n)
                return
        self.x = xn
        self.convergence = False
        print("Max. iterations reached.")
        return

