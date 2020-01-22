'''
Optimize a stimulus for a certain network configuration.
'''

import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from scipy.optimize import minimize, basinhopping

import pylgn

from . import util
from . import network


def himmelblau(p):
    """
    Himmelblau's function for testing optimization algorithms.
    """
    x, y = p
    a = x*x + y - 11
    b = x + y*y - 7
    
    return a*a + b*b

class Newton1D(object):
    """
    Toy example of Newton's method.
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


class Newton(object):
    """
    Newton's method for root finding.
    """
    
    def __init__(self, f, J, x0, max_iter=200, eps=np.finfo('float').eps):
        
        self.f = f
        self.J = J
        self.x0 = x0
        self.n = 0
        self.max_iter = max_iter
        self.eps = eps

    def iterate(self):
        
        xn = self.x0
        for n in range(0, self.max_iter):
            self.n = n + 1
            f_xn = self.f(xn)
            
            if abs(f_xn) < self.eps:
                self.x = xn
                self.convergence = True
                print("Convergence at %d iterations." % self.n)
                return
            
            J_xn = self.J(xn)
            
            try:
                xn = xn - np.linalg.inv(J) @ f_xn.T
            
            except LinAlgError:
                self.x = xn
                self.convergence = False
                print("Singular Jacobian.")
                return
        
        self.x = xn
        self.convergence = False
        print("Max. iterations reached.")
        
        return


# initialize network, get size
network = networks.create_staticnewtwork()
Nr = network.integrator.Nr

# create whitenoise stimulus matching network size
whitenoise = (np.random.random((Nr, Nr)) * 2) - 1 
stimulus = pylgn.stimulus.create_array_image(whitenoise, duration=1*pq.ms)

# function to return response of neuron at image center
def get_response(x, network_params):
    
    network = create_staticnewtwork(network_params)
    Nr = network.integrator.Nr
    array = x.reshape((Nr,Nr))
    stimulus = pylgn.stimulus.create_array_image(array, duration=1*pq.ms)
    network.set_stimulus(stimulus, compute_fft=True)
    [relay] = get_neuron('Relay', network)
    network.compute_response(relay)
    
    return - relay.center_response.item() 

# minimize the response function
res = minimize(get_response, 
               whitenoise.ravel(), 
               method='L-BFGS-B', 
               args=(network_params)
               )
               
basinhopping(get_response, x0=whitenoise, )

