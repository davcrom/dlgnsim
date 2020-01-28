'''
Optimize a stimulus for a certain network configuration.
'''

import numpy as np
import quantities as pq
from matplotlib import pyplot as plt

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
            J_xn += np.identity(len(J_xn)) * np.finfo('float').eps
            
            try:
                xn = xn - np.linalg.inv(J_xn) @ f_xn.T
            
            except LinAlgError:
                self.x = xn
                self.convergence = False
                print("Singular Jacobian.")
                return
        
        self.x = xn
        self.convergence = False
        print("Max. iterations reached.")
        
        return




               )

