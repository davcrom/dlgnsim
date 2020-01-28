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
    
    Attributes
    ----------
    f : callable
        objective function (must return int, float, or array)
        
    J : callable
        objective function Jacobian (must return array)
        
    x0 : int, float, array
        intial parameters passed to f
        
    max_iter : int
        number of iterations
    
    eps : float
        tolerance
        
    iter_func : callable
        function to be called each iteration
        
    iter_func_args : dict
        arguments passed to iter_func
    """
    
    def __init__(self, f, J, x0, max_iter=200, eps=np.finfo('float').eps,
                 iter_func=None, iter_func_args={}):
        
        self.f = f
        self.J = J
        self.x0 = x0
        self.n = 0
        self.max_iter = max_iter
        self.eps = eps
        self.iter_func = iter_func
        self.iter_func_args = iter_func_args

    def iterate(self):
        
        self.x = self.x0
        for n in range(0, self.max_iter):
            f_x = self.f(self.x)
                        
            if (abs(f_x) < self.eps).all():
                self.convergence = True
                print("Convergence at %d iterations." % self.n)
                return
            
            J_x = self.J(self.x)
            
            if isinstance(f_x, (int, float)):
                f_x = np.tile(f_x, len(J_x))
            
            if J_x.ndim == 1:
                J_x = np.diag(J_x)

            J_x += np.identity(len(J_x)) * np.finfo('float').eps
            
            try:
                self.x = self.x - np.linalg.inv(J_x) @ f_x
            
            except LinAlgError:
                self.convergence = False
                print("Singular Jacobian.")
                return
            
            if self.iter_func:
                self.iter_func(self, **self.iter_func_args)
                
            self.n = n + 1
        
        self.convergence = False
        print("Max. iterations reached.")
        
        return




               )

