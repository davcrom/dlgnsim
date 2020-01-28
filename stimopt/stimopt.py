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
    

def himmelblau_gradient(p):
    
    x, y = p
    a = 4. * x * (x*x + y - 11.)
    b = 2. * (x + y*y - 7.)
    c = 4. * y * (y*y + x - 11.)
    d = 2. * (y + x*x - 7.)

    return np.array([a + b,  c + d])
    
def himmelblau_hessian(p):
    
    x, y = p
    a = 4 * (x*x + y - 11)+ 8 * x*x + 2
    b = 4 * x + 4 * y
    c = 4 * (y*y + x - 11) + 8 * y*y + 2
    
    return np.array([[a, b], [b, c]])


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


def _optimization_progress_3Dplot(optimizer, obj_func=None, ax=None):
    """
    Plot point representing progress of optimization of an objective function
    f: R^2 -> R.
    
    Parameters
    ----------
    optimizer : object
        the active optimizer object instance
        
    obj_fun : callable
        the objective function
        
    ax : matplotlib.axes._subplots.Axes3DSubplot
        an existing 3D axis on which to draw the point
    """
    if not isinstance(obj_func, callable):
        raise ValueError("objective function must be callable")
    
    if ax is None:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    
    ax.scatter(optimizer.x[0], 
               optimizer.x[1], 
               func(optimizer.x), 
               alpha=.5, 
               c='black'
               )


               )

