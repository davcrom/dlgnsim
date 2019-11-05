import pylgn
import numpy as np
import quantities as pq

import matplotlib.pyplot as plt

# create some trivial neuron reponse model
class Neuron(object):
        
    def __init__(self, x=1, y=1, m=0.1):
        self.x = x
        self.y = y
        self.m = m
        
    def response(self):
        def evaluate(s):
            return - self.m * (s - self.x) ** 2 + self.y
        return evaluate
        
    def dr(self):
        def evaluate(s):
            return -2 * self.m * (s - self.x)
        return evaluate
        
    def d2r(self):
        def evaluate(s):
            return -2 * self.m
        return evaluate
        
    def set_stimulus(self, s):
        self.s = s
        
    def get_response(self):
        response = self.response()
        return response(self.s)
    
    def get_dr(self):
        dr = self.dr()
        return dr(self.s)
        
    def get_d2R(self):
        d2r = self.d2r()
        return d2r(self.s)
        
# create Newton's method object class
class Newton(object):
    
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

# initialize a stimulus space
stim = np.linspace(0, 10, 100)
# create a neuron
neuron = Neuron(x=3, y=1, m=0.01)

# plot responses across stimulus space
fig, ax = plt.subplots()
ax.set_xlabel("Stimulus parameter")
ax.set_label("Neuron response")
for s in stim:
    neuron.set_stimulus(s)
    ax.scatter(s, neuron.get_response(), color='black')
    
# instantiate iterator for neuron response derivatives
s0 = 0 # pick an initial stimulus value
newton = Newton(neuron.dr(), neuron.d2r(), s0)
newton.iterate()
print("Optimal response at s = %01f" % newton.x)
neuron.set_stimulus(newton.x)
ax.scatter(newton.x, neuron.get_response(), s=50, color='red')
            
                
                
        
