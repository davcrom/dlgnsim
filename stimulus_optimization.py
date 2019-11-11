import pylgn
import numpy as np
import quantities as pq

import matplotlib.pyplot as plt

from util import Neuron, Newton

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
            
                
                
        
