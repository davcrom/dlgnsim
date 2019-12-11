'''
Plot the impulse-response function of relay cells in a static network.
'''

import pylgn
import numpy as np
import quantities as pq

import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

from network_config import *
from util import *

params = {'nr':7, 'nt':1, 'dt':1*pq.ms, 'dr':0.05*pq.deg,
          'A_g':1, 'a_g':0.62*pq.deg, 'B_g':0.85, 'b_g':1.26*pq.deg,
          'w_rg':1, 'A_rg':1, 'a_rg':0.1*pq.deg,
          'w_rig':1, 'A_rig':-0.5, 'a_rig':0.3*pq.deg,
          'w_rc_ex':1, 'A_rc_ex':0.5, 'a_rc_ex':0.83*pq.deg,
          'w_rc_in':1, 'A_rc_in':-0.5, 'a_rc_in':0.83*pq.deg,
          }

network = create_staticnewtwork(params)    
[relay] = get_neuron('Relay', network)
network.compute_irf(relay)
irf = relay.irf[0]

positions = network.integrator.positions
Nr = network.integrator.Nr
x_id = 52
x_m = int(Nr/2-x_id)
x_p = int(Nr/2+x_id)
plt.plot(positions[x_m:x_p], irf[x_m:x_p, int(Nr/2)]) 
