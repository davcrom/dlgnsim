''' 
Plot the responses of a static network to a series of grating patches of varying 
diameter and spatial frequency.
'''

import pylgn
import numpy as np
import quantities as pq

import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

from network_config import *
from util import *

params = {'nr':7, 'nt':1, 'dt':1*pq.ms, 'dr':0.1*pq.deg,
          'A_g':1, 'a_g':0.62*pq.deg, 'B_g':0.85, 'b_g':1.26*pq.deg,
          'w_rg':1, 'A_rg':1, 'a_rg':0.1*pq.deg,
          'w_rig':0.5, 'A_rig':-1, 'a_rig':0.3*pq.deg,
          'w_rc_ex':1, 'A_rc_ex':0.5, 'a_rc_ex':0.83*pq.deg,
          'w_rc_in':0, 'A_rc_in':-0.5, 'a_rc_in':0.83*pq.deg,
          }

npts = 30
diameters = np.linspace(0, 10, npts)
network = create_staticnewtwork(params)
sfs = network.integrator.spatial_angular_freqs
inds = np.linspace(0, network.integrator.Nr / 2, npts, endpoint=False, dtype='int')
wavenumbers = sfs[inds] 
responses = np.full((len(diameters),len(wavenumbers)), np.nan)
for i, diameter in enumerate(diameters):
    for j, wavenumber in enumerate(wavenumbers):
        print(i,j)
        network = create_staticnewtwork(params)   
        stimulus = pylgn.stimulus.create_patch_grating_ft(
            wavenumber=wavenumber,
            patch_diameter=diameter * pq.deg
            )
        network.set_stimulus(stimulus)
        relay = get_neuron('Relay', network)[0]
        network.compute_response(relay)
        response = relay.center_response[0].item()
                    
        responses[i,j] = response            

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(diameters,wavenumbers)
ax.plot_surface(x, y, responses.T, cmap=cm.coolwarm)
ax.set_xlabel('patch diameter')
ax.set_ylabel('wave number') 
