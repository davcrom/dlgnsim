import pylgn
import numpy as np
import quantities as pq

import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

from network_config import *
from util import *

def spotstim_response(diameter, fb_weight):
    
    network = create_staticnewtwork(fb_weight=fb_weight)    
    stimulus = pylgn.stimulus.create_flashing_spot_ft(
        patch_diameter=diameter * pq.deg,
        duration=1 * pq.ms,
        )
    network.set_stimulus(stimulus)
    relay = get_neuron('Relay', network)[0]
    network.compute_response(relay)
    response = relay.center_response[0].item()
                
    return -1 * response # return negative in order to find peak            

# perform optimization for 3 different excitatory feedback weight strengths
# note: scipy.optimize.newton fails to converge
# note: 'Newton-CG' method for minimize requires Jacobian vector
d0 = 5
min0 = minimize(spotstim_response, d0, args=(0,), method='CG')
min1 = minimize(spotstim_response, d0, args=(0.5,), method='CG')
min2 = minimize(spotstim_response, d0, args=(1,), method='CG')

# note: qualitatively reproduces fig7 (top left) from Mobarhan et al. (2018)
