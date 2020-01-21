'''
Optimize a stimulus for a certain network configuration.
'''

import quantities as pq
from matplotlib import pyplot as plt
from scipy.optimize import minimize, basinhopping

import pylgn
from util import get_neuron
from network_config import create_staticnewtwork

# define network parameters
network_params = {
    'nr':7, 'nt':0, 'dt':1*pq.ms, 'dr':0.1*pq.deg,
    'A_g':1, 'a_g':0.62*pq.deg, 'B_g':0.85, 'b_g':1.26*pq.deg,
    'w_rg':1, 'A_rg':1, 'a_rg':0.1*pq.deg,
    'w_rig':0.5, 'A_rig':-1, 'a_rig':0.3*pq.deg,
    'w_rc_ex':1, 'A_rc_ex':0.5, 'a_rc_ex':0.83*pq.deg,
    'w_rc_in':1, 'A_rc_in':-0.5, 'a_rc_in':0.83*pq.deg,
    }

# initialize network, get size
network = create_staticnewtwork(network_params)
Nr = network.integrator.Nr

# create whitenoise stimulus matching network size
whitenoise = (np.random.random((Nr, Nr)) * 2) - 1 
stimulus = pylgn.stimulus.create_array_image(whitenoise, duration=1*pq.ms)

# for normalizing power to one?
#area = area * np.sqrt(1 / np.sum(area*area))

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

