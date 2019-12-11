'''
Functions for creating and connecting PyLGN networks.
'''

import quantities as pq
from matplotlib import pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl
import pylgn.tools as tools

from util import *

def create_staticnewtwork(input_params={}):
    
    # default parameters
    params = {'nr':7, 'nt':1, 'dt':1.*pq.ms, 'dr':0.1*pq.deg,
              'A_g':1., 'a_g':0.62*pq.deg, 'B_g':0.85, 'b_g':1.26*pq.deg,
              'w_rg':1., 'A_rg':1., 'a_rg':0.1*pq.deg,
              'w_rig':1, 'A_rig':-0.5, 'a_rig':0.3*pq.deg,
              'w_rc_ex':1, 'A_rc_ex':0.5, 'a_rc_ex':0.83*pq.deg,
              'w_rc_in':1, 'A_rc_in':-0.5, 'a_rc_in':0.83*pq.deg,
              }
    
    # replace defualt parameters with those passed by user
    for p in input_params:
        params[p] = input_params[p]
    
    # network
    network = pylgn.Network()
    integrator = network.create_integrator(
        nt=params['nt'], 
        nr=params['nr'], 
        dt=params['dt'], 
        dr=params['dr']
        )
        
    # neurons
    ganglion = network.create_ganglion_cell()
    relay = network.create_relay_cell()
    cortical = network.create_cortical_cell()
    
    # RGC impulse-response
    delta_t = tpl.create_delta_ft()
    Wg_r = spl.create_dog_ft(
        A=params['A_g'], 
        a=params['a_g'], 
        B=params['B_g'], 
        b=params['b_g']
        )
    ganglion.set_kernel((Wg_r, delta_t))
    
    # excitatory FF connection
    Krg_r = spl.create_gauss_ft(A=params['A_rg'], a=params['a_rg'])
    network.connect(ganglion, relay, (Krg_r, delta_t), weight=params['w_rg'])
    
    # inhibitory FF
    Krig_r = spl.create_gauss_ft(A=params['A_rig'], a=params['a_rig'])
    network.connect(ganglion, relay, (Krig_r, delta_t), weight=params['w_rig'])

    # excitatory FB
    Krc_ex_r = spl.create_gauss_ft(A=params['A_rc_ex'], a=params['a_rc_ex'])
    network.connect(cortical, relay, (Krc_ex_r, delta_t), weight=params['w_rc_ex'])
    
    # inhibitory FB
    Krc_in_r = spl.create_gauss_ft(A=params['A_rc_in'], a=params['a_rc_in'])
    network.connect(cortical, relay, (Krc_in_r, delta_t), weight=params['w_rc_in'])
    
    # TC feed-forward
    Kcr_r = spl.create_delta_ft()
    network.connect(relay, cortical, (Kcr_r, delta_t), weight=1)
    
    return network

def create_dynamicnetwork():
    # network
    nt = 10
    dt = 1 * pq.ms
    nr = 7
    dr = 0.1 * pq.deg
    network = pylgn.Network()
    integrator = network.create_integrator(nt=nt, nr=nr, dt=dt, dr=dr)
    
    # neurons
    ganglion = network.create_ganglion_cell()
    relay = network.create_relay_cell()
    cortical = network.create_cortical_cell()
    
    # RGC impulse-response
    Wg_r = spl.create_dog_ft(A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg)
    Wg_t = tpl.create_biphasic_ft(phase=43*pq.ms, damping=0.38)
    ganglion.set_kernel((Wg_r, Wg_t))
    
    # excitatory FF connection
    Krg_r = spl.create_gauss_ft(A=1, a=0.1*pq.deg)
    Krg_t = tpl.create_exp_decay_ft(tau=5*pq.ms, delay=0*pq.ms)
    network.connect(ganglion, relay, (Krg_r, Krg_t), weight=1)
    
    # inhibitory FF
    Krig_r = spl.create_gauss_ft(A=-1, a=0.3*pq.deg)
    Krig_t = tpl.create_exp_decay_ft(tau=5*pq.ms, delay=3*pq.ms)
    network.connect(ganglion, relay, (Krig_r, Krig_t), weight=0.5)
    
    # excitatory FB
    Krc_ex_r = spl.create_gauss_ft(A=0.5, a=0.83*pq.deg)
    Krc_ex_t = tpl.create_exp_decay_ft(tau=5*pq.ms, delay=15*pq.ms)
    network.connect(cortical, relay, (Krc_ex_r, Krc_ex_t), weight=0.5)
    
    # inhibitory FB
    Krc_in_r = spl.create_gauss_ft(A=-0.5, a=0.83*pq.deg)
    Krc_in_t = tpl.create_exp_decay_ft(tau=5*pq.ms, delay=15*pq.ms)
    network.connect(cortical, relay, (Krc_in_r, Krc_in_t), weight=0.5)
    
    # TC feed-forward
    Kcr_r = spl.create_delta_ft()
    Kcr_t = tpl.create_delta_ft()
    network.connect(relay, cortical, (Kcr_r, Kcr_t), weight=1)


    
