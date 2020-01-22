'''
Functions for creating and connecting PyLGN networks.
'''

import quantities as pq
from matplotlib import pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

from . import util    


def create_staticnewtwork(params=None):
    """
    Create a PyLGN network where all temporal kernels are delta functions.
    
    Parameters
    ----------
    params : None, dict, str
        passed to util._parse_parameters
    """
    
    params = util.parse_parameters(params)
    
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
    
    # RGC impulse-response function
    delta_t = tpl.create_delta_ft()
    Wg_r = spl.create_dog_ft(
        A=params['A_g'], 
        a=params['a_g'], 
        B=params['B_g'], 
        b=params['b_g']
        )
    ganglion.set_kernel((Wg_r, delta_t))
    
    # excitatory FF connection
    Krg_r = spl.create_gauss_ft(A=params['A_rg_ex'], a=params['a_rg_ex'])
    network.connect(ganglion, relay, (Krg_r, delta_t), weight=params['w_rg_ex'])
    
    # inhibitory FF
    Krig_r = spl.create_gauss_ft(A=params['A_rg_in'], a=params['a_rg_in'])
    network.connect(ganglion, relay, (Krig_r, delta_t), weight=params['w_rg_in'])

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


def create_dynamicnetwork(params=None):
    """
    Create PyLGN network with temporal kernels.
    
    Parameters
    ----------
    params : None, dict, str
        passed to util._parse_parameters
    """
    
    params = util.parse_parameters(params)

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
    
    # RGC impulse-response function
    Wg_r = spl.create_dog_ft(
        A=params['A_g'], 
        a=params['a_g'], 
        B=params['B_g'], 
        b=params['b_g']
        )
    Wg_t = tpl.create_biphasic_ft(
        phase=params['phase_g'], 
        damping=params['damping_g']
        )
    ganglion.set_kernel((Wg_r, Wg_t))
    
    # excitatory FF connection
    Krg_r = spl.create_gauss_ft(A=params['A_rg_ex'], a=params['a_rg_ex'])
    Krg_t = tpl.create_exp_decay_ft(
        tau=params['tau_rg_ex'], 
        delay=params['delay_rg_ex']
        )
    network.connect(ganglion, relay, (Krg_r, Krg_t), weight=params['w_rg_ex'])
    
    # inhibitory FF
    Krig_r = spl.create_gauss_ft(A=params['A_rg_in'], a=params['a_rg_in'])
    Krig_t = tpl.create_exp_decay_ft(
        tau=params['tau_rg_in'], 
        delay=params['delay_rg_in']
        )
    network.connect(ganglion, relay, (Krig_r, Krig_t), weight=params['w_rg_in'])
    
    # excitatory FB
    Krc_ex_r = spl.create_gauss_ft(A=params['A_rc_ex'], a=params['a_rc_ex'])
    Krc_ex_t = tpl.create_exp_decay_ft(
        tau=params['tau_rc_ex'], 
        delay=params['delay_rc_ex'])
    network.connect(cortical, relay, (Krc_ex_r, Krc_ex_t), weight=params['w_rc_ex'])
    
    # inhibitory FB
    Krc_in_r = spl.create_gauss_ft(A=params['A_rc_in'], a=params['a_rc_in'])
    Krc_in_t = tpl.create_exp_decay_ft(
        tau=params['tau_rc_in'], 
        delay=params['delay_rc_in'])
    network.connect(cortical, relay, (Krc_in_r, Krc_in_t), weight=params['w_rc_in'])
    
    # TC feed-forward
    Kcr_r = spl.create_delta_ft()
    Kcr_t = tpl.create_delta_ft()
    network.connect(relay, cortical, (Kcr_r, Kcr_t), weight=params['w_cr'])

    return network
    
