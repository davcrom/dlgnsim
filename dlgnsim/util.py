'''
Helper functions for running simulations with PyLGN.
'''

import os
import json
import numpy as np
import quantities as pq

from scipy.special import erf

    
def load_json(fname):
    """
    Load a json file as a python dict.
    
    Parameters
    ----------
    fname : str
        path to json file, '.json' extension will be appended if not present
        
    Returns
    -------
    out : dict

    """
    if not fname.endswith('.json'):
        fname += '.json'
        
    with open(fname) as f:
        json_file = json.load(f)
    
    assert type(json_file) == dict
    
    return json_file


def parse_parameters(input_params=None):
    """
    Convert various input types to a dictionary containing network parameters.
    
    Parameters
    ----------
    input_params : None, dict, str
        None:   default parameters will be used
        dict:   entries in default paramters will be replaced with input
        str:    parameters will be loaded from json file specifiec by input 
    
    Returns
    -------
    out : dict
    """
    if (input_params is None) | isinstance(input_params, dict):
        # load deault parameters
        fdir = os.path.dirname(os.path.abspath(__file__))
        fpath = os.path.join(fdir, 'default_params')
        params =  load_json(fpath)
        
    elif isinstance(input_params, str):
        # import json file as dict
        params = load_json(input_params)
    
    else:
        raise ValueError("input_params type not recognized.")
        
    if isinstance(input_params, dict):
         # replace defualt parameters with those passed by user
        for key in input_params:
            params[key] = input_params[key]
            
    for key in params:
        # check for value-unit pair
        if isinstance(params[key], dict):
            try:
                # convert to quantities array
                params[key] = pq.Quantity(params[key]['value'], params[key]['unit'])
            except KeyError:
                continue
    
    return params        


def get_neuron(name, network):
    """
    Returns requested pylgn.Neuron object. Taken from 
    https://github.com/miladh/edog-simulations.

    Parameters
    ----------
    name : string
        name of the neuron

    network : pylgn.Network

    Returns
    -------
    out : pylgn.Neuron
    """
    neuron = [neuron for neuron in network.neurons if type(neuron).__name__ == name]
    
    if not neuron:
        raise NameError("neuron not found in network", name)
    
    elif len(neuron) > 1 and name == "Relay":
        raise ValueError("more than one Relay cell found in network")
    
    return neuron


def ratio_of_gaussians(x, kc, wc, ks, ws):
    """
    Ratio of Gaussians model from Cavanaugh et al. (2002 J Neurophysiol). Taken
    from djd.model.py (Busse Lab library).
    
    Parameters
    ----------
    x : float
        input variable
    kc : float
        center gain
    wc : float
        center width
    ks : float
        surround gain
    ws : float
        surround width
        
    Returns
    -------
    out : float
        RoG function with specified input parameters evaluated at x
    """
    Lc = erf(x/wc) ** 2
    Ls = erf(x/ws) ** 2
    y = kc*Lc / (1 + ks*Ls)
    
    return y
        
    

