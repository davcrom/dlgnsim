import quantities as pq
from matplotlib import pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl
import pylgn.tools as tools

from util import *

def create_staticnewtwork(fb_weight=0):
    
    # network
    network = pylgn.Network()
    integrator = network.create_integrator(nt=1, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)
    
    # neurons
    ganglion = network.create_ganglion_cell()
    relay = network.create_relay_cell()
    cortical = network.create_cortical_cell()
    
    # RGC impulse-response
    delta_t = tpl.create_delta_ft()
    Wg_r = spl.create_dog_ft(A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg)
    ganglion.set_kernel((Wg_r, delta_t))
    
    # excitatory FF connection
    Krg_r = spl.create_gauss_ft(A=1, a=0.1*pq.deg)
    network.connect(ganglion, relay, (Krg_r, delta_t), weight=1)
    
    # inhibitory FF
    Krig_r = spl.create_gauss_ft(A=-1, a=0.3*pq.deg)
    network.connect(ganglion, relay, (Krig_r, delta_t), weight=0.5)

    # excitatory FB
    Krc_ex_r = spl.create_gauss_ft(A=0.5, a=0.83*pq.deg)
    network.connect(cortical, relay, (Krc_ex_r, delta_t), weight=fb_weight)
    
    # inhibitory FB
    # Krc_in_r = spl.create_gauss_ft(A=-0.5, a=0.83*pq.deg)
    # network.connect(cortical, relay, (Krc_in_r, delta_t), weight=0.5)
    
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


    
