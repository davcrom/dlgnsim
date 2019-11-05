import quantities as pq
from matplotlib import pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl
import pylgn.tools as tools

from util import *


# network
network = pylgn.Network()
integrator = network.create_integrator(nt=0, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)

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
network.connect(cortical, relay, (Krc_ex_r, delta_t), weight=0.5)

# inhibitory FB
Krc_in_r = spl.create_gauss_ft(A=-0.5, a=0.83*pq.deg)
network.connect(cortical, relay, (Krc_in_r, delta_t), weight=0.5)

# TC feed-forward
Kcr_r = spl.create_delta_ft()
network.connect(relay, cortical, (Kcr_r, delta_t), weight=1)
    

# stimulus
patch_diameter = 5 * pq.deg
delay = 50 * pq.ms
duration = 50 * pq.ms
stimulus = pylgn.stimulus.create_flashing_spot_ft(
    patch_diameter=patch_diameter,
    delay=delay, 
    duration=duration
    )
network.set_stimulus(stimulus)

# response
network.compute_response(relay)
#response = tools.scale_rates(relay.response[0].copy())
response = relay.response[0].copy()

# plot
im = plt.imshow(response)
plt.colorbar(im)
plt.show(block=True)
