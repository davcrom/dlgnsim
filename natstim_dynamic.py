import quantities as pq
from matplotlib import pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

from util import *


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


# stimulus
stimulus = pylgn.stimulus.create_natural_movie('stimuli/MAS_1400_B.gif')
# note: MemoryError at this step for a duration of 5s
network.set_stimulus(stimulus, compute_fft=True)

# response
network.compute_response(relay)

anim = pylgn.plot.animate_cube(relay.response,
                        title="Relay cell responses",
                        dt=integrator.dt.rescale("ms"))
#plt.show(block=True)
