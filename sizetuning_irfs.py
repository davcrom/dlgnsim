import quantities as pq
from dlgnsim import networks
from dlgnsim import util

nwk_default = networks.create_staticnetwork()
[relay_default] = networks.get_neuron('Relay', nwk_default)
nwk_default.compute_irf(relay_default)

nwk_small = networks.create_staticnetwork({'a_rc_in':0.1 * pq.deg})
[relay_small] = networks.get_neuron('Relay', nwk_small)
nwk_small.compute_irf(relay_small)

nwk_nofb = networks.create_staticnetwork({'w_rc_ex':0, 'w_rc_in':0})
[relay_nofb] = networks.get_neuron('Relay', nwk_nofb)
nwk_nofb.compute_irf(relay_nofb)

nwk_eq = networks.create_staticnetwork({'a_rc_in':0.3 * pq.deg})
[relay_eq] = networks.get_neuron('Relay', nwk_eq)
nwk_eq.compute_irf(relay_eq)

fig, ax = plt.subplots()
ax.plot(relay_default.irf[0][63, 32:96], label='large inhibitory fb kernel')
ax.plot(relay_eq.irf[0][63, 32:96], label='equal inhibitory fb kernel')
ax.plot(relay_small.irf[0][63, 32:96], label='small inhibitory fb kernel')
ax.plot(relay_nofb.irf[0][63, 32:96], color='k', ls='--', label='no fb')
ax.set_title('relay cell impulse-response function') 
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlabel('spatial position (deg.)')
ax.set_ylabel('magnitude (a.u.)')
ax.legend()
