def get_neuron(name, network):
    """
    Returns a specific pylgn.Neuron

    Parameters
    ----------
    name : string
         Name of the neuron

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

