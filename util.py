def get_neuron(name, network):
    """
    Returns a specific pylgn.Neuron, from edog_simulations repo on hitb

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

class Neuron(object):
        
    def __init__(self, x=1, y=1, m=0.1):
        self.x = x
        self.y = y
        self.m = m
        
    def response(self):
        def evaluate(s):
            return - self.m * (s - self.x) ** 2 + self.y
        return evaluate
        
    def dr(self):
        def evaluate(s):
            return -2 * self.m * (s - self.x)
        return evaluate
        
    def d2r(self):
        def evaluate(s):
            return -2 * self.m
        return evaluate
        
    def set_stimulus(self, s):
        self.s = s
        
    def get_response(self):
        response = self.response()
        return response(self.s)
    
    def get_dr(self):
        dr = self.dr()
        return dr(self.s)
        
    def get_d2R(self):
        d2r = self.d2r()
        return d2r(self.s)

class Newton(object):
    
    def __init__(self, f, df, x0, max_iter=200, eps=np.finfo('float').eps):
        
        self.f = f
        self.df = df
        self.x0 = x0
        self.max_iter = max_iter
        self.eps = eps
        
    def iterate(self):
        
        xn = self.x0
        for n in range(0, self.max_iter):
            f_xn = self.f(xn)
            if abs(f_xn) < self.eps:
                self.x = xn
                self.convergence = True
                print("Convergence at %d iterations." % n)
                return
            df_xn = self.df(xn)
            try:
                xn = xn - f_xn / df_xn
            except ZeroDivisionError:
                self.x = xn
                self.convergence = False
                print("Zero derivative, exiting iterator at %d iterations" % n)
                return
        self.x = xn
        self.convergence = False
        print("Max. iterations reached.")
        return

