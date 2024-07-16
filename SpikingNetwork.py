import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

class Neuron:
    """
    Defines Leaky-Integrate-and-Fire (LIF) neuron which sends and receives 
    spikes at each time step.
    """
    def __init__(self, num, V_T = 1, lam = 1, t_int = 10):
        self.num = num       # identifying number
        self.V_T = V_T       # threshold voltage
        self.lam = lam       # leakage rate
        self.t_int = t_int   # internal time steps (within spike -- int>0)
        self.V = 0           # internal voltage
        self.spikes_out = [] # output spike train
        self.V_mem = []      # internal voltages over time
        self.adj = 1         # set receiving adjustment factor
    
    def time_step(self, spikes_in, w_in, feedb_in = None, w_r = None, rec = False):
        """
        This function defines how the neuron changes with each time step.
        Effectively implements iterative LIF neuron dynamics.
        
        Inputs:
        spikes_in = numpy array of 1s or 0s corresponding to whether or not 
                    each input source sent a spike at the current time
        w_in      = numpy array of connection weights from each input source
        feedb_in  = numpy array of 1s or 0s corresponding to whether or not 
                    each other neuron sent a spike at the current time
        w_r =     = numpy array of connection weights from each other neuron
        Note: last three inputs only used for recurrent layers
        Note: based on method described in:
            - "Enabling deep spiking neural networks with hybrid conversion
               and spike timing dependent backpropagation", N Rathi et al (2020)
        """
        if rec:
            self.V = (self.lam * self.V) + np.sum(spikes_in * w_in) + np.sum(feedb_in * w_r)
        else:
            self.V = (self.lam * self.V) + np.sum(spikes_in * w_in)
        self.spikes_out.append(1 * (self.V >= self.V_T))
        self.V = (self.V * (self.V > 0)) - (self.V_T * (self.V >= self.V_T))
        self.V_mem.append(self.V)

    ### RK4 BELOW NOW DEPRECIATED (but I don't have the heart to delete it) ###

    def dvdt(self, in_V):
        return in_V - ((1 - self.lam) * self.V)

    def time_step_rk4(self, spikes_in, w_in, feedb_in = None, w_r = None, rec = False):
        """
        Solve differential equation for voltage given inputs/weights using
        Range-Kutta method (4th order) with given time steps.

        Inputs:
        spikes_in = numpy array of 1s or 0s corresponding to whether or not 
                    each input source sent a spike at the current time
        w_in      = numpy array of connection weights from each input source
        feedb_in  = numpy array of 1s or 0s corresponding to whether or not 
                    each other neuron sent a spike at the current time
        w_r =     = numpy array of connection weights from each other neuron
        Note: last three inputs only used for recurrent layers
        """
        if rec:
            in_V = self.adj * (np.sum(spikes_in * w_in) + np.sum(feedb_in * w_r))
        else:
            in_V = self.adj * np.sum(spikes_in * w_in)
        h = 1 / self.t_int
        for t in range(self.t_int):
            F1 = h * self.dvdt(in_V)
            F2 = h * self.dvdt(in_V + (F1/2))
            F3 = h * self.dvdt(in_V + (F2/2))
            F4 = h * self.dvdt(in_V + F3)
            self.V = self.V + ((1/6)*(F1 + (2*F2) + (2*F3) + F4))
        self.spikes_out.append(1*(self.V >= self.V_T))
        self.V = (self.V * (self.V > 0)) - (self.V_T * (self.V >= self.V_T))
        self.V_mem.append(self.V)

    ### END RK4 ###

    def clear_spikes(self, num = 0):
        """
        Clears internal memory of spikes. Allows used to select how far back 
        to clear spikes
        """
        if num < 1:
            self.spikes_out = []
        else:
            self.spikes_out = self.spikes_out[:-num]

    def clear_V(self, clear_V_mem = False, num = 0):
        """
        Sets internal voltage to 0. Reset of voltage memory optional.
        """
        self.V = 0
        if clear_V_mem:
            if num < 1:
                self.V_mem = []
            else:
                self.V_mem = self.V_mem[:-num]

    def __str__(self):
        """
        Defines what happens when you try to print neuron.
        """
        return "Neuron {0}".format(self.num)

    def __repr__(self): 
        """
        Defines what happens when you print within a list.
        """
        return "{0}".format(self.num)

class Layer:
    """
    This class defines a layer of a larger SNN. Contains a list of neurons and the
    weights INTO the layer.
    """
    def __init__(self, w, lam, V_T, rec, w_r):
        """
        Inputs:
            w    = weights into layer 
            lam  = internal decay rate of neurons
            V_T  = internal threshold voltage of neurons
            rec  = True/False -- whether or not layer is recurrent
            w_r  = recurrent weights into self
        """
        self.neur_list = []
        for i in range(w.shape[0]):
            new_neuron = Neuron(i+1, V_T = V_T, lam = lam)
            self.neur_list.append(new_neuron)
        self.neur_list = np.array(self.neur_list).astype(Neuron)
        self.num_neurs = self.neur_list.shape[0]
        self.w = w
        self.rec = rec
        self.w_r = w_r
        self.set_adj()

    def set_adj(self):
        """
        Sets adjustment factor for all neurons s.t. an incoming spike with weight=1
        causes an output spike to be emitted. (for RK4)
        
        Inputs:
        neurons = a list of neurons
        
        Outputs:
        neurons = identical list of neurons with proper adjustment factors set
        """
        for neur in self.neur_list:
            neur.time_step_rk4(np.array([1]), np.array([1]))
            neur.adj = 1 / (neur.V + neur.spikes_out[0])
            neur.clear_V(clear_V_mem = True)
            neur.clear_spikes()
    
    def get_spikes_out(self, start_idx, stop_idx):
        """
        Get output spikes over some duration from all neurons.

        Inputs:
        start_index = start index position of desired neuron spike output 
        stop_index = stop index position of desired neuron spike output 
        
        Outputs:
        spikes_out = 2D numpy array of spikes; shape: (number of neurons, start_idx-stop_idx)

        Note: allows user to grab spikes from END of list if stop_idx is set to 0
            - in this case, output will contain -1*start_idx spikes from end of output train
            - ex: input=(-10, 0) -> receive last 10 spikes from all neurons
        """
        if stop_idx == 0:
            num_spikes = -1 * start_idx
            spikes_out = np.zeros((self.num_neurs, num_spikes))
            for i in range(self.num_neurs):
                spikes_out[i] = self.neur_list[i].spikes_out[start_idx:]
        else:
            num_spikes = stop_idx - start_idx
            spikes_out = np.zeros((self.num_neurs, num_spikes))
            for i in range(self.num_neurs):
                spikes_out[i] = self.neur_list[i].spikes_out[start_idx:stop_idx]
        return spikes_out

    def thresh_bal(self, batch_inp, start_idx, stop_idx, t_d, reuse=False):
        v_max = 0
        t_b = stop_idx - start_idx

        # if recurrent layer, get spikes from other neurons (from prev. batch)
        if self.rec:
            if start_idx == 0:
                rec_inp = np.zeros((self.neur_list.shape[0], t_b)).T
            else:
                rec_inp = self.get_spikes_out(start_idx-t_b, stop_idx-t_b)
                if reuse:
                    rec_inp = reuse_spikes(rec_inp, t_b, t_d).T
                else:
                    rec_inp = extrap_spikes(rec_inp, t_b, t_d).T

        for i in range(self.num_neurs):
            # compute activation/output spikes across batch
            if self.rec:
                for t in range(t_b):
                    self.neur_list[i].time_step(batch_inp[t], self.w[i], rec_inp[t], self.w_r[i], rec=True)

                    # save total received voltatge if larger than any prev. observed 
                    v_max = np.max([v_max, self.neur_list[i].V + self.neur_list[i].spikes_out[-1]])
            else:
                for t in range(t_b):
                    self.neur_list[i].time_step(batch_inp[t], self.w[i])
        
                    # save total received voltatge if larger than any prev. observed 
                    v_max = np.max([v_max, self.neur_list[i].V + self.neur_list[i].spikes_out[-1]])
            
            # reset voltage and spikes
            self.neur_list[i].clear_spikes(num = t_b)
            self.neur_list[i].clear_V(clear_V_mem = True, num = t_b)
        
        # set thresholds after finding maximum voltage input
        for neur in self.neur_list:
            neur.V_T = v_max

        # recompute activation/output
        for i in range(self.num_neurs):
            if self.rec:
                for t in range(t_b):
                    self.neur_list[i].time_step(batch_inp[t], self.w[i], rec_inp[t], self.w_r[i], rec=True)
            else:
                for t in range(t_b):
                    self.neur_list[i].time_step(batch_inp[t], self.w[i])

        return self.get_spikes_out(start_idx, stop_idx).T

    def infer(self, batch_inp, start_idx, stop_idx, t_d, reuse=True):
        t_b =  stop_idx - start_idx
        # if recurrent layer, get spikes from other neurons (from prev. batch)
        if self.rec:
            if start_idx == 0:
                rec_inp = np.zeros((self.neur_list.shape[0], t_b)).T
            else:
                rec_inp = self.get_spikes_out(start_idx-t_b, stop_idx-t_b)
                if reuse:
                    rec_inp = reuse_spikes(rec_inp, t_b, t_d).T
                else:
                    rec_inp = extrap_spikes(rec_inp, t_b, t_d).T
        # compute activation/output spikes across batch
        for i in range(self.num_neurs):
            if self.rec:
                for t in range(t_b):
                    self.neur_list[i].time_step(batch_inp[t], self.w[i], rec_inp[t], self.w_r[i], rec=True)
            else:
                for t in range(t_b):
                    self.neur_list[i].time_step(batch_inp[t], self.w[i])
        return self.get_spikes_out(start_idx, stop_idx).T
    
    def clear_neurs(self):
        for neur in self.neur_list:
            neur.clear_spikes()
            neur.clear_V(clear_V_mem = True)

class SNN:
    """
    This class defines a spiking neural network containing some number of layers each 
    containing different numbers of neurons with some weights connecting the layers. 
    Weights are pre-trained on an ANN and used here to test the analogous performance 
    of spiking networks (with variatons).
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, w, lam=1, V_T=1, rec=False, w_r=None):
        """"
        Note: size of w should be: [size of layer, size of previous layer]
              if first layer, second dimension of w should be size of input 
        """
        self.layers.append(Layer(w, lam, V_T, rec, w_r))

    def thresh_bal(self, spikes_in, t_b, t_d, reuse=True):
        """
        Performs simulation of connected, spiking neurons with variable binning. Performs threshold
        balancing as inference is processed.

        Inputs:
        spikes_in = list of spikes input to first layer (size: (size of first layer, any length))
                    Note: if second dimension not multiple of batch size, remainder will be unused
        t_b       = number of steps in each time batch to be computed before passing 
                    new spikes to other neurons
        t_d       = communication delay between neurons (number of steps). no information lost if 
                    t_b = t_d. otherwise, 'reuse_spikes' or 'extrap_spikes' must be used
        reuse     = set to True to use spike reuse if t_b-t_b>0. Otherwise, rate extrapolation is 
                    employed instead
        
        Outputs:
        label = inferred classification label based on input (int from 0 to 9)
        """
        # pass input through network batch by batch
        num_batches = np.floor(spikes_in.shape[1] / t_b).astype(int)
        for batch in range(num_batches):
            t_start = batch * t_b
            t_stop = t_start + t_b
            batch_inp = spikes_in[:, t_start:t_stop].T
            for i in range(len(self.layers)):
                batch_inp = self.layers[i].thresh_bal(batch_inp, t_start, t_stop, t_d, reuse=reuse)
        
        # clear spike outputs and voltages for all layers
        for layer in self.layers:
            layer.clear_neurs()

    def infer(self, spikes_in, t_b, t_d, reuse=True):
        """
        Performs simulation of connected, spiking neurons with variable binning.

        Inputs:
        spikes_in   = list of spikes input to first layer (size: (size of first layer, any length))
                      Note: if second dimension not multiple of batch size, remainder will be unused
        t_b         = number of steps in each time batch to be computed before passing 
                      new spikes to other neurons
        t_d         = communication delay between neurons (number of steps). no information lost if 
                      t_b = t_d. otherwise, 'reuse_spikes' or 'extrap_spikes' must be used

        Outputs:
        activation  = numpy array of real numbers of size: [number of spikes in final layer]
                      Note: created by summing last layer's output spikes (equiv. to ReLU)
        """
        # pass input through network batch by batch
        num_batches = np.floor(spikes_in.shape[1] / t_b).astype(int)
        for batch in range(num_batches):
            t_start = batch * t_b
            t_stop = t_start + t_b
            batch_inp = spikes_in[:, t_start:t_stop].T
            for i in range(len(self.layers)):
                batch_inp = self.layers[i].infer(batch_inp, t_start, t_stop, t_d, reuse=reuse)

        # convert output spikes from last layer into single real number (sum i.e. ReLU)
        spikes_out = self.layers[-1].get_spikes_out(-1 * num_batches * t_b, 0)
        activation = np.zeros(self.layers[-1].num_neurs)
        for i in range(activation.shape[0]):
            activation[i] = np.sum(spikes_out[i])
        #activation[activation < 0] = 0 * activation[activation < 0] # I know lol... but ReLU

        return activation
    
    def clear_lays(self):
        # clear spike outputs and voltage for all layers
        for layer in self.layers:
            layer.clear_neurs()

    def __str__(self):
        num_layers = len(self.layers)
        num_neurs = [layer.num_neurs for layer in self.layers]
        return "Custom SNN -- {0} layer(s) w/ {1} neurons".format(num_layers, num_neurs)
    
    def __repr__(self):
        num_layers = len(self.layers)
        num_neurs = [layer.num_neurs for layer in self.layers]
        return "Custom SNN -- {0} layer(s) w/ {1} neurons".format(num_layers, num_neurs)
    
    def print(self):
        num_layers = len(self.layers)
        num_neurs = [layer.num_neurs for layer in self.layers]
        return "Custom SNN -- {0} layer(s) w/ {1} neurons".format(num_layers, num_neurs)

def poisson_train(rates, num_steps):

    """"
    Takes as input an array of rates, and outputs poissonion spike trains of length 
    num_steps.

    Inputs:
    rates     = 1-D array of rate values
    num_steps = number of time steps
    
    Outputs:
    spikes    = numpy array of spike trains (shape: (num_neurons, num_steps))
    """
    # create poissonian spike train for each neuron
    num_neurons = rates.shape[0]
    spikes = np.empty((num_neurons, num_steps))
    for i in range(num_neurons):
        spike_thresh = rates[i]
        rand_nums = rand(num_steps)
        spikes[i, :] = (rand_nums < spike_thresh).astype(int)

    return spikes

def reuse_spikes(spikes, t_b, t_d):
    """
    Utilizes unused spikes in batch to fill information gap from mismatch
    between batch size and delay -- last t_b-t_d elements have no spikes to
    use for calculation. Also properly lines up spikes to calculate resultant  
    output spike at a time given by the delay.

    Inputs:
    spikes = list of spikes from many neurons to perform extrapolation with
             (shape = (num_neurons, t_b))
    t_b    = number of steps in batch
    t_d    = communication delay between neurons

    Outputs:
           = altered (properly rolled) list of spikes (shape = (num_neurons, t_b))
    """
    return np.roll(spikes, t_b - t_d, axis = 1)

def extrap_spikes(spikes, t_b, t_d):
    """
    Extrapolates from spikes in batch to fill information gap from mismatch
    between batch size and delay -- last t_b-t_d elements have no spikes to
    use for calculation. Also properly lines up spikes to calculate resultant  
    output spike at a time given by the delay. Uses poissonian rate generation 
    based on batch rate to extrapolate.

    Inputs:
    spikes = list of spikes from many neurons to perform extrapolation with
             (shape = (num_neurons, t_b))
    t_b    = number of steps in batch
    t_d    = communication delay between neurons

    Outputs:
    extrap = list of spikes from all neurons for given window 
             (shape = (num_neurons, t_b))
    """
    if t_b > t_d:
        for i in range(spikes.shape[0]):
            spikes_curr = spikes[i]
            rate = np.sum(spikes_curr) / t_b
            rand_nums = rand(t_b - t_d)
            spikes_new = np.roll(spikes_curr, t_b - t_d)
            spikes_new[-(t_b-t_d):] = (rand_nums < rate).astype(int)
            if i == 0:
                extrap = spikes_new
            elif i == 1:
                extrap = np.append([spikes], [spikes_new], axis=0)
            else:
                extrap = np.append(spikes, [spikes_new], axis=0)
        return extrap
    else:
        extrap = spikes
        return extrap