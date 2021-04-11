#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F


# In[9]:


class CTLSTM(nn.Module):
    
    def __init__(self, K, hD):
        super().__init__()
        
        # K : input dimension
        self.K = K
        
        # hD : dimensionality of hidden nodes
        self.hD
        
        # input to L_U will be a feature vector, which is K-dim
        self.L_U = nn.Linear(K, 6*hD) # for each of the six gates + delta
        
        # input to L_V will be h_t, which is hD dim
        self.L_V = nn.Linear(hD, 6*hD)
        
        # Remember : the decay rate is one-dimensional
        # it has different non-linearity, so it needs to be done separately
        self.D_U == nn.Linear(K, 1)
        self.D_V == nn.Linear(hD, 1)
        
        # We need another linear layer to compute lambda_tilde
        # This layer takes hD-dimensional h(t) and returns K-dimensional vector
        self.L_lamb_til = nn.Linear(hD, K)
        
        # Then, to predict lambda from lambda_tilde using softplus,
        # we need scaling parameters
        # we need to make these scales a part of the
        # learnable parameter set
        self.scale = nn.Parameter(pt.rand(K, requires_grad=True))
        
        # let's work with reLU for now
        self.sigma = F.sigmoid
    
    def MC_Loss(self, times, Clows, Cbars, deltas, OutGates, Nsamples=1000):
        
        # All the inputs are constructed during the forward pass
        # on a sequence of events. See the forward function
        # To compute the integral, we'll use "Nsamples" samples
        # Our time invterval will be between 0 to times[sampleIndex]
        trands = pt.rand(Nsamples)*times
        
        # Once the random time instants have been formed, we need to store
        # the intervals in which they lie
        # Our assumption here is that the input "times" is an ascending-order
        # sorted array, and that times[0] =  0
        
        t_up = pt.searchsorted(times, trands)
        # t_up[i] = idx, such that times[idx-1]<trands[i]<times[idx]
        
        # Using the intervals in which the sample times lie,
        # we have to find the rates
        
        I = torch.tensor(0.)
        for tInd in range(trands.shape[0]):
            t = trands[tInd]
            
            # Need to use cbar, clow and delta for the next time index
            clow = Clows[t_up[tInd]]
            cbar = Cbars[t_up[tInd]]
            delta = deltas[t_up[tInd]]
            tlow = times[t_up[tInd-1]]
            
            # compute c(t)
            # Note here we use "t - tlow"
            ct = cbar + (clow - cbar)*pt.exp((t - tlow)*delta)
            
            # compute h(t)
            o = OutGates[t_up[tInd]]
            ht = o * (2*self.sigma(2*ct) - 1)
            
            # compute lambda_k(t)
            lamb_til = self.L_lamb_til(ht.view(-1, K)).view(K)
            
            # this will contain the event intensities for all the K events
            lamb = s * pt.log(1 + pt.exp(lamb_til / s))
            
            # get the sum total rate of all events
            lamb_total = pt.sum(lamb, dim=0)
            
            I += lamb_total
        
        return I
    
    def forward(self, seq, times):
        # seq : one hot encoded vectors of events (size N_batch x N_events x K)
        # times : times of occurences of the events (size N_batch x N_events)
        N_events = seq.shape[1]
        N_batch = seq.shape[0]
        
        # Need to initialize the cell memories
        # When nothing has occurred, the cell memories should be zero
        ct = pt.zeros(N_batch, self.hd)
        cbar = pt.zeros(N_batch, self.hd)
        ht = pt.zeros(N_batch, self.hd)
        
        lambOuts = pt.zeros(N_batch, N_events - 1, self.K)
        
        # We also need the following quantities to do the MC sampling
        # We also need the "c" values and deltas for the MC sampling
        Clows = pt.zeros(N_batch, N_events - 1, self.hD)
        Cbars = pt.zeros(N_batch, N_events - 1, self.hD)
        
        deltas = pt.zeros(N_batch, N_events - 1, 1)
        
        OutGates = pt.zeros(N_batch, N_events - 1, self.hD)
        
        # Now let's propagate through the event sequence
        # We'll go from event 0 to event N_events-1
        # at each time index, the lambda values will be predicted
        # for the next time index.
        for evInd in range(N_events - 1):
            
            x = seq[:, evInd, :]  # feature vectors from all batches at this timeStamp
            
            # Let's get all the output together first
            NNOuts = self.sigma(self.L_U(x) + self.L_V(h_t))
            
            # Now separate out the quantities
            # The first index will be for all samples in the batch
            i, f = NNOuts[:, :self.hd], NNOuts[:, self.hd:2*self.hd]
            
            iBar, fBar = NNOuts[:, 2*self.hd:3*self.hd], NNOuts[:, 3*self.hd:4*self.hd]
            
            # Remember that "z" has a factor of 2 in front of it
            z, o = 2*NNOuts[:, 4*self.hd:5*self.hd], NNOuts[:, 5*self.hd:6*self.hd]
            
            # let's use leaky_relu for delta for now
            delta = F.leaky_relu(self.D_U(x) + self.D_V(h_t))
            # delta is (N_batchx1)
            
            # Now, from these outputs, we need to construct our cell memories
            clow = f * ct + i * z
            cbar = fbar * cbar + ibar * z
            # clow and cbar are (N_batch x hD)
            
            # get the times
            tnow = times[:, evInd].view(-1, 1) # evInd-th time for all sequences in the batch
            #TODO : make sure here that 0th events have time 0 across the batch
            
            tnext = times[:, evInd + 1].view(-1, 1) # evInd+1-th time for all sequences in the batch
            
            # get c(t)
            ct = cbar + (clow - cbar)*pt.exp((tnext - tnow)*delta)
            # ct is N_batch x hD
            
            # with the c(t), we now have to determine h(t)
            # eqn 4(b) on page 4 in the paper
            ht = o * (2*self.sigma(2*ct) - 1)
            # o is N_batch x hD
            # (2*self.sigma(2*ct) - 1) is N_batchxhD
            # so ht is also N_batchxhD
            
            # ht is now (N_batch x hD)
            # Now, eqn. 4(a) linear part
            lamb_til = self.L_lamb_til(ht)
            # lamb_til is (N_batch x K)
            
            lamb = self.scale * pt.log(1 + pt.exp(lamb_til/self.scale))
            # Note : scale ("s") is K-dim vector
            # lamb_til is (N_batch x K)
            # lamb_til / s will divide each K-dim row of "lamb_til"
            # with K-dim "s" element-wise
            # which is exactly what we want
            # same goes for the multiplication with s
            
            # lamb is therefore (N_batch x K) - the predicted event rates at this timeStamp
            lambOuts[:, evInd, :] = lamb
            
            # Record the cell memories for the MC sampling
            CLows[:, evInd, :] = clow
            Cbars[:, evInd, :] = cbar
            deltas[:, evInd, :] = delta
            OutGates[:, evInd, :] = o
        
        return lambOuts, CLows, Cbars, deltas, OutGates


# In[ ]:




