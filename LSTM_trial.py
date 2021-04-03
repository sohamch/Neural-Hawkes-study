#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F


# In[9]:


class CTLSTM(nn.Module):
    
    def __init__(self, K):
        super().__init__()
        
        # K : input dimension
        self.K = K
        
        # Let's make all hidden quantities K-dim for now
        # i.e, i, f, o, z and c
        # the decay rate is one-dimensional
        
        # Li = [W_i | U_i] in block form
        # The input will vector will be [k | h]
        
        # For the lower limits of c
        self.Li = nn.Linear(2*K, K)
        self.Lf = nn.Linear(2*K, K)
        
        # For the upper limits of c
        self.Libar = nn.Linear(2*K, K)
        self.Lfbar = nn.Linear(2*K, K)
        
        self.Lz = nn.Linear(2*K, K)
        self.Lo = nn.Linear(2*K, K)
        self.Ld = nn.Linear(2*K, 1) # to predict the decay parameter delta
        
        # to predict lambda_tilde from h(t) - eqn.s (3a) ad (4a)
        # This linear transformation has no bias
        self.L_lamb_til = nn.Linear(K, K, bias=False)
        
        # Then, to predict lambda from lambda_tilde using softplus,
        # we need scaling parameters
        # we need to make these scales a part of the
        # learnable parameter set
        self.scale = nn.Parameter(pt.rand(K, requires_grad=True))
        
        # let's work with reLU for now
        self.sigma = F.relu
    
    def MC_Loss(self, times, Clows, Cbars, deltas, OutGates, Nsamples=1000):
        # To compute the integral, we'll use "Nsamples" samples
        # Our time invterval will be between 0 to times[-1]
        trands = pt.rand(Nsamples)*times[-1]
        
        # Once the random time instants have been formed, we need to store
        # the intervals in which they lie
        # Our assumption here is that the input "times" is an ascending-order
        # sorted array, and that times[0] =  0
        
        t_up = pt.searchsorted(times, trands)
        # t_up[i] = idx, such that times[idx-1]<trands[i]<times[idx]
        
        # Using the intervals in which the sample times lie,
        # we have to find the rates
        
        I = torch.tensor([0])
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
        # seq : one hot encoded vectors of events (size N_events x K)
        # times : times of occurences of the events (size N_events)
        N_events = seq.shape[0]
        
        # Need to initialize the cell memories
        # When nothing has occurred, the cell memories should be zero
        ct = pt.zeros((self.K))
        cbar = pt.zeros((self.K))
        ht = pt.zeros((self.K))
        
        # Before event 0, there is no history
        # So the output records predicted intensities 
        # of events 1 through N_events
        out = pt.zeros(N_events - 1, K)
        
        # We also need the "c" values and deltas for the MC sampling
        Clows = pt.zeros(N_events - 1, K)
        Cbars = pt.zeros(N_events - 1, K)
        
        deltas = pt.zeros(N_events - 1, K)
        
        OutGates = pt.zeros(N_events - 1, K)
        
        # Now let's propagate through the event sequence
        # We'll go from event 0 to event N_events-1
        # at each time index, the lambda values will be predicted
        # for the next time index.
        for evInd in range(N_events - 1):
            
            ev = pt.cat((seq[evInd], ht)).view(-1, K)
            
            # Now let's get the parameters
            
            # eqn (5a) and (5b) in the paper on page 5
            i = self.sigma(self.Li(ev)).view(K)
            f = self.sigma(self.Lf(ev)).view(K)
            
            # footnote (3) in the paper on page 5
            ibar = self.sigma(self.Li(ev)).view(K)
            fbar = self.sigma(self.Lf(ev)).view(K)
            
            # eqn (5c) and (5d) in the paper on page 5
            z = self.sigma(self.Lz(ev)).view(K)
            o = self.sigma(self.Lo(ev)).view(K)
            
            # eqn(6c) in the paper in the paper on page 5
            delta = self.sigma(self.Ld(ev)).view(K)
            
            # Once these parameters are learned, we need
            # to do the updates to c
            
            # eqn (6a) and (6b) in the paper on page 5
            clow = f * ct + i * z
            cbar = fbar * cbar + ibar * z
            
            # once clow, cbar and delta are found, we need to compute
            # equation (7) in the paper
            tnow = times[evInd] if evInd > 0 else 0
            tnext = times[evInd + 1]
            ct = cbar + (clow - cbar)*pt.exp((tnext - tnow)*delta)
            
            # with the c(t), we now have to determine h(t)
            # eqn 4(b) on page 4 in the paper
            ht = o * (2*self.sigma(2*ct) - 1)
            
            # Now, eqn. 4(a) linear part
            lamb_til = self.L_lamb_til(ht.view(-1, K)).view(K)
            
            # Then the softplus part
            # this will contain the event intensities for all the K events
            # in the next step.
            lamb = s * pt.log(1 + pt.exp(lamb_til / s))
            
            # Then record the event intensities for all the events
            out[evInd, :] = lamb
            
            # Record the cell memories for the MC sampling
            CLows[evInd, :] = clow
            Cbars[evInd, :] = cbar
            deltas[evInd, :] = delta
            OutGates[evInd, :] = o


# In[ ]:




