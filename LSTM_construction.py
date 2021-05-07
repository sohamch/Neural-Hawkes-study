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
        self.hD = hD
        
        # input to L_U will be a feature vector, which is K-dim
        self.L_U = nn.Linear(K, 7*hD) # for each of the six gates + delta
        
        # input to L_V will be h_t, which is hD dim
        self.L_V = nn.Linear(hD, 7*hD)
        
        # We need another linear layer to compute lambda_tilde
        # This layer takes hD-dimensional h(t) and returns K-dimensional vector
        self.L_lamb_til = nn.Linear(hD, K, bias=False)
        
        nn.init.normal_(self.L_U.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.L_V.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.L_U.bias, mean=0.0, std=0.01)
        nn.init.normal_(self.L_V.bias, mean=0.0, std=0.01)
        nn.init.normal_(self.L_lamb_til.weight, mean=0.0, std=0.01)
        
        # Then, to predict lambda from lambda_tilde using softplus,
        # we need scaling parameters
        # we need to make these scales a part of the
        # learnable parameter set
        self.scale = nn.Parameter(pt.ones(K, requires_grad=True))
        
        # let's work with reLU for now
        self.sigma = pt.sigmoid
    
    def MC_Loss(self, times, tMax, Clows, Cbars, deltas, OutGates, Nsamples=1000):
        
        # "times" is of dimension (N_mask x N_events+1)
        # tMax contains the maximum time of each sequence in the batch
        # Our time invterval will be between 0 to tMax for each sequence
        N_batch = times.shape[0]
        randNums = pt.rand(N_batch, Nsamples)
        trands = randNums * tMax.view(-1, 1)
        # trands will be of shape N_mask x Nsamples
        # each row of trands will correspond to Nsamples
        # random times between 0 to the end time
        # of the corresponding sequence.
        
        # Once the random time instants have been formed, we need to store
        # the intervals in which they lie
        # Our assumption here is that the input "times" is an ascending-order
        # sorted array, and that times[0] =  0
        
        t_up = pt.searchsorted(times, trands)
        # searchSorted works row-wise automatically
        # check examples on pytorch documentation website.
        # For every n^th row of t_up:
        # t_up[n, i] = idx, such that times[n, idx-1]<trands[n, i]<times[n, idx]
        
        # Using the intervals in which the sample times lie,
        # we have to find the rates
        
        I = pt.zeros(N_batch).double()

        for tInd in range(Nsamples):
            t = trands[:, tInd].view(-1,1)
            
            # Need to use cbar, clow and delta for the next time index
            # For each different sample, different times have been selected.
            # See stackoverflow 
            # stackoverflow.com/questions/58523290
            # to understand how the indices are being tracked.
            idx = t_up[:, tInd]
            tlow = times.gather(1, (idx-1).view(-1, 1))
            
            # To understand indexing multi-d tensors using different indices
            # see stackoverflow.com/questions/55628014
            clow = Clows[pt.arange(Clows.shape[0]), idx-1]
            cbar = Cbars[pt.arange(Clows.shape[0]), idx-1]
            delta = deltas[pt.arange(Clows.shape[0]), idx-1]
            o = OutGates[pt.arange(Clows.shape[0]), idx-1]
                
            
            # compute c(t)
            # Note here we use "t - tlow"
            # print(cbar.shape, clow.shape, delta.shape, t.shape, tlow.shape, (t - tlow).view(-1,1).shape)
            ct = cbar + (clow - cbar)*pt.exp(-(t - tlow)*delta)
            
            # compute h(t)
            ht = o * (2*self.sigma(2*ct) - 1)
            
            # compute lambda_k(t)
            lamb_til = self.L_lamb_til(ht)
            
            # this will contain the event intensities for all the K events
            lamb = self.scale * pt.log(1 + pt.exp(lamb_til / self.scale))
            
            # get the sum total rate of all events
            lamb_total = pt.sum(lamb, dim=1)
            # print(clow.shape, cbar.shape, delta.shape, ct.shape, ht.shape, lamb_til.shape, lamb.shape, lamb_total.shape)
            I += lamb_total*tMax/Nsamples
        
        # return trands and the indices also for testing the integral
        return pt.sum(I, dim=0)/N_batch, trands, t_up
    
    def logLoss(self, seq, mask, lambOuts):
        
        # Here seq should not be one-hot encoded, but just be
        # the index of which outcome has occurred.
        N_batch = seq.shape[0]
        N_events = seq.shape[1]
        
        loss = 0.
        for ev in range(N_events):
            # For every sequence, get the log of the intensity
            # of the event that has occured at ev^th timestamp
            lambs = lambOuts[pt.arange(N_batch), ev, seq[:, ev]]
            logLambs = pt.log(lambs[mask[:, ev]])
            
            # Then add them up
            loss += pt.sum(logLambs)
        
        # return the batch average
        return -loss/N_batch
    
    def forward(self, seq, mask, times):
        # seq : one hot encoded vectors of events (size N_mask x N_events x K)
        # mask : contains the mask for each sequence
        # times : times of occurences of the events (size N_mask x N_events)
        N_events = seq.shape[1]
        N_batch = seq.shape[0]
        
        # Need to initialize the cell memories
        # When nothing has occurred, the cell memories should be zero
        ct = pt.zeros(N_batch, self.hD).double()
        cbar = pt.zeros(N_batch, self.hD).double()
        ht = pt.zeros(N_batch, self.hD).double()
        
        lambOuts = pt.zeros(N_batch, N_events, self.K).double()
        
        # We also need the following quantities to do the MC sampling
        # We also need the "c" values and deltas for the MC sampling
        CLows = pt.zeros(N_batch, N_events, self.hD).double()
        Cbars = pt.zeros(N_batch, N_events, self.hD).double()
        
        deltas = pt.zeros(N_batch, N_events, self.hD).double()
        
        OutGates = pt.zeros(N_batch, N_events, self.hD).double()
        
        # Now let's propagate through the event sequence
        # We'll go from event 0 to event N_events-1
        # at each time index, the lambda values will be predicted
        # for the next time index.
        for evInd in range(N_events):
            
            xNext = seq[:, evInd, :][mask[:, evInd]]
            # feature vectors from all batches at this timeStamp
            # mask is applied to eliminate samples that don't have data at this time stamp
            # So, instead of N_batch events, we will select N_mask events
            # where N_mask is the no. of elements of mask[:, evInd] that
            # have "True" value
            # Clearly, if all sequences have a valid event at this time, N_mask = N_batch
            
            # Let's get all the output together first
            # from ht, use only those sequences that have a valid event (mask is not false)
            NNOuts = self.L_U(xNext) + self.L_V(ht[mask[:, evInd]])
            # Now separate out the quantities
            # The first index will be for all samples in the batch
            i, f = self.sigma(NNOuts[:, :self.hD]), self.sigma(NNOuts[:, self.hD:2*self.hD])
            
            iBar, fBar = self.sigma(NNOuts[:, 2*self.hD:3*self.hD]), self.sigma(NNOuts[:, 3*self.hD:4*self.hD])
            
            # Remember that "z" has a factor of 2 in front of it
            z, o = 2*self.sigma(NNOuts[:, 4*self.hD:5*self.hD]), self.sigma(NNOuts[:, 5*self.hD:6*self.hD])
            
            # let's use leaky_relu for delta for now
            delta = F.softplus(NNOuts[:, 6*self.hD:7*self.hD])
            # delta is (N_maskx1)
            # All the gates are N_mask x hD
            
            # Now, from these outputs, we need to construct our cell memories
            clow = f * ct[mask[:, evInd]] + i * z
            cbar[mask[:, evInd]] = fBar * cbar[mask[:, evInd]] + iBar * z
            # clow and cbar are (N_mask x hD)
            
            # get the times
            tnow = times[:, evInd][mask[:, evInd]].view(-1, 1) # evInd-th time for all sequences in the batch
            #TODO : make sure here that 0th events have time 0 across the batch
            # the mask ensures that only sequences that have a valid time at this time stamp is selected
            
            tnext = times[:, evInd + 1][mask[:, evInd]].view(-1, 1) 
            # evInd+1-th time for all sequences in the batch
            
            # get c(t) for samples with a valid event at this time
            ct[mask[:, evInd]] = cbar[mask[:, evInd]] + (clow - cbar[mask[:, evInd]])*pt.exp(-(tnext - tnow)*delta)
            # ct is N_mask x hD
            
            # with the c(t), we now have to determine h(t)
            # we record it for only samples with valid events
            # eqn 4(b) on page 4 in the paper
            ht[mask[:, evInd]] = o * (2*self.sigma(2*ct[mask[:, evInd]]) - 1)
            # o is N_mask x hD
            # (2*self.sigma(2*ct[mask[:, evInd]]) - 1) is N_maskxhD
            # so ht is also N_maskxhD
            
            # ht is now (N_mask x hD)
            # Now, eqn. 4(a) linear part
            lamb_til = self.L_lamb_til(ht[mask[:, evInd]])
            # lamb_til is (N_mask x K)
            
            lamb = self.scale * pt.log(1 + pt.exp(lamb_til/self.scale))
            # Note : scale ("s") is K-dim vector
            # lamb_til is (N_mask x K)
            # lamb_til / s will divide each K-dim row of "lamb_til"
            # with K-dim "s" element-wise
            # which is exactly what we want
            # same goes for the multiplication with s
            
            # lamb is therefore (N_mask x K) - the predicted event rates at this timeStamp
            lambOuts[:, evInd, :][mask[:, evInd]] = lamb
            
            # Record the cell memories for the MC sampling
            # mask ensures that only those sequence that have an event at this
            # timestamp are recorded - rest are kept as zero.
            CLows[:, evInd, :][mask[:, evInd]] = clow
            Cbars[:, evInd, :][mask[:, evInd]] = cbar[mask[:, evInd]]
            deltas[:, evInd, :][mask[:, evInd]] = delta
            OutGates[:, evInd, :][mask[:, evInd]] = o
        
        return lambOuts, CLows, Cbars, deltas, OutGates


# In[ ]:




