#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm


# In[2]:


Nseq = 10000
I = 25 # Sequence length
K = 5 # No. of event types


# In[3]:


mu = np.random.rand(K)
alpha = np.random.rand(K, K)
delta = np.random.rand(K, K)*10+10
delta


# In[4]:


tarr = np.zeros((Nseq, I))
evArr = np.zeros((Nseq, I), dtype=int)
for seq in tqdm(range(Nseq), position=0, leave=True): # No. of sequences
    for i in range(I): # No. of timestamps in each sequence
        
        # We have to select the next event
        # First, we find the next time for each event type
        t_i_k = np.zeros(K)
        for k in range(K):
            # Find the intensity of the k^th type at i^th timestamp
            
            C = True
            t = 0. if i==0 else tarr[seq, i-1]
            # Find the supremum
            # iterate through previous time stamps
            lamb_sup_k_i = mu[k]
            for j in range(i):
                lamb_sup_k_i += alpha[k, evArr[seq, j] ]
            while C:                    
                Del = -np.log(np.random.rand())/lamb_sup_k_i
                t += Del
                lamb_k_i = mu[k]
                for j in range(i):
                    lamb_k_i += alpha[k, evArr[seq, j]] * np.exp(-delta[k, evArr[seq, j]]*
                                                                 (t-tarr[seq, j]))
                    
                # Decide to accept the time
                if lamb_k_i/lamb_sup_k_i > np.random.rand():
                    C = False
                    
            t_i_k[k] = t
        
        # Store the earliest event
        tarr[seq, i] = np.min(t_i_k)
        evArr[seq, i] = np.argmin(t_i_k)


# In[10]:


t0arr = np.zeros((Nseq, I+1))
t0arr[:, 1:] = tarr


# In[11]:


t0arr[1]


# In[12]:


np.save("Events.npy", evArr)
np.save("Times.npy", t0arr)
np.save("Mu.npy", mu)
np.save("alpha.npy", alpha)
np.save("Delta.npy", delta)


# In[13]:


evArr.shape


# In[ ]:




