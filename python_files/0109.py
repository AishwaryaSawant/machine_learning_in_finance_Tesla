# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:34:42 2019
% Metropolis Hastings Sampler
@author: Yizhen Zhao
-----------------------------------------------------------------
Generate a random variable from a standard Cauchy distribution.
-----------------------------------------------------------------
We now generate samples in the chain.
Generate 1000 samples in the chain.
Set up the constants.
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n = 1000
sig = 2
x = np.zeros(n)

" generate the starting point"
x[0] = np.random.normal(0,1,1)

for i in range(1,n):
    """
    generate a candidate from the proposal distribution
    which is the normal in this case. This will be a
    normal with mean given by the previous value in the
    chain and standard deviation of 'sig'
    """
    y = x[i-1]+sig*np.random.normal(0,1,1)
    "generate a uniform for comparison"
    u = np.random.uniform(0,1,1)
    alpha = np.min([1, ss.cauchy.pdf(y)*ss.norm.pdf(x[i-1],y,sig)/ \
                    (ss.cauchy.pdf(x[i-1])*ss.norm.pdf(y,x[i-1],sig))])
    if u <= alpha:
       x[i] = y
    else:
       x[i] = x[i-1]

plt.plot(x)
plt.show()
