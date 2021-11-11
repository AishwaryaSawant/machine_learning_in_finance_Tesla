# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 21:33:30 2018
Acceptance - Rejection Method for random number generation
@author: Yizhen Zhao
 Rejection method for random number generation  
 ===============================================  
 Uses the rejection method for generating random numbers derived from an arbitrary   
 probability distribution. For reference, see Bevington's book, page 84. Based on  
 rejection*.py.  
   
 Usage:  
 >>> randomvariate(P,N,xmin,xmax)  
  where  
  P : probability distribution function from which you want to generate random numbers  
  N : desired number of random values  
  xmin,xmax : range of random numbers desired  
    
 Returns:   
  the sequence (ran,ntrials) where  
   ran : array of shape N with the random variates that follow the input P  
   ntrials : number of trials the code needed to achieve N  
   
 Here is the algorithm:  
 - generate x' in the desired range  
 - generate y' between Pmin and Pmax (Pmax is the maximal value of your pdf)  
 - if y'<P(x') accept x', otherwise reject  
 - repeat until desired number is achieved  
  """  
  # Calculates the minimal and maximum values of the PDF in the desired  
  # interval. The rejection method needs these values in order to work  
  # properly.  
  
import numpy as np

c = 2
n = 100 
x = np.zeros(n)
xy = np.zeros(n)
rej = np.zeros(n)
rejy = np.zeros(n) 
irv = 0
irej = 0
while irv <= n-1:
        y = np.random.uniform(0,1,1) 
        u = np.random.uniform(0,1,1) 
        if u <= 2*y/c:
           x[irv] = y
           xy[irv] = u*c
           irv = irv+1
        else:
           rej[irej] = y
           rejy[irej] = u*c
           irej = irej + 1