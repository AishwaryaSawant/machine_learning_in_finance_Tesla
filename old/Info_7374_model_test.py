#!/usr/bin/env python
# coding: utf-8

# ## GARCH MODEL

# In[14]:


import numpy as np
import pandas as pd
# import pyflux as pf
from scipy.optimize import minimize
from pandas_datareader import data as pdr
from datetime import datetime
import matplotlib.pyplot as plt

def GARCH(param, *args):
 "Initialize Params:"
 mu = param[0]
 omega = param[1]
 alpha = param[2]
 beta = param[3]
 T = Y.shape[0]
 GARCH_Dens = np.zeros(T) 
 sigma2 = np.zeros(T)   
 F = np.zeros(T)   
 v = np.zeros(T)   
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    F[t] = Y[t] - mu-np.sqrt(sigma2[t])*np.random.normal(0,1,1)
    v[t] = sigma2[t]
    GARCH_Dens[t] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(v[t])+                    (1/2)*(F[t]/v[t])     
    Likelihood = np.sum(GARCH_Dens[1:-1])  
    return Likelihood


def GARCH_PROD(params, Y0, T):
 mu = params[0]
 omega = params[1]
 alpha = params[2]
 beta = params[3]
 Y = np.zeros(T)  
 sigma2 = np.zeros(T)
 Y[0] = Y0
 sigma2[0] = 0.003
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    Y[t] = mu+np.sqrt(sigma2[t])*np.random.normal(0,1,1)    
 return Y    

# 1. Simulated Data
# T = 1000
# mu = 35;
# sig = 5;
# Y = np.random.normal(mu,sig,T);
# 2. Real Data
TSLA = pdr.get_data_yahoo('TSLA',datetime(2021,1,1), datetime(2021,9,30))
# Y = TSLA['Adj Close'].values
Y = np.diff(np.log(TSLA['Adj Close'].values))
T = Y.shape[0]

param0 = np.array([0, 0.003, 0.3, 0.3])
param_star = minimize(GARCH, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
Y_GARCH = GARCH_PROD(param_star.x, Y[0], T)
timevec = np.linspace(1,T,T)


FORD = pdr.get_data_yahoo('F',datetime(2021,1,1), datetime(2021,9,30))
# Y = TSLA['Adj Close'].values
Y2 = np.diff(np.log(FORD['Adj Close'].values))
T2 = Y2.shape[0]

param0 = np.array([0, 0.003, 0.3, 0.3])
param_star = minimize(GARCH, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
Y_GARCH2 = GARCH_PROD(param_star.x, Y2[0], T2)
timevec2 = np.linspace(1,T2,T2)







plt.figure(figsize=(24,12))
#plt.plot(timevec, Y,'b',timevec, Y_GARCH,'r:')
#plt.plot(timevec2, Y2,'b',timevec2, Y_GARCH2,'r:')


plt.plot(timevec, Y,'r:')
plt.plot(timevec2, Y2,'g')



#plt.plot(timevec, Y_GARCH,'r')
#plt.plot(timevec2, Y_GARCH2,'g')


plt.title('Comparison- TESLA vs FORD')
plt.legend(['TESLA','FORD']);
plt.savefig('FORD_V_TESLA_GARCH.jpeg')



# ## GARCH-t MODEL
# 

# In[33]:


import pandas as pd
import numpy as np
import scipy.special as ss
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def GARCH_t(Y):
 "Initialize Params:"
 mu = param0[0]
 omega = param0[1]
 alpha = param0[2]
 beta = param0[3]
 nv = param0[4]
 
 T = Y.shape[0]
 GARCH_t = np.zeros(T) 
 sigma2 = np.zeros(T)   
 F = np.zeros(T)   
 v = np.zeros(T)   
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    F[t] = Y[t] - mu-np.sqrt(sigma2[t])*np.random.standard_t(nv,1)
    v[t] = sigma2[t]
    GARCH_t[t] = np.log(ss.gamma((nv+1)/2))-np.log(np.sqrt(nv*np.pi))-                    np.log(ss.gamma(nv/2))-((nv+1)/2)*np.log(1+((F[t]**2)/v[t])/nv)     
    Likelihood = np.sum(GARCH_t[1:-1])  
    return Likelihood


def GARCH_PROD_t(params, Y0, T):
 mu = params[0]
 omega = params[1]
 alpha = params[2]
 beta = params[3]
 nv = params[4]
 Y = np.zeros(T)  
 sigma2 = np.zeros(T)
 Y[0] = Y0
 sigma2[0] = 0.003
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    Y[t] = mu+np.sqrt(sigma2[t])*np.random.standard_t(nv,1) 
 return Y    


#T = 1000
#mu = 35;
#sig = 5;
#Y = np.random.normal(mu,sig,T);
TSLA = pdr.get_data_yahoo('TSLA', datetime(2021,1,1), datetime(2021,9,30))
# Y = TSLA['Adj Close'].values
Y = np.diff(np.log(TSLA['Adj Close'].values))
T = Y.shape[0]
param0 = np.array([0, 0.003, 0.3, 0.3, 30])
param_star = minimize(GARCH_t, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
Y_GARCH_t = GARCH_PROD_t(param_star.x, Y[0], T)
timevec = np.linspace(1,T,T)

FORD = pdr.get_data_yahoo('F', datetime(2021,1,1), datetime(2021,9,30))
# Y = TSLA['Adj Close'].values
Y2 = np.diff(np.log(FORD['Adj Close'].values))
T2 = Y2.shape[0]
param0 = np.array([0, 0.003, 0.3, 0.3, 30])
param_star = minimize(GARCH_t, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
Y_GARCH_t2 = GARCH_PROD_t(param_star.x, Y2[0], T2)
timevec2 = np.linspace(1,T2,T2)







plt.figure(figsize=(24,12))
#plt.plot(timevec, Y,'b',timevec, Y_GARCH_t,'r:')
#plt.plot(timevec2, Y2,'b',timevec2, Y_GARCH_t2,'r:')

plt.plot(timevec, Y,'r')
plt.plot(timevec2, Y2,'b',alpha=0.7)


plt.savefig('GARCH_t_model_TESLA_V_FORD.jpeg')



# ## Based on 0209

# In[10]:


import numpy as np
import pandas as pd
import pyflux as pf
from scipy.optimize import minimize
from pandas_datareader import data as pdr
from datetime import datetime
import matplotlib.pyplot as plt
yf.pdr_override()

def GARCH(param, *args):
 "Initialize Params:"
 mu = param[0]
 omega = param[1]
 alpha = param[2]
 beta = param[3]
 T = Y.shape[0]
 GARCH_Dens = np.zeros(T) 
 sigma2 = np.zeros(T)   
 F = np.zeros(T)   
 v = np.zeros(T)   
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    F[t] = Y[t] - mu-np.sqrt(sigma2[t])*np.random.normal(0,1,1)
    v[t] = sigma2[t]
    GARCH_Dens[t] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(v[t])+                    (1/2)*(F[t]/v[t])     
    Likelihood = np.sum(GARCH_Dens[1:-1])  
    return Likelihood


def GARCH_PROD(params, Y0, T):
 mu = params[0]
 omega = params[1]
 alpha = params[2]
 beta = params[3]
 Y = np.zeros(T)  
 sigma2 = np.zeros(T)
 Y[0] = Y0
 sigma2[0] = 0.0001
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    Y[t] = mu+np.sqrt(sigma2[t])*np.random.normal(0,1,1)    
 return Y    

TSLA = pdr.get_data_yahoo('TSLA', datetime(2021,1,1), datetime(2021,8,31))
# Y = np.diff(np.log(TSLA['Adj Close'].values))
Y = TSLA['Adj Close'].values 
T = Y.shape[0];
param0 = np.array([np.mean(Y), np.std(Y)/2, 0.5, 0.5])
param_star = minimize(GARCH, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
Y_GARCH = GARCH_PROD(param_star.x, Y[0], T)
timevec = np.linspace(1,T,T)
plt.figure(figsize=(15,10))
plt.plot(timevec, Y,'b',timevec, Y_GARCH,'r')


# ## Kalman Filter: Preliminary (Toy Model)
# 

# In[22]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
 
def Kalman_Filter(param, *args):
 S = Y.shape[0]
 S = S + 1
 "Initialize Params:"
 Z = param[0]
 T = param[1]
 H = param[2]
 Q = param[3]
 "Kalman Filter Starts:"
 u_predict = np.zeros(S)
 u_update = np.zeros(S)
 P_predict = np.zeros(S)
 P_update = np.zeros(S)
 v = np.zeros(S)
 F = np.zeros(S)
 KF_Dens = np.zeros(S)
 for s in range(1,S):
  if s == 1: 
    P_update[s] = 1000
    P_predict[s] =  T*P_update[1]*np.transpose(T)+Q    
  else: 
    F[s] = Z*P_predict[s-1]*np.transpose(Z)+H 
    v[s]=Y[s-1]-Z*u_predict[s-1]   
    u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
    u_predict[s] = T*u_update[s]; 
    P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1];
    P_predict[s] = T*P_update[s]*np.transpose(T)+Q
    KF_Dens[s] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(abs(F[s]))+(1/2)*np.transpose(v[s])*(1/F[s])*v[s]      
    
    Likelihood = sum(KF_Dens[1:-1]) # a loss function
    
    return Likelihood
          
def Kalman_Smoother(params, Y, *args):
 S = Y.shape[0]
 S = S + 1
 "Initialize Params:"
 Z = params[0]
 T = params[1]
 H = params[2]
 Q = params[3]
 "Kalman Filter Starts:"
 u_predict = np.zeros(S)
 u_update = np.zeros(S)
 P_predict = np.zeros(S)
 P_update = np.zeros(S)
 v = np.zeros(S)
 F = np.zeros(S)
 for s in range(1,S):
   if s == 1: 
    P_update[s] = 100
    P_predict[s] =  T*P_update[1]*np.transpose(T)+Q    
   else: 
    F[s] = Z*P_predict[s-1]*np.transpose(Z)+H 
    v[s]=Y[s-1]-Z*u_predict[s-1]   
    u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
    u_predict[s] = T*u_update[s]; 
    P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1];
    P_predict[s] = T*P_update[s]*np.transpose(T)+Q
    
    u_smooth = np.zeros(S)
    P_smooth = np.zeros(S)
    u_smooth[S-1] = u_update[S-1]
    P_smooth[S-1] = P_update[S-1]    
 for  t in range(S-1,0,-1):
        u_smooth[t-1] = u_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(u_smooth[t]-T*u_update[t])
        P_smooth[t-1] = P_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(P_smooth[t]-P_predict[t])/P_predict[t]*T*P_update[t]
 u_smooth = u_smooth[1:-1]
 return u_smooth

#Z:coeff attached to latent component
#Z=1 
#Optimize T value in future --> Autoregressive  component 

start_date = datetime(2021,1,1)
end_date = datetime(2021,8,30)
TSLA = yf.download('TSLA',start_date ,end_date)
Y = TSLA['Adj Close'].values
Y = np.diff(np.log(TSLA['Adj Close'].values))
T = Y.size;

param0 = np.array([0.9, 0.2, np.std(Y), np.std(Y)])
param_star = minimize(Kalman_Filter, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
u = Kalman_Smoother(param_star.x,Y)
timevec = np.linspace(1,T-1,T-1)

FORD = yf.download('F',start_date ,end_date)
Y2 = FORD['Adj Close'].values
Y2 = np.diff(np.log(FORD['Adj Close'].values))
T2 = Y2.size;
param0 = np.array([0.9, 0.2, np.std(Y2), np.std(Y2)])
param_star = minimize(Kalman_Filter, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
u2 = Kalman_Smoother(param_star.x,Y2)
timevec2 = np.linspace(1,T2-1,T2-1)


plt.figure(figsize=(24,12))
#plt.plot(timevec, u,'r',timevec, Y[0:-1],'r:')
#plt.plot(timevec, u2,'g',timevec, Y2[0:-1],'g:')
plt.plot(timevec, u,'r',alpha=0.6);
plt.plot(timevec2, u2,'g');
plt.title('KALMAN FILTER - TESLA vs FORD')
plt.legend(['TESLA','FORD']);
plt.savefig('Tesla_vs_Ford_KALMAN.jpeg')


# In[12]:


help(plt.legend)


# 

# In[5]:


get_ipython().system('pip install yfinance')


# In[8]:


get_ipython().system('pip install pyflux')


# In[11]:







plt.figure(figsize=(24,12))
#plt.plot(timevec, Y,'b',timevec, Y_GARCH,'r:')
#plt.plot(timevec2, Y2,'b',timevec2, Y_GARCH2,'r:')


plt.plot(timevec, Y_GARCH,'r:')
plt.plot(timevec2, Y_GARCH2,'b')


plt.title('GARCH PREDICTED- TESLA vs FORD')
plt.legend(['TESLA','FORD']);
plt.savefig('FORD_V_TESLA_GARCH.jpeg')


# In[ ]:




