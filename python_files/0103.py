# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 20:13:22 2018

@author: Yizhen Zhao

Student's t Distribution
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

df = 4
x = np.linspace(-5, 5, 1000)
plt.plot(x, ss.t.pdf(x, df),'r-', lw=5, alpha=0.6, label='t pdf')


df = 50
x = np.linspace(-5, 5, 1000)
plt.plot(x, ss.t.pdf(x, df),'b-.', lw=5, alpha=0.6, label='t pdf')
''''
df = 30
plt.plot(x, ss.t.pdf(x, df),'r:', lw=5, alpha=0.6, label='t pdf')
'''
plt.plot(x, ss.norm.pdf(x),'k:', lw=5, alpha=0.6, label='normal pdf')
