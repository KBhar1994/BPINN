# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:38:19 2023

@author: Kevin Bhar
"""

import math
import random as rd
import numpy as np
import matplotlib.pyplot as plt

# pi = math.pi

# x = np.linspace(0, 2*pi, num = 100)

# y = np.sin(x)
# std1 = 0.05
# noise = np.random.normal(0, std1, size = x.shape)
# yn = y + noise 

# f = np.cos(x)
# std2 = 0.07
# noise = np.random.normal(0, std2, size = x.shape)
# fn = f + noise 

# Data = np.zeros([3,100])
# Data[0,:] = x
# Data[1,:] = yn
# Data[3,:] = fn


# plt.plot(x,y,'k--', x,yn,'r.')
# plt.show()

# filename = 'Sample_ODE_data.npy'
# np.save(filename, Data)

##

lam = 0.01
num = 500 # number of data points

x = np.linspace(-0.7, 0.7, num)

u = np.power( np.sin(6*x), 3)
std1 = 0.01
noise = np.random.normal(0, std1, size = x.shape)
un = u + noise 

f = 18 * lam * np.sin(6*x)**2 #  108 * lam * np.sin(6*x) * ( 2*np.power( np.cos(6*x),2 ) - np.power( np.sin(6*x),2 ) )
std2 = 0.01
noise = np.random.normal(0, std2, size = x.shape)
fn = f + noise 

plt.plot(x, un, 'r.', x, u, 'k--')
plt.plot(x, fn, 'b.', x, f, 'k--')
plt.ylim(-3,2)
plt.show()

sz = np.shape(x)[0]
Data = np.zeros([5,sz])
Data[0,:] = x
Data[1,:] = u
Data[2,:] = un
Data[3,:] = f
Data[4,:] = fn

# filename = 'Sample_ODE2_data_.npy'
# np.save(filename, Data)





