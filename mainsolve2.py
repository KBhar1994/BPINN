# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:14:29 2023

@author: Kevin Bhar
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import copy
import math
# plotting
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
# system
#from time import time
import time
import sys
import os
import gc
import pdb
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
# local import
import NN_struc
from BayesNN2 import BayesNN2
from LangD import LangD


# 0.01 d^2u/dx^2 = f
# Solution : u(x) = sin^3(6x)
# f(x) =  (108 * 0.01) sin(6*x) ( 2*cos^2(6*x) - sin^2(6*x) )
 

n_samples = 5
n_feature = 1
n_hidden_1 = 50
n_hidden_2 = 50
n_output = 1
epochs = 1000 # 20000   
noise = 1e-6
method = 'Lang'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denseFCNN = NN_struc.Net(n_feature, n_hidden_1, n_hidden_2, n_output)
bayes_nn = BayesNN2(denseFCNN, n_samples=n_samples, noise=noise).to(device)
Data = np.load('Sample_ODE2_data_.npy')
x = Data[0,:]
u = Data[1,:]
un = Data[2,:]
f = Data[3,:]
fn = Data[4,:]

# fb = np.array([fn[0], fn[-1]])
# xb = np.array([x[0], x[-1]])
fb = fn[0:-1:10]
xb1 = x[0:-1:10]
nf = len(fb)
data = torch.utils.data.TensorDataset(torch.FloatTensor(xb1), torch.FloatTensor(fb))
train_loader1 = torch.utils.data.DataLoader(data, batch_size=nf, shuffle=True)

# ub = un[0:-1:10]
# xb = x[0:-1:10]
ub = np.array([un[0], un[-1]])
xb2 = np.array([x[0], x[-1]])
nu = len(ub)
data = torch.utils.data.TensorDataset(torch.FloatTensor(xb2), torch.FloatTensor(ub))
train_loader2 = torch.utils.data.DataLoader(data, batch_size=nu, shuffle=True)

# data = torch.utils.data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(u))
# train_loader3 = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

langd = LangD(bayes_nn, n_samples, train_loader1, train_loader2)

loss = []
log_post = []
log_like_f = []
log_like_b = []
log_prior = []

print('Training starting...')
for epoch in range(epochs):
    if method == 'Lang': # (step size = 1e-7)
        mean_log_post, mean_log_like_f, mean_log_like_b, mean_log_prior = langd.train(epoch, nu, nf)
        log_post.append(mean_log_post)
        log_like_f.append(mean_log_like_f)
        log_like_b.append(mean_log_like_b)
        log_prior.append(mean_log_prior)
    elif method == 'GD':  # (step size = 1e-8)
        mean_loss = langd.train2(epoch)
        loss.append(mean_loss)
    else:
        print('Wrong method')
        break
        
    if epoch % 20 == 0:
        print('Epochs complete:', epoch, '/', epochs)
print('Training finished...')

if method == 'Lang':
    plt.plot( np.abs(log_post[100:epochs-1]) , 'k.')
    # plt.plot( np.abs(log_post[100:epochs-1]) , 'k.', label="log posterior")
    # plt.plot( np.abs(log_like_f[100:epochs-1]) , 'r.', label="log likelihood (domain)")
    # plt.plot( np.abs(log_like_b[100:epochs-1]) , 'g.', label="log likelihood (bound)")
    # plt.plot( np.abs(log_prior[100:epochs-1]) , 'b.', label="log prior")
    # plt.legend()
    plt.show()       
elif method == 'GD':
    plt.plot( np.abs(loss[100:epochs-1]) , 'k.')
    plt.show()


data = torch.utils.data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(un))
train_loader2 = torch.utils.data.DataLoader(data, batch_size=500, shuffle=False)
u_hat_all, f_hat_all = langd.compute(train_loader2)
u_mean = torch.zeros(len(u_hat_all[0]))
u_std = torch.zeros(len(u_hat_all[0]))
f_mean = torch.zeros(len(u_hat_all[0]))
f_std = torch.zeros(len(u_hat_all[0]))
for i in range(len(u_hat_all[0])):
    e = []
    d = []
    for j in range(len(u_hat_all)):
        e.append(u_hat_all[j][i])
        d.append(f_hat_all[j][i])
    u_mean[i] = torch.tensor( np.mean(e) )
    u_std[i] = torch.tensor( np.std(e) )
    f_mean[i] = torch.tensor( np.mean(d) )
    f_std[i] = torch.tensor( np.std(d) )
    

plt.plot(xb2, ub, 'ro', label="data")
plt.plot(x, u, 'b--', label="actual")
plt.plot(x, u_mean, 'k-', label="predicted")
plt.fill_between(x, u_mean - u_std, u_mean + u_std, color=[0.0,0.0,0.0], alpha=0.2)
# plt.plot(x, u_mean + u_std, 'r-', x, u_mean - u_std, 'r-')
plt.legend()
plt.show()

plt.plot(xb1, fb, 'ro', label="data")
plt.plot(x, f, 'b--', label="actual")
plt.plot(x, f_mean, 'k-', label="predicted")
plt.fill_between(x, f_mean - f_std, f_mean + f_std, color=[0.0,0.0,0.0], alpha=0.2)
# plt.plot(x, u_mean + u_std, 'r-', x, u_mean - u_std, 'r-')
plt.legend()
plt.show()





