# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:36:04 2023

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
from time import time
import sys
import os
import gc
import pdb
import subprocess # Call the command line
from subprocess import call
# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.ticker as ticker
# local import

class Lang_sampler(torch.optim.Optimizer):
      
    # def __init__(self, params, lr=1e-4):
    def __init__(self, bayes_nn, params, lr=1e-6):
        super(Lang_sampler, self).__init__(params, defaults={'lr': lr})
        # super(Lang_sampler, self).__init__(params, defaults={'lr': lr})
        self.bayes_nn = bayes_nn
        self.params = params
        self.lr = lr
      
    # Step Method
    # def step(self, epoch, i):
    #     parameters = self.bayes_nn.features.parameters()
    #     co = 0
    #     P1 = []
    #     for p in parameters:
    #         co += 1
    #         P1.append(p)
    #         alpha = self.lr
    #         grad = p.grad
    #         if grad == "Nan":
    #             print(epoch)
    #             print(i)
    #             break
    #         v = torch.Tensor( np.random.normal(0, 1, size = p.data.size() ) )
    #         p.data += alpha * grad + np.sqrt(2*alpha) * v
    #         # p.data += alpha * grad
    #         # d_p = alpha*grad + v
    #         # p.data = p.data.add_(d_p)
        
    #     P2 = []
    #     parameters = self.bayes_nn.features.parameters()
    #     for p in parameters:
    #         P2.append(p)
            
    #     for i in range(co):
    #         print(P1[i])
    #         print(P2[i])
    #         print("\n")
    
    def step(self, epoch, i, method):
        params = self.bayes_nn.features.parameters()
        alpha = self.lr # / (epoch+1)**0.001
        for p in params:
            grad = p.grad
            if method == 'Lang':
                v = torch.Tensor( np.random.normal(0, 1, size = p.data.size() ) )
                p.data += alpha*grad + np.sqrt(2*alpha)*v
            elif method == 'GD':
                p.data += - alpha*grad
            else:
                print('Wrong method')
                break
            
            
            
            
            
            
            
            