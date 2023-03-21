# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:21:00 2023

@author: Kevin Bhar
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
from args import args, device

## Import local modules (Prof.JX-W's python code)
# RWOF_dir = os.path.expanduser("/home/luning/Documents/utility/pythonLib")
# RWOF_dir_1 = os.path.expanduser("/home/luning/Documents/utility/pythonLib/python_openFoam")
# sys.path.append(RWOF_dir)
# sys.path.append(RWOF_dir_1)
from Lang_sampler import Lang_sampler

class LangD(object):

    def __init__(self, bayes_nn, n_samples, train_loader1, train_loader2):
        self.bayes_nn = bayes_nn
        self.train_loader1 = train_loader1
        self.train_loader2 = train_loader2
        # self.train_loader3 = train_loader3
        self.n_samples = n_samples
        self.lr = 1e-8 # 1e-8
        self.optimizers = self._optimizers_schedulers()


    def _optimizers_schedulers(self):
        optimizers = []
        for i in range(self.n_samples):
            parameters = self.bayes_nn[i].features.parameters()
            lr = self.lr
            optimizer_i = Lang_sampler(self.bayes_nn[i], parameters, lr=lr)
            # optimizer_i = Lang_sampler(parameters, lr=lr)
            optimizers.append(optimizer_i)
        return optimizers
    
    def cal_prior(self, i):
        params = self.bayes_nn[i].features.parameters()
        norm = 0
        lamb = 1
        for p in params:
            norm += - lamb/2 *  torch.Tensor.norm(p.data,2)**2
        return norm
        
    def cal_likelihood_f(self, i, x, f, ntrain):
        x = torch.FloatTensor(x).to(device)
        f = torch.FloatTensor(f).to(device)
        f_d = f.detach()
        # x.requires_grad = True
        pi = np.pi
        u_hat = self.bayes_nn[i].forward(x)
        # u_hat = x ** 2
        u_hat_x = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(u_hat), create_graph = True, retain_graph = True, only_inputs=True)[0]
        
        # H = torch.zeros(u_hat_x.shape)
        # for j in range(x.shape[0]):
        #     H[j] = torch.autograd.functional.hessian(self.bayes_nn[i], x[0])
        # u_hat_xx = torch.reshape(H, u_hat_x.shape)
        
        # u_hat_xx = torch.zeros(u_hat.shape)
        # for j in range(u_hat.shape[0]):
        #     u_hat_xx[j] = torch.autograd.grad(u_hat_x[j], x[j], grad_outputs=torch.ones_like(x[j]), only_inputs=True)[0]
        
        u_hat_xx = torch.autograd.grad(u_hat_x, x, grad_outputs=torch.ones_like(u_hat), only_inputs=True)[0]
        
        f_hat = 0.01 * u_hat_xx
        log_like = - 0.5*( (f_d - f_hat)/0.01 )**2 - 0.5*np.log(2*pi*0.01**2)
        return log_like
        
    def cal_likelihood_b(self, i, x, u, ntrain):
        x = torch.FloatTensor(x).to(device)
        u = torch.FloatTensor(u).to(device)
        u_d = u.detach()
        # x.requires_grad = True
        pi = np.pi
        u_hat = self.bayes_nn[i].forward(x)
        u_hat_d = u_hat.detach()
        # likelihood = 1/ np.sqrt(2*pi*0.1**2) * np.exp( -  np.power( (u - u_hat_d), 2 ) / (2*0.1**2) )
        # log_like = np.log(likelihood)
        log_like = - 0.5*( (u_d - u_hat)/0.01 )**2 - 0.5*np.log(2*pi*0.01**2)
        return log_like
    
    
    def compute2(self, i, x):
        x = torch.FloatTensor(x).to(device)
        u_hat = self.bayes_nn[i].forward(x)
        u_hat = u_hat.detach()
        return u_hat
    
    
    def train(self, epoch, nu, nf):
        
        self.bayes_nn.train() # prepares the model for training (but does NOT actually train the model)
        
        log_like_f = 0
        log_like_b = 0
        log_prior = 0
        log_post = 0
        Log_post = []
        Log_like_f = []
        Log_like_b = []
        Log_prior = []
        
        self.bayes_nn.zero_grad() # zero the parameter gradients
        
        for i in range(self.n_samples):
            
            # log_prior = self.bayes_nn.cal_prior(i)
            log_prior = self.cal_prior(i)
            
    
            for batch_id, (x,y) in enumerate(self.train_loader1):
                x = torch.reshape( x, (x.size()[0],1) )
                y = torch.reshape( y, (y.size()[0],1) )
                x.requires_grad = True
                
                log_like_f_all = self.cal_likelihood_f(i, x, y, nf)
                log_like_f = torch.sum(log_like_f_all) # sum of all log likelihoods
                
            for batch_id, (x,y) in enumerate(self.train_loader2):
                x = torch.reshape( x, (x.size()[0],1) )
                y = torch.reshape( y, (y.size()[0],1) )
                x.requires_grad = True
                
                # log_like_b_all = self.bayes_nn.cal_likelihood_b(i, x, y, 16) # individual log likelihoods for all data in the batch (stored as an element)
                log_like_b_all = self.cal_likelihood_b(i, x, y, nu)
                log_like_b = torch.sum(log_like_b_all) # sum of all log likelihoods
                
            # log_post = (nf/(nf+nu)) * log_like_f + (nu/(nf+nu)) * log_like_b + log_prior
            log_post = log_like_f + log_like_b + log_prior
            self.optimizers[i].zero_grad()
            log_post.backward()
            log_post_d = log_post.detach()
            log_like_f_d = log_like_f.detach()
            log_like_b_d = log_like_b.detach()
            log_prior_d = log_prior.detach()
            Log_post.append(log_post_d)
            Log_like_f.append(log_like_f_d)
            Log_like_b.append(log_like_b_d)
            Log_prior.append(log_prior_d)
            self.optimizers[i].step(epoch, i, 'Lang')
            
        mean_log_post = np.average(Log_post)
        mean_log_like_f = np.average(Log_like_f)
        mean_log_like_b = np.average(Log_like_b)
        mean_log_prior = np.average(Log_prior)
        return mean_log_post, mean_log_like_f, mean_log_like_b, mean_log_prior
    
    def train2(self, epoch):
        
        self.bayes_nn.train()
        
        loss1 = 0
        loss2 = 0
        loss = 0
        Loss = []
        
        for i in range(self.n_samples):
            
            for batch_id, (x,y) in enumerate(self.train_loader1):
                x = torch.reshape( x, (x.size()[0],1) )
                y = torch.reshape( y, (y.size()[0],1) )
                y_d = y.detach()
                x.requires_grad = True
                u_hat = self.bayes_nn[i].forward(x)
                u_hat_x = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(x),create_graph = True, only_inputs=True)[0]
                u_hat_xx = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
                f_hat = 0.01 * u_hat_xx
                L1 = nn.MSELoss(reduction = 'sum')
                loss1 = L1(y_d, f_hat)
            
            for batch_id, (x,y) in enumerate(self.train_loader2):
                x = torch.reshape( x, (x.size()[0],1) )
                y = torch.reshape( y, (y.size()[0],1) )
                y_d = y.detach()
                x.requires_grad = True
                u_hat = self.bayes_nn[i].forward(x)
                L2 = nn.MSELoss(reduction = 'sum')
                loss2 = L2(y_d, u_hat)
            
            loss = loss1 + loss2
            self.optimizers[i].zero_grad()
            loss.backward()
            loss_d = loss.detach()
            Loss.append(loss_d)
            self.optimizers[i].step(epoch, i, 'GD')
            
        mean_loss = np.average(Loss)
        return mean_loss
        
    
    def compute(self, test_loader):
        u_hat_all = []
        f_hat_all = []
        for i in range(self.n_samples):
            
            for batch_id, (x,y) in enumerate(test_loader):
                x = torch.reshape( x, (x.size()[0],1) )
                x.requires_grad = True
                # u_hat, f_hat = self.bayes_nn.compute2(i,x)
                u_hat = self.bayes_nn[i].forward(x)
                u_hat_x = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(x),create_graph = True, only_inputs=True)[0]
                u_hat_xx = torch.autograd.grad(u_hat_x, x, grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
                f_hat = 0.01 * u_hat_xx
            
            u_hat_all.append(u_hat.detach())
            f_hat_all.append(f_hat.detach())
        return u_hat_all, f_hat_all
    
    # def compute(self, x):
    #     x = torch.tensor(x)
    #     x = torch.reshape( x, (x.size()[0],1) )
    #     u_hat = []
    #     u_mean = torch.zeros(x.shape)
    #     for i in range(self.n_samples):
    #         u_hat = self.bayes_nn[i].forward(x)
    #         u_mean += u_hat
    #     u_mean = u_mean/self.n_samples
    #     return u_mean
        
        
    
        
        
        
        