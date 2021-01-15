import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from TMM import *
from tqdm import tqdm
from net import Generator, GeneratorNF, ResGenerator, GeneratorMM

class GLOnet():
    def __init__(self, params):
        # GPU 
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
        # construct
        if params.net == 'NF':
            self.generator = GeneratorNF(params)
        elif params.net == 'Res':
            self.generator = ResGenerator(params)
        elif params.net == 'MM':
            self.generator = GeneratorMM(params)
        else:
            self.generator = Generator(params)
        
        if self.cuda: 
            self.generator.cuda()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=params.lr, betas = (params.beta1, params.beta2), weight_decay = params.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = params.step_size, gamma = params.step_size)
        
        # training parameters
        self.noise_dim = params.noise_dim
        self.numIter = params.numIter
        self.batch_size = params.batch_size
        self.sigma = params.sigma
        self.alpha_sup = params.alpha_sup
        self.iter0 = 0
        self.alpha = 0.1
    
        # simulation parameters
        self.user_define = params.user_define
        if params.user_define:
            self.n_database = params.n_database
        else:
            self.materials = params.materials
            self.matdatabase = params.matdatabase

        self.n_bot = params.n_bot.type(self.dtype)  # number of frequencies or 1
        self.n_top = params.n_top.type(self.dtype)  # number of frequencies or 1
        self.k = params.k.type(self.dtype)  # number of frequencies
        self.theta = params.theta.type(self.dtype) # number of angles
        self.pol = params.pol # str of pol
        self.target_reflection = params.target_reflection.type(self.dtype) 
        # 1 x number of frequencies x number of angles x (number of pol or 1)
        
        # tranining history
        self.loss_training = []
        self.refractive_indices_training = []
        self.thicknesses_training = []
        
        
    def train(self):
        self.generator.train()
            
        # training loop
        with tqdm(total=self.numIter) as t:
            it = self.iter0  
            while True:
                it +=1 

                # normalized iteration number
                normIter = it / self.numIter

                # discretizaton coeff.
                self.update_alpha(normIter)
                
                # terminate the loop
                if it > self.numIter:
                    return 

                # sample z
                z = self.sample_z(self.batch_size)

                # generate a batch of iamges
                thicknesses, refractive_indices, _ = self.generator(z, self.alpha)

                # calculate efficiencies and gradients using EM solver
                reflection = TMM_solver(thicknesses, refractive_indices, self.n_bot, self.n_top, self.k, self.theta, self.pol)
               
                # free optimizer buffer 
                self.optimizer.zero_grad()

                # construct the loss 
                g_loss = self.global_loss_function(reflection)
                
                
                # record history
                self.record_history(g_loss, thicknesses, refractive_indices)
                
                # train the generator
                g_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # update progress bar
                t.update()
    
    def evaluate(self, num_devices, kvector = None, inc_angles = None, pol = None, grayscale=True):
        if kvector is None:
            kvector = self.k
        if inc_angles is None:
            inc_angles = self.theta
        if pol is None:
            pol = self.pol            

        self.generator.eval()
        z = self.sample_z(num_devices)
        thicknesses, refractive_indices, P = self.generator(z, self.alpha)
        result_mat = torch.argmax(P, dim=2).detach() # batch size x number of layer

        if not grayscale:
            if self.user_define:
                n_database = self.n_database # do not support dispersion
            else:
                n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, True).unsqueeze(0).unsqueeze(0).type(self.dtype)
            
            one_hot = torch.eye(len(self.materials)).type(self.dtype)
            ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database, dim=2)
        else:
            if self.user_define:
                ref_idx = refractive_indices
            else:
                n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, True).unsqueeze(0).unsqueeze(0).type(self.dtype)
                ref_idx = torch.sum(P.unsqueeze(-1) * n_database, dim=2)

        reflection = TMM_solver(thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
        return (thicknesses, ref_idx, result_mat, reflection)
    
    def _TMM_solver(self, thicknesses, result_mat, kvector = None, inc_angles = None, pol = None):
        if kvector is None:
            kvector = self.k
        if inc_angles is None:
            inc_angles = self.theta
        if pol is None:
            pol = self.pol  
        n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, True).unsqueeze(0).unsqueeze(0).type(self.dtype)
        one_hot = torch.eye(len(self.materials)).type(self.dtype)
        ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database, dim=2)
        reflection = TMM_solver(thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
        return reflection
        
    def update_alpha(self, normIter):
        self.alpha = round(normIter/0.05) * self.alpha_sup + 1.
        
    def sample_z(self, batch_size):
        return (torch.randn(batch_size, self.noise_dim, requires_grad=True)).type(self.dtype)
    
    def global_loss_function(self, reflection):
        return -torch.mean(torch.exp(-torch.mean(torch.pow(reflection - self.target_reflection, 2), dim=(1,2,3))/self.sigma))

    def global_loss_function_robust(self, reflection, thicknesses):
        metric = torch.mean(torch.pow(reflection - self.target_reflection, 2), dim=(1,2,3))
        dmdt = torch.autograd.grad(metric.mean(), thicknesses, create_graph=True)
        return -torch.mean(torch.exp((-metric - self.robust_coeff *torch.mean(torch.abs(dmdt[0]), dim=1))/self.sigma))

    def record_history(self, loss, thicknesses, refractive_indices):
        self.loss_training.append(loss.detach())
        self.thicknesses_training.append(thicknesses.mean().detach())
        self.refractive_indices_training.append(refractive_indices.mean().detach())
        
    def viz_training(self):
        plt.figure(figsize = (20, 5))
        plt.subplot(131)
        plt.plot(self.loss_training)
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Iterations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        