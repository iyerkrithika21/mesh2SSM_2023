import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

         
        self.args = args
        self.latent_dim = args.latent_dim
        input_dim = 3
        self.num_points = args.num_points
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 128, 1)
        self.conv3 = nn.Conv1d(128, 1, 1)
    
        self.fc1_m = nn.Linear(self.num_points, 64)
        self.fc3_m = nn.Linear(64, self.latent_dim)
        self.fc1_v = nn.Linear(self.num_points, 64)
        self.fc3_v = nn.Linear(64, self.latent_dim)
        
        
        self.fc1_z = nn.Linear(self.latent_dim, 128)
        self.fc2_z = nn.Linear(128, 256)
        self.fc3_z = nn.Linear(256, self.num_points*3)

    def encoder(self, x):
        
        x = x.transpose(1, 2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        x = x.view(-1, self.num_points)
        
       

        m = F.leaky_relu(self.fc1_m(x))
        m = self.fc3_m(m)
        v = F.leaky_relu(self.fc1_v(x))
        v = self.fc3_v(v)

        return m, v

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self,z):

        z = F.leaky_relu(self.fc1_z(z))
        z = F.leaky_relu(self.fc2_z(z))
        z = (self.fc3_z(z))

        return z

    def sample(self,samples_size):
        z = torch.randn(samples_size,self.latent_dim).double()

        z = z.to(self.args.device)

        samples = self.decoder(z).view(-1,3,self.num_points)
        
        samples = samples.permute(0, 2, 1)
        return samples

    def forward(self,x):
        
        self.num_points = x.shape[1]
        mu,logvar = self.encoder(x)
        z = self.reparameterize(mu,logvar)
        
        x_recon = self.decoder(z).view(-1,3,self.num_points)
        x_recon = x_recon.permute(0, 2, 1)
        return mu,logvar,z,x_recon

    