import sys
try:
    sys.path.remove('/home/huy/Desktop/stuff/spinningup')
    sys.path.append("/home/huy/Desktop/github/RL-Projects/model/varibad_for_game")
except:
    pass

import numpy as np
import scipy.signal
# from gym.spaces import Box, Discrete
from gym.spaces import Box, Discrete


import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam,SGD
import gym
import time
import torch
import torch.nn as nn

from config.decoder_cfg import decoder_CFG

class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function, device):
        super(FeatureExtractor, self).__init__()
        self.device = device
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(self.device)

def mlp(sizes, activation , output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class reward_network(nn.Module):
    def __init__(self,CFG,obs_size,action_size):
        super(reward_network, self).__init__()
        self.CFG = CFG
        self.fc_s = FeatureExtractor(obs_size,self.CFG.obs_embed_dim, F.relu,self.CFG.device)
        self.fc_s_next = FeatureExtractor(obs_size,self.CFG.obs_embed_dim, F.relu,self.CFG.device)
        self.fc_a = FeatureExtractor(action_size,self.CFG.action_embed_dim, F.relu,self.CFG.device)
        self.input_dim = self.CFG.obs_embed_dim * 2 +  self.CFG.action_embed_dim + \
            self.CFG.latent_dim
        self.mlp = mlp([self.input_dim] + self.CFG.decoder_hidden,self.CFG.decoder_activation)

        input_dim = self.CFG.decoder_hidden[-1]
        self.fc_mu = nn.Linear(input_dim, 1)
        self.fc_logvar = nn.Linear(input_dim, 1)

    def gaussian_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, s, a, s_next, latent):
        s = self.fc_s(s)
        a = self.fc_a(a)
        s_next = self.fc_s_next(s_next)
        x = torch.cat((s,a,s_next,latent),dim = 1)
        x = self.mlp(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return(mu,logvar,self.gaussian_sample(mu,logvar),torch.exp(0.5 * logvar))

class state_network(nn.Module):
    def __init__(self,CFG,obs_size,action_size):
        super(state_network, self).__init__()
        self.CFG = CFG
        self.fc_s = FeatureExtractor(obs_size,self.CFG.obs_embed_dim, F.relu,self.CFG.device)
        self.fc_a = FeatureExtractor(action_size,self.CFG.action_embed_dim, F.relu,self.CFG.device)
        self.input_dim = self.CFG.obs_embed_dim +  self.CFG.action_embed_dim + \
            self.CFG.latent_dim
        self.mlp = mlp([self.input_dim] + self.CFG.decoder_hidden,self.CFG.decoder_activation)

        input_dim = self.CFG.decoder_hidden[-1]
        self.fc_mu = nn.Linear(input_dim, obs_size)
        self.fc_logvar = nn.Linear(input_dim, obs_size)

    def gaussian_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, s, a, latent):
        s = self.fc_s(s)
        a = self.fc_a(a)
        x = torch.cat((s,a,latent),dim = 1)
        x = self.mlp(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return(mu,logvar,self.gaussian_sample(mu,logvar),torch.exp(0.5 * logvar))


class Decoder():
    def __init__(self,CFG):
        self.CFG = CFG
        env = self.CFG.env(task = CFG.task)
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.reward_net = reward_network(CFG,obs_size,action_size)
        self.state_net = state_network(CFG,obs_size,action_size)

def main():
    CFG = decoder_CFG()
    CFG.print()
    model = Decoder(CFG)


if __name__ == '__main__':
    main()