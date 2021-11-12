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

class Decoder(nn.Module):
    def __init__(self,CFG):
        super(Decoder, self).__init__()
        self.CFG = CFG
        env = self.CFG.env(task = CFG.task)
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        self.input_dim = self.CFG.latent_dim
        self.mlp = mlp([self.input_dim] + self.CFG.decoder_hidden,self.CFG.decoder_activation)

        input_dim = self.CFG.decoder_hidden[-1]
        self.out_r = nn.Linear(input_dim, 1)
        self.out_o = nn.Linear(input_dim, obs_size)
        self.r_act = nn.Tanh()
        self.o_act = nn.Tanh()

    def forward(self, latent):
        x = self.mlp(latent)
        r = self.out_r(x)
        r = self.r_act(r)
        o = self.out_o(x)
        o = self.o_act(o)
        return r, o


def main():
    CFG = decoder_CFG()
    CFG.print()
    model = Decoder(CFG)


if __name__ == '__main__':
    main()

