from collections import deque
import sys
try:
    sys.path.remove('/home/huy/Desktop/stuff/spinningup')
    sys.path.append("/home/huy/Desktop/github/RL-Projects/model/varibad_for_game")
except:
    pass

import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from spinup.utils.run_utils import setup_logger_kwargs


import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam,SGD
import gym
import time
import torch
import torch.nn as nn
from config.lstm_encoder_cfg import lstm_encoder_CFG

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

class Attention(nn.Module):

    def __init__(self, CFG, input_dim) -> None:
        super(Attention, self).__init__()
        self.CFG = CFG
        self.attn = nn.MultiheadAttention(self.CFG.lstm_hidden_dim * 2, 1, kdim=self.CFG.lstm_hidden_dim * 2, vdim=self.CFG.lstm_hidden_dim * 2)
        self.linear = nn.Linear(input_dim, self.CFG.lstm_hidden_dim * 2)

    def forward(self, input, list_hiddens):
        hiddens = []
        for i in range(self.CFG.n_saved_hidden):
            hidden = torch.cat([list_hiddens[i][0], list_hiddens[i][1]], dim=2)
            hidden = hidden.view(1, hidden.shape[1] * 2, self.CFG.lstm_hidden_dim * 2).squeeze(0)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        input = torch.cat([input, input], dim=0).view(1, input.shape[1] * 2, -1)
        input = self.linear(input)
        output, _ = self.attn(input, hiddens, hiddens) 
        output = output.view(2, output.shape[1] // 2, -1)
        output = torch.split(output, self.CFG.lstm_hidden_dim, 2)
        return output



class lstm_encoder(nn.Module):
    def __init__(self,CFG):
        super(lstm_encoder, self).__init__()
        self.CFG = CFG
        env = CFG.env(task = CFG.task)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        reward_dim = 1

        self.input_dim = self.CFG.obs_embed_dim 
        self.fc_obs = FeatureExtractor(obs_dim, self.CFG.obs_embed_dim, self.CFG.encoder_activation,self.CFG.device)
        if (self.CFG.use_action):
            self.fc_action = FeatureExtractor(action_dim, self.CFG.action_embed_dim, self.CFG.encoder_activation,self.CFG.device)
            self.input_dim += self.CFG.action_embed_dim
        if (self.CFG.use_reward):
            self.fc_reward = FeatureExtractor(reward_dim, self.CFG.reward_embed_dim, self.CFG.encoder_activation,self.CFG.device)
            self.input_dim += self.CFG.reward_embed_dim

        self.lstm = nn.LSTM(self.input_dim, self.CFG.lstm_hidden_dim, self.CFG.lstm_layers)
        self.fc_mu = nn.Linear(self.CFG.lstm_hidden_dim, self.CFG.latent_dim)
        self.fc_logvar = nn.Linear(self.CFG.lstm_hidden_dim, self.CFG.latent_dim)
        self.attn = Attention(CFG, self.input_dim)
        self.list_saved_hidden = deque([], maxlen=self.CFG.n_saved_hidden)
    
    def reset_list_saved_hidden(self):
        self.list_saved_hidden.clear()

    def gaussian_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, obs, action, reward, old_hidden):
        obs = self.fc_obs(obs)
        input = obs
        if (self.CFG.use_action):
            action = self.fc_action(action)
            input = torch.cat((input, action), dim = 1)
        
        if (self.CFG.use_reward):
            reward = self.fc_reward(reward)
            input = torch.cat((input, reward), dim = 1)
        input = input.reshape(-1,input.shape[0],input.shape[1])
        #input shape = (1, 256, 30)

        if len(self.list_saved_hidden) >= self.CFG.n_saved_hidden:
            old_hidden = self.attn.forward(input, self.list_saved_hidden)

        out, hidden = self.lstm(input, old_hidden)
        self.list_saved_hidden.append(hidden)
        out = out.reshape(-1, self.CFG.lstm_hidden_dim)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return (mu,logvar,self.gaussian_sample(mu,logvar),hidden)

def main():
    CFG = lstm_encoder_CFG()
    logger_kwargs = setup_logger_kwargs(CFG.exp_name, CFG.seed,'./experiences')
    print(logger_kwargs)
    CFG.print()

    encoder = lstm_encoder(CFG) 
    print(encoder)
    hidden = (torch.zeros([1, CFG.batch_size, CFG.lstm_hidden_dim], dtype=torch.float), torch.zeros([1, CFG.batch_size, CFG.lstm_hidden_dim], dtype=torch.float))
    for i in range(10):
        print()
        print(hidden[0])
        (mu,logvar,latent,hidden) = encoder(torch.randn(CFG.batch_size,17),torch.randn(CFG.batch_size,6), torch.randn(CFG.batch_size,1), hidden)
        print(hidden[0].shape, hidden[1].shape)
        print((mu.shape,logvar.shape,latent.shape))


if __name__ == '__main__':
    main()
