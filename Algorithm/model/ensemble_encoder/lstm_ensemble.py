import sys
try:
    #sys.path.remove('/home/huy/Desktop/stuff/spinningup')
    sys.path.append('C:/Users/trnth/Desktop/ML/Meta-RL/Meta-Model-Based-RL/Algorithm/')
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

    
    
    def gaussian_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, obs, action, reward):
        obs = self.fc_obs(obs)
        input = obs
        if (self.CFG.use_action):
            action = self.fc_action(action)
            input = torch.cat((input, action), dim = 1)
        
        if (self.CFG.use_reward):
            reward = self.fc_reward(reward)
            input = torch.cat((input, reward), dim = 1)

        input = input.reshape(-1,input.shape[0],input.shape[1])
        out, new_hidden = self.lstm(input, self.hidden)
        out = out.reshape(-1, self.CFG.lstm_hidden_dim) 
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        self.hidden = new_hidden
        return (mu, logvar, self.gaussian_sample(mu,logvar), new_hidden)

class lstm_ensemble():
    
    def set_hidden_shape(self, layer, batch_size, hidden_dim):
        for i in range(self.ensemble_size):
            self.models[i].hidden = (torch.zeros([layer, batch_size, hidden_dim], dtype=torch.float), torch.zeros([layer, batch_size, hidden_dim], dtype=torch.float))

    def __init__(self, CFG):
        self.ensemble_size = CFG.num_ensemble
        self.models = []
        for i in range(self.ensemble_size):
            model = lstm_encoder(CFG)
            self.models.append(model)

    def forward(self, obs, action, reward):
        mus = []
        logvars = []
        latents = []
        new_hiddens = []
        for i in range(self.ensemble_size):
            (mu, logvar, latent, new_hidden) = self.models[i].forward(obs, action, reward)
            mus.append(mu)
            logvars.append(logvar)
            latents.append(latent)
            new_hiddens.append(new_hidden)

        mean_latent = torch.mean(torch.stack(latents), dim = 0)
        mus = torch.stack(mus)
        logvars = torch.stack(logvars)

        return (mus, logvars, mean_latent, new_hiddens)



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