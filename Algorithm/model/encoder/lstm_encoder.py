from doctest import COMPARISON_FLAGS
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
    def __init__(self, CFG, q_dim, v_dim, k_dim):
        super(Attention, self).__init__()
        self.CFG = CFG
        self.f_chosen_hidden = [0 for i in range(self.CFG.n_saved_hidden)]
        self.q = nn.Linear(q_dim, k_dim)
        self.k = nn.Linear(k_dim, k_dim)
        self.v = nn.Linear(v_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def standardlize_data(self, input, saved_hidden):
        # Solve hidden 
        tmp = []
        for i in range(len(saved_hidden)):
            tmp.append(torch.cat((saved_hidden[i][0][self.CFG.lstm_layers - 1], saved_hidden[i][1][self.CFG.lstm_layers - 1]), dim=1))
        hiddens = torch.stack(tmp)
        hiddens = torch.transpose(hiddens, 0, 1)
        # Solve input (1, 1, 30)
        return input[0], hiddens
    
    def f_hidden(self):
        return self.f_chosen_hidden

    def forward(self, input, hiddens_list):
        Q, K = self.standardlize_data(input, hiddens_list)
        V = self.v(K)
        K = self.k(K)
        Q = self.q(Q)
        Q = torch.unsqueeze(Q, 2)
        dot_product = K @ Q
        
        V = torch.reshape(V, (K.shape[0], self.CFG.n_saved_hidden, -1))
        sof = self.softmax(dot_product)
        context = sof * V
        positions =  context.max(dim=1)[1]
        for x in positions:
            self.f_chosen_hidden[x[0]] += 1
        list = []
        for i in range(len(hiddens_list)):
            list.append((torch.transpose(hiddens_list[i][0], 0, 1), torch.transpose(hiddens_list[i][1], 0, 1)))
        list_h = []
        list_c = []
        for i in range(len(positions)):
            try:
                list_h.append(list[positions[i]][0][i])
            except:
                print("Q:" + str(Q.shape))
                print("K:" + str(K.shape))
                print("sof:" + str(sof.shape))
                print("context:" + str(context.shape))
                print(positions)
                print(list[0][0].shape)
            list_c.append(list[positions[i]][1][i])

        list_h = torch.transpose(torch.stack(list_h), 0, 1)
        list_c = torch.transpose(torch.stack(list_c), 0, 1)

        hidden_chosen = (list_h, list_c)
        return hidden_chosen

class lstm_encoder(nn.Module):
    def __init__(self,CFG):
        super(lstm_encoder, self).__init__()
        self.CFG = CFG
        env = CFG.env(task = CFG.task)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        reward_dim = 1
        # For saving hiddens in history to attention with state
        initial_hidden = (torch.zeros([self.CFG.lstm_layers, 1, self.CFG.lstm_hidden_dim], dtype=torch.float), torch.zeros([self.CFG.lstm_layers, 1, self.CFG.lstm_hidden_dim], dtype=torch.float))
        self.list_saved_hidden = [initial_hidden] * self.CFG.n_saved_hidden

        #Attention for chosing hiddens
        self.attention = Attention(self.CFG, 30, self.CFG.lstm_hidden_dim * 2, self.CFG.lstm_hidden_dim * 2)

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
    
    def reset_saved_hidden(self, batch_size):
        self.attention.f_chosen_hidden = [0 for i in range(self.CFG.n_saved_hidden)]
        initial_hidden = (torch.zeros([self.CFG.lstm_layers, batch_size, self.CFG.lstm_hidden_dim], dtype=torch.float), torch.zeros([self.CFG.lstm_layers, batch_size, self.CFG.lstm_hidden_dim], dtype=torch.float))
        self.list_saved_hidden = [initial_hidden] * self.CFG.n_saved_hidden

    def save_hidden(self, new_hidden):
        self.list_saved_hidden.append(new_hidden)
        if (len(self.list_saved_hidden) > self.CFG.n_saved_hidden):
            self.list_saved_hidden.pop(0)
    
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
        
        #Chosing hidden
        old_hidden = self.attention(input, self.list_saved_hidden)

        out, hidden = self.lstm(input, old_hidden)
        self.save_hidden(hidden)
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
