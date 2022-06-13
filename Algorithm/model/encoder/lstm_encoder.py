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
from memory_profiler import memory_usage

class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards"""

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

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        
        return q.transpose(0, 1), attn

class Attention(nn.Module):

    def __init__(self, CFG, input_dim) -> None:
        super(Attention, self).__init__()
        self.CFG = CFG
        self.attn = MultiHeadAttention(self.CFG.n_head_attn, self.CFG.lstm_hidden_dim, d_k=self.CFG.lstm_hidden_dim, d_v=self.CFG.lstm_hidden_dim, dropout=self.CFG.attn_dropout)
        self.linear = nn.Linear(input_dim, self.CFG.lstm_hidden_dim)

    def forward(self, input, list_hiddens):
        hiddens_0 = []
        hiddens_1 = []
        for i in range(self.CFG.n_saved_hidden):
            hidden = list_hiddens[i]
            hiddens_0.append(hidden[0].view(1, hidden[0].shape[1] * self.CFG.lstm_layers, self.CFG.lstm_hidden_dim).squeeze(0))
            hiddens_1.append(hidden[1].view(1, hidden[1].shape[1] * self.CFG.lstm_layers, self.CFG.lstm_hidden_dim).squeeze(0))
    
        hiddens_0 = torch.stack(hiddens_0)
        hiddens_1 = torch.stack(hiddens_1)
        input = torch.cat([input, input], dim=1)
        input = self.linear(input)
        output_0, _ = self.attn(input, hiddens_0, hiddens_0)
        output_1, _ = self.attn(input, hiddens_1, hiddens_1)
        output_0 = output_0.view(self.CFG.lstm_layers, output_0.shape[1] // 2, -1)
        output_1 = output_1.view(self.CFG.lstm_layers, output_1.shape[1] // 2, -1)
        
        return (output_0, output_1)



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
    
    def reset_list_saved_hidden(self, batch_size):
        self.list_saved_hidden.clear()
        for i in range(self.CFG.n_saved_hidden):
            hidden = (torch.zeros([self.CFG.lstm_layers, batch_size, self.CFG.lstm_hidden_dim], dtype=torch.float), torch.zeros([self.CFG.lstm_layers, batch_size, self.CFG.lstm_hidden_dim], dtype=torch.float))
            self.list_saved_hidden.append(hidden)

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

        old_hidden = self.attn.forward(input, self.list_saved_hidden)

        out, hidden = self.lstm(input, old_hidden)

        #f = open("/home/loc/Desktop/ML/Project/Meta-Model-Based-RL/Algorithm/sources/hiddens", "a")
        #f.write(str(old_hidden[1][1]) + "\n")
        #for params in self.lstm.parameters():
        #   f.write(str(params))

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
