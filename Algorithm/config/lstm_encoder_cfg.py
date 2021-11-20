import numpy as np
import torch
from torch.optim import Adam
import gym
import time

class lstm_encoder_CFG :
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = 'HalfCheetah-v2'
        self.seed = 0
        self.cpu = 1
        self.exp_name = 'VAE'
        self.lstm_hidden_dim = 64
        self.latent_dim = 32
        self.lstm_layers = 1
        self.batch_size = 1

        self.obs_embed_dim = 10
        self.action_embed_dim = 10
        self.reward_embed_dim = 5
        self.use_action = True
        self.use_reward = True


    def print(self):
        print('----------------------configure of PPO------------------------------')
        attrs = vars(self)
        for item in attrs.items():
            print(item)
        print('--------------------------------------------------------------------')