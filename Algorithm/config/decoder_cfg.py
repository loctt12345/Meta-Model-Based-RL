import numpy as np
import torch
import torch.nn as nn
import time

class decoder_CFG :
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = 'HalfCheetah-v2'
        self.seed = 0
        self.cpu = 1
        self.latent_dim = 32
        self.batch_size = 3

        self.obs_embed_dim = 10
        self.action_embed_dim = 10
        self.reward_embed_dim = 5
        self.construct_step = 5 # use to construct latent
        self.inference_step = 5 # use latent to predict next steps

        self.network_hidden = [64,64]
        self.network_activation = nn.ReLU


    def print(self):
        print('----------------------configure of PPO------------------------------')
        attrs = vars(self)
        for item in attrs.items():
            print(item)
        print('--------------------------------------------------------------------')