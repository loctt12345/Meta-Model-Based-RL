import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import gym
from envs import *

class main_CFG :
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'-------------{self.device}-------------------------')
        self.env = HalfCheetahHFieldEnv#HalfCheetahBlocksEnv#HalfCheetahHFieldEnv
        self.task = 'None'
        # training settings
        self.mode = 'train' # 'train' or 'test'
        self.num_ensemble = 2
        self.seed =5
        self.cpu = 1
        self.exp_name = 'metaRL'     # name for save state dict
        self.use_pretrained = True
        self.trained_folder = '.\\experiences\\metaRL\\metaRL_s4\\pyt_save'
        self.render = False
        self.batch_size = 256         # batch size for train
        self.VAE_update_epoch = 20
        self.reward_scale = 1e-1
        self.use_latent = True
        self.train_tasks = ['hfield', 'hill', 'gentle']
        self.test_tasks = ['basin','steep']
        self.test_tasks = ['gentle']
        


        # Encoder config

        self.lstm_hidden_dim = 64   # hidden dim of lstm
        self.latent_dim = 64        # latent dim after encoder 
        self.lstm_layers = 2        # number layer of lstm 3
        self.obs_embed_dim = 15
        self.action_embed_dim = 10
        self.reward_embed_dim = 5
        self.encoder_activation = F.relu
        self.use_action = True
        self.use_reward = True
        self.lr_VAE = 1e-5#1e-4

        # decoder config

        self.construct_step = 40 # use to construct latent
        self.inference_step = 80 # use latent to predict next steps
        self.decoder_hidden = [64,64]
        self.decoder_activation = nn.ReLU

        # Policy config

        self.hid = 64
        self.l = 2

        # PPO config
        self.gamma = 0.99
        self.steps_per_epoch = 20000
        self.epochs = 1000
        self.clip_ratio = 0.2
        self.pi_lr = 1e-5#3e-4
        self.vf_lr = 1e-5#1e-3
        self.train_pi_iters = 80
        self.train_v_iters = 80
        self.lam = 0.97
        self.max_ep_len = 1000
        self.target_kl = 0.01
        self.save_freq = 10


    def print(self):
        print('----------------------configure of PPO------------------------------')
        attrs = vars(self)
        for item in attrs.items():
            print(item)
        print('--------------------------------------------------------------------')


if __name__ == '__main__':
    CFG = main_CFG()
