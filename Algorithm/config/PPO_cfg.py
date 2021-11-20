import numpy as np
import torch
from torch.optim import Adam
import gym

class PPO_CFG :
    def __init__(self):
        self.env = 'HalfCheetah-v2'
        
        env = gym.make(self.env)

        self.hid = 64
        self.l = 2
        self.gamma = 0.99
        self.seed = 1
        self.cpu = 1
        self.steps_per_epoch = 4000
        self.epochs = 400
        self.exp_name = 'ppo'

        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
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
    CFG = PPO_CFG()