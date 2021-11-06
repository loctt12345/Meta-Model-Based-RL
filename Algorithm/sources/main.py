import sys
import os
try:
    sys.path.append(os.path.abspath("."))
    # sys.path.append("C:\\Users\\84915\\Desktop\\RL-Projects\\model\\varibad_for_game")
except:
    pass
import numpy as np
import torch
from torch.optim import Adam,SGD
import gym
import time
from envs import *
from config.main_cfg import main_CFG

from model.policy.PPO_spinup import PPO
from model.encoder.lstm_encoder import lstm_encoder
from model.decoder.decoder import Decoder
from model.metaRL.metaRL import metaRL
import numpy as np
import model.policy.PPO_spinup_core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

def load_model(model,CFG):
    print(f'{CFG.mode} use pretrained at folder: {CFG.trained_folder}')
    try:
        model.load_model(CFG.trained_folder)
    except:
        print('fail to load model')


def main():
    CFG = main_CFG()
    mpi_fork(CFG.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(CFG.exp_name, CFG.seed,'.\\experiences')
    logger_kwargs = setup_logger_kwargs(CFG.exp_name, CFG.seed,'./experiences')
    print(logger_kwargs)
    env = CFG.env(task = CFG.task)
    # print(env.task)

    encoder = lstm_encoder(CFG)
    decoder = Decoder(CFG)
    policy = PPO(lambda : CFG.env(task = CFG.task), CFG, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[CFG.hid]*CFG.l),logger_kwargs=logger_kwargs,use_latent=CFG.use_latent,
        setup_writer = False)

    model = metaRL(CFG, env = env, policy = policy, encoder = encoder,
        decoder = decoder, logger_kwargs = logger_kwargs,use_latent = CFG.use_latent)

    # -------------------------train------------------------------------
    if (CFG.mode == 'train'):
        if (CFG.use_pretrained):
            load_model(model, CFG)
        else:
            print('training from scratch')

        model.train()

    # --------------------------test--------------------------------------
    if (CFG.mode == 'test'):
        if (CFG.use_pretrained):
            load_model(model, CFG)
        else:
            print('test random policy')

        model.test()





if __name__ == '__main__':
    main()




    # def tensor(x):
    #     return torch.as_tensor(x, dtype=torch.float32)

    # latent = torch.randn(1,CFG.latent_dim)
    # o = [env.reset()]
    # save = []
    # for i in range(CFG.construct_step + CFG.inference_step):
    #     input = torch.cat((tensor(o),latent),dim = 1)

    #     a,v,logp = model.policy.ac.step(input)
    #     next_o, r, d, _ = env.step(a)
    #     save.append((o,a,next_o,r))
    #     o = [next_o]

    # hidden = (torch.zeros([1, 1, CFG.lstm_hidden_dim], dtype=torch.float), torch.zeros([1, 1, CFG.lstm_hidden_dim], dtype=torch.float))
    # for i in range(CFG.construct_step):
    #     (o,a,next_o,r) = save[i]
    #     (mu,logvar,latent,hidden) = encoder(tensor(o),tensor(a),tensor([[r]]),hidden)

    # for i in range(CFG.construct_step + CFG.inference_step-1):
    #     (o,a,next_o,r) = save[i]
    #     # print(tensor(o).shape,tensor(a).shape,tensor([next_o]).shape,tensor([[r]]).shape)
    #     (_,_,r_expect,_) = decoder.reward_net(tensor(o),tensor(a),tensor([next_o]),latent)
    #     (_,_,o_expect,_) = decoder.state_net(tensor(o),tensor(a),latent)
    #     print(i+1,r_expect.shape,o_expect.shape)


    # return
