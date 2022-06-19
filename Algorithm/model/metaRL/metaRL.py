
from tensorboardX import SummaryWriter
import os
import numpy as np
import torch
from torch.optim import Adam,SGD
from torch.distributions import Normal 
import gym
import collections
import time
import random
import os.path as osp
import math
import random

import model.policy.PPO_spinup_core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs




class metaRL():
    def __init__(self, CFG, env = None, policy = None, encoder = None, decoder = None,
    logger_kwargs = None,use_latent = True):
        self.CFG = CFG
        self.env = env
        self.policy = policy
        self.encoder = encoder
        self.decoder = decoder
        self.logger_kwargs = logger_kwargs
        self.use_latent = use_latent

        print(self.encoder)
        print(self.decoder.reward_net)
        print(self.decoder.state_net)
        print(self.policy.ac.pi)
        print(self.policy.ac.v)
        # return
        
        TB_name = f'seed = {self.CFG.seed} ' +logger_kwargs['exp_name']
        os.makedirs('TB_logs', exist_ok = True)
        dir = 'TB_logs/'+str(TB_name)
        self.writer = SummaryWriter(log_dir = dir)

        self.policy.assign_writer(self.writer)

        # self.decoder_reward_optimizer   = Adam(self.decoder.reward_net.parameters(), lr=self.CFG.lr_reward_net)
        # self.decoder_state_optimizer    = Adam(self.decoder.state_net.parameters() , lr=self.CFG.lr_state_net)
        # self.encoder_optimizer          = Adam(self.encoder.parameters() , lr=self.CFG.lr_encoder)
        self.VAE_optimizer = Adam([*self.encoder.parameters(), *self.decoder.reward_net.parameters(),*self.decoder.state_net.parameters()], lr=CFG.lr_VAE)

    def tensor(self,x):
        try:
            tensor_ = torch.as_tensor(np.array(x), dtype=torch.float32).to(self.CFG.device)
        except:
            tensor_ = torch.as_tensor(x, dtype=torch.float32).to(self.CFG.device)
        return tensor_

    def compute_kl_loss(self, latent_mean,latent_mean_new, latent_logvar,latent_logvar_new):
        # -- KL divergence
        gauss_dim = latent_mean.shape[-1]
        # add the gaussian prior
        # https://arxiv.org/pdf/1811.09975.pdf
        # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
        mu = latent_mean_new
        m = latent_mean
        logE = latent_logvar_new
        logS = latent_logvar
        kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
            1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))


        return kl_divergences

    def update_VAE(self, data,epoch_):



        obs,rew,act = data['obs'],data['rew'],data['act']
        o_list = obs[:-1]
        next_o_list = obs[1:]
        r_list = rew[:-1].reshape(-1,1)
        a_list = act[:-1]
        length = o_list.shape[0]
        max_start = length - self.CFG.construct_step - self.CFG.inference_step
        ls_reward,ls_state,ls_loss,ls_std_o,ls_std_r,ls_loss_std_r,ls_loss_std_o = [],[],[],[],[],[],[]
        KL_loss_ls = []
        ls_o_expect, ls_o_next = [],[]

        for update_epoch in range(self.CFG.VAE_update_epoch):

            self.VAE_optimizer.zero_grad()
            KL_loss = None
            start_idx = np.asarray([random.randint(0,max_start) for i in range(self.CFG.batch_size)])
            hidden = (torch.zeros([self.CFG.lstm_layers, self.CFG.batch_size, self.CFG.lstm_hidden_dim], dtype=torch.float), torch.zeros([self.CFG.lstm_layers, self.CFG.batch_size, self.CFG.lstm_hidden_dim], dtype=torch.float))
            self.encoder.reset_list_saved_hidden(self.CFG.batch_size)
            for i in range(self.CFG.construct_step):
                o,next_o,r,a = o_list[start_idx + i],next_o_list[start_idx + i],r_list[start_idx + i],a_list[start_idx + i]
                (mu_new,logvar_new,latent,hidden) = self.encoder(self.tensor(o),self.tensor(a),self.tensor(r),hidden)
                if (i):
                    val = self.compute_kl_loss(mu,mu_new,logvar,logvar_new).mean()
                    if (KL_loss):
                        KL_loss+= val
                    else:
                        KL_loss = val
                mu = mu_new
                logvar = logvar_new
            KL_loss_ls.append(KL_loss.detach().cpu().numpy())
            loss_r = None
            loss_o = None
            loss_std_r = None
            loss_std_o = None
            o_expect = None
            # loss_latent_std = -torch.log(torch.exp(0.5 * logvar).mean())
            for i in range(self.CFG.construct_step + self.CFG.inference_step-1):
                o_his,next_o,r,a = self.tensor(o_list[start_idx + i]),self.tensor(next_o_list[start_idx + i]),self.tensor(r_list[start_idx + i]),self.tensor(a_list[start_idx + i])

                if (o_expect is None):
                    o = o_his
                else:
                    if (random.randint(0,1)==1):
                        o = o_expect
                    else:
                        o = o_his

                (r_mu,r_logvar,r_expect,r_std) = self.decoder.reward_net(self.tensor(o),self.tensor(a),self.tensor(next_o),latent)
                (o_mu,o_logvar,o_expect,o_std) = self.decoder.state_net(self.tensor(o),self.tensor(a),latent)
                
                ls_o_expect.append(o_expect.detach().cpu().numpy())
                ls_o_next.append(next_o.detach().cpu().numpy())

                ls_std_r.append(np.mean(r_std.detach().cpu().numpy()))
                ls_std_o.append(np.mean(o_std.detach().cpu().numpy()))
                if (loss_r):

                    loss_r += (r_expect - r).pow(2).mean() 
                    loss_o += (o_expect - next_o).pow(2).mean() 
                    loss_std_r += - torch.log(r_std).mean() 
                    loss_std_o += - torch.log(o_std).mean()

                else:
                    loss_r = (r_expect - r      ).pow(2).mean()
                    loss_o = (o_expect - next_o ).pow(2).mean()
                    loss_std_r = - torch.log(r_std).mean()
                    loss_std_o = - torch.log(o_std).mean()
            loss = (
                loss_r + 
                loss_o  + 
                loss_std_o/50 + 
                loss_std_r/100 +
                # loss_latent_std/200 + 
                KL_loss
            )

            ls_reward.append(loss_r.detach().cpu().numpy())
            ls_state.append(loss_o.detach().cpu().numpy())
            ls_loss_std_r.append(loss_std_r.detach().cpu().numpy())
            ls_loss_std_o.append(loss_std_o.detach().cpu().numpy())
            ls_loss.append(loss.detach().cpu().numpy())

            loss.backward()

            self.VAE_optimizer.step()
        self.writer.add_scalar('VAE_loss/loss reward',np.mean(ls_reward),epoch_)
        self.writer.add_scalar('VAE_loss/loss latent KL',np.mean(KL_loss_ls),epoch_)
        self.writer.add_scalar('VAE_loss/loss state',np.mean(ls_state),epoch_)
        self.writer.add_scalar('VAE_loss/loss std reward',np.mean(ls_loss_std_r),epoch_)
        self.writer.add_scalar('VAE_loss/loss std observation',np.mean(ls_loss_std_o),epoch_)
        # self.writer.add_scalar('VAE_loss/loss latent std',loss_latent_std.detach().numpy(),epoch_)
        self.writer.add_scalar('VAE_loss/VAE loss',np.mean(ls_loss),epoch_)
        self.writer.add_scalar('VAE_loss/std reward',np.mean(ls_std_r),epoch_)
        self.writer.add_scalar('VAE_loss/std observation',np.mean((ls_std_o)),epoch_)
        self.writer.add_histogram('VAE_loss/observarion expect',np.asarray(ls_o_expect).reshape(-1),epoch_)
        self.writer.add_histogram('VAE_loss/observation true',np.asarray(ls_o_next).reshape(-1),epoch_)

    def update(self, epoch_):
        data = self.policy.buf.get()
        # ---------------------------------train VAR-------------------------------------
        if (self.use_latent):
            self.update_VAE(data,epoch_)
        # ---------------------------------train policy-----------------------------------
        target_value = data['target_value'].detach().cpu().numpy()
        self.writer.add_scalar('value function/target value mean',np.mean(target_value),epoch_)
        self.writer.add_scalar('value function/target value std',np.std(target_value),epoch_)
        self.writer.add_scalar('value function/target value max',np.max(target_value),epoch_)
        self.writer.add_scalar('value function/target value min',np.min(target_value),epoch_)
        self.writer.add_histogram('histogram/target value',target_value,epoch_)


        pi_l_old, pi_info_old = self.policy.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old,_ = self.policy.compute_loss_v(data,True,epoch_)
        v_l_old = v_l_old.item()
        ent_his = []
        # Train policy with multiple steps of gradient descent
        for i in range(self.CFG.train_pi_iters):
            self.policy.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.policy.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            ent = mpi_avg(pi_info['ent'])
            self.policy.maximum_ent = max(self.policy.maximum_ent,ent)

            self.policy.KL_his[i].append(kl)
            ent_his.append(ent)
            mean_kl = np.mean(self.policy.KL_his[i])
            max_kl = np.max(self.policy.KL_his[i])
            min_kl = np.min(self.policy.KL_his[i])
            std_kl = np.std(self.policy.KL_his[i])
            self.writer.add_scalar('KL/KL divergence_mean',mean_kl,i)
            self.writer.add_scalar('KL/KL divergence_std',std_kl,i)
            self.writer.add_scalar('KL/KL divergence_max',max_kl,i)
            self.writer.add_scalar('KL/KL divergence_min',min_kl,i)
            
            if kl > 1.5 * self.CFG.target_kl:
                for j in range(i+1,self.CFG.train_pi_iters):
                    self.policy.KL_his[j].append(kl)
                    mean_kl = np.mean(self.policy.KL_his[j])
                    max_kl = np.max(self.policy.KL_his[j])
                    min_kl = np.min(self.policy.KL_his[j])
                    std_kl = np.std(self.policy.KL_his[j])
                    self.writer.add_scalar('KL/KL divergence_mean',mean_kl,j)
                    self.writer.add_scalar('KL/KL divergence_std',std_kl,j)
                    self.writer.add_scalar('KL/KL divergence_max',max_kl,j)
                    self.writer.add_scalar('KL/KL divergence_min',min_kl,j)

                self.policy.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.policy.ac.pi)    # average grads across MPI processes
            self.policy.pi_optimizer.step()
        self.writer.add_scalar('entropy/relative policy entropy',np.mean(ent_his)/self.policy.maximum_ent,epoch_)
        self.writer.add_scalar('entropy/policy entropy',np.mean(ent_his),epoch_)
        
        self.policy.logger.store(StopIter=i)
        value_residual_variance = []
        # Value function learning
        for i in range(self.CFG.train_v_iters):
            self.policy.vf_optimizer.zero_grad()
            loss_v,v_info = self.policy.compute_loss_v(data)
            value_residual_variance.append(v_info['value_residual_variance'].detach().cpu().numpy())
            loss_v.backward()
            mpi_avg_grads(self.policy.ac.v)    # average grads across MPI processes
            self.policy.vf_optimizer.step()
        self.writer.add_scalar('value function/value_residual_variance',np.mean(value_residual_variance),epoch_)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.writer.add_scalar('Loss/Loss_pi',pi_l_old,epoch_)
        self.writer.add_scalar('Loss/Loss_v',v_l_old,epoch_)

        self.policy.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def train(self):
        self.encoder.train()
        self.decoder.reward_net.train()
        self.decoder.state_net.train()
        self.policy.ac.train()
        start_time = time.time()
        self.env.reset_task(task_ls = self.CFG.train_tasks)
        o, ep_ret, ep_len,rewards,std_ls = self.env.reset(), 0, 0, [],[]
        episode_cnt = 0
        episode_return = collections.deque(maxlen=100)
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.CFG.epochs):
            hidden = (torch.zeros([self.CFG.lstm_layers, 1, self.CFG.lstm_hidden_dim], dtype=torch.float).to(self.CFG.device), torch.zeros([self.CFG.lstm_layers, 1, self.CFG.lstm_hidden_dim], dtype=torch.float).to(self.CFG.device))
            self.encoder.reset_list_saved_hidden(1)
            latent = torch.zeros(1,self.CFG.latent_dim).to(self.CFG.device)
            mu_ls = np.asarray([])
            latent_diff = []
            for t in range(self.policy.local_steps_per_epoch):
                if (t% 50 == 0):
                    self.writer.add_histogram('histogram/VAE_latent',latent,t/50)
                if (self.use_latent):
                    input = torch.cat((self.tensor([o]),latent),dim = 1)
                else:
                    input = self.tensor([o])
                a, v, logp = self.policy.ac.step(input)
                
                """
                if (round(time.time() - start_time)) % 100 == 0:
                    f = open("/home/loc/Desktop/ML/Project/Meta-Model-Based-RL/Algorithm/sources/hiddens", "a")
                    f.write(str(hidden[1][1]) + "\n")
                    f.close()
                """

                next_o, r, d, _ = self.env.step(a)
                r *= self.CFG.reward_scale
                rewards.append(r)
                ep_ret += r
                ep_len += 1

                # save and log
                self.policy.buf.store(o, a, r, v, logp,latent.detach().cpu().numpy())
                self.policy.logger.store(VVals=v)
                
                # Update obs (critical!)
                if (self.use_latent):
                    (mu,logvar,new_latent,hidden) = self.encoder(self.tensor([o]),self.tensor(a),self.tensor([[r]]),hidden)
                    latent_diff.append(np.sum((latent-new_latent).pow(2).detach().cpu().numpy()))
                    latent = new_latent
                    mu_ls = np.concatenate(((mu_ls,mu.detach().cpu().numpy().reshape(-1))))
                    std_ls.append(np.mean(np.exp(0.5 * logvar.detach().cpu().numpy())))
                o = next_o

                timeout = ep_len == self.CFG.max_ep_len
                terminal = d or timeout
                epoch_ended = t==self.policy.local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        if (self.use_latent):
                            input = torch.cat((self.tensor([o]),latent),dim = 1)
                        else:
                            input = self.tensor([o])
                        _, v, _ = self.policy.ac.step(input)
                    else:
                        v = 0
                    self.policy.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.policy.logger.store(EpRet=ep_ret, EpLen=ep_len)
                        episode_return.append(ep_ret)
                        self.writer.add_scalar('environment/return_mean',np.mean(episode_return),episode_cnt)
                        self.writer.add_scalar('environment/return_std',np.std(episode_return),episode_cnt)
                        self.writer.add_scalar('environment/return_max',np.max(episode_return),episode_cnt)
                        self.writer.add_scalar('environment/return_min',np.min(episode_return),episode_cnt)

                        self.writer.add_scalar('environment/reward_mean',np.mean(rewards),episode_cnt)
                        self.writer.add_scalar('environment/reward_std',np.std(rewards),episode_cnt)
                        self.writer.add_scalar('environment/reward_max',np.max(rewards),episode_cnt)
                        self.writer.add_scalar('environment/reward_min',np.min(rewards),episode_cnt)
                        
                        self.writer.add_scalar('environment/episode_length',ep_len,episode_cnt)
                        
                        self.writer.add_scalar('VAE_latent_std/mean',np.mean(std_ls),episode_cnt)
                        self.writer.add_scalar('VAE_latent_mean/mean',np.mean(mu_ls),episode_cnt)
                        self.writer.add_scalar('VAE_latent_mean/max',np.max(mu_ls),episode_cnt)
                        self.writer.add_scalar('VAE_latent_mean/min',np.min(mu_ls),episode_cnt)
                        self.writer.add_scalar('VAE_latent_mean/std',np.std(mu_ls),episode_cnt)

                                    
                        episode_cnt += 1
                    self.env.reset_task(task_ls = self.CFG.train_tasks)
                    if (self.use_latent):
                        latent = torch.zeros(1,self.CFG.latent_dim).to(self.CFG.device)
                        hidden = (torch.zeros([self.CFG.lstm_layers, 1, self.CFG.lstm_hidden_dim], dtype=torch.float).to(self.CFG.device), torch.zeros([self.CFG.lstm_layers, 1, self.CFG.lstm_hidden_dim], dtype=torch.float).to(self.CFG.device))
                        self.encoder.reset_list_saved_hidden(1)

                    o, ep_ret, ep_len,rewards = self.env.reset(), 0, 0,[]
                    std_ls = []
                    mu_ls = np.asarray([])

            self.writer.add_scalar('VAE_latent_diff/L2 norm',np.mean(latent_diff),epoch)


            # Save model
            if (epoch % self.CFG.save_freq == 0) or (epoch == self.CFG.epochs-1):
                self.save_model()

            # Perform PPO update!
            self.update(epoch)

            # Log info about epoch
            self.policy.logger.log_tabular('Epoch', epoch)
            self.policy.logger.log_tabular('EpRet', with_min_and_max=True)
            self.policy.logger.log_tabular('EpLen', average_only=True)
            self.policy.logger.log_tabular('VVals', with_min_and_max=True)
            self.policy.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.CFG.steps_per_epoch)
            self.policy.logger.log_tabular('LossPi', average_only=True)
            self.policy.logger.log_tabular('LossV', average_only=True)
            self.policy.logger.log_tabular('DeltaLossPi', average_only=True)
            self.policy.logger.log_tabular('DeltaLossV', average_only=True)
            self.policy.logger.log_tabular('Entropy', average_only=True)
            self.policy.logger.log_tabular('KL', average_only=True)
            self.policy.logger.log_tabular('ClipFrac', average_only=True)
            self.policy.logger.log_tabular('StopIter', average_only=True)
            self.policy.logger.log_tabular('Time', time.time()-start_time)
            self.policy.logger.dump_tabular()
        self.writer.close()

    def test(self):
        self.encoder.valid()
        self.decoder.reward_net.valid()
        self.decoder.state_net.valid()
        self.policy.ac.valid()
        for ttest in range(10):
            hidden = (torch.zeros([self.CFG.lstm_layers, 1, self.CFG.lstm_hidden_dim], dtype=torch.float), torch.zeros([self.CFG.lstm_layers, 1, self.CFG.lstm_hidden_dim], dtype=torch.float))
            self.encoder.reset_list_saved_hidden(1)
            latent = torch.zeros(1,self.CFG.latent_dim)
            self.env.reset_task(task_ls = self.CFG.test_tasks)
            o, ep_ret, ep_len,rewards = self.env.reset(), 0, 0, []
            done = False
            turn = 0
            print(self.env.task)
            while( not done):
                if (self.CFG.render):
                    self.env.render()
                    print(turn)
                    time.sleep(0.03) 
                if (self.use_latent):
                    input = torch.cat((self.tensor([o]),latent),dim = 1)
                else:
                    input = self.tensor([o])
                a, v, logp = self.policy.ac.step(input)
                next_o, r, done, _ = self.env.step(a)
                # r *= self.CFG.reward_scale
                rewards.append(r)
                if (self.use_latent):
                    (mu,logvar,latent,hidden) = self.encoder(self.tensor([o]),self.tensor(a),self.tensor([[r]]),hidden)
                
                
                o = next_o
                turn += 1
                # print(turn)
                if (turn == 1000):
                    done = True
                    print(np.sum(rewards))
                    if (self.use_latent):
                        latent = torch.zeros(1,self.CFG.latent_dim)
            self.env.stop_viewer()

    def save_model(self):
        fpath = self.logger_kwargs['output_dir']
        fpath = osp.join(fpath,'pyt_save')
        os.makedirs(fpath, exist_ok = True)

        # self.policy.logger.save_state({'env': self.env}, None)

        if (self.policy):
            torch.save(self.policy.ac.pi.state_dict(), osp.join(fpath,'actor.pt'))
            torch.save(self.policy.ac.v.state_dict(), osp.join(fpath,'critic.pt'))
        if (self.encoder):
            torch.save(self.encoder.state_dict(), osp.join(fpath,'encoder.pt'))
        if (self.decoder):
            torch.save(self.decoder.reward_net.state_dict(), osp.join(fpath,'decoder_reward.pt'))
            torch.save(self.decoder.state_net.state_dict(), osp.join(fpath,'decoder_state.pt'))

    def load_model(self,fpath):
        # self.policy.ac = torch.load(osp.join(fpath,'policy.pt'))

        self.policy.ac.pi.load_state_dict(torch.load(osp.join(fpath,'actor.pt')))
        self.policy.ac.v.load_state_dict(torch.load(osp.join(fpath,'critic.pt')))
        self.encoder.load_state_dict(torch.load(osp.join(fpath,'encoder.pt')))
        self.decoder.reward_net.load_state_dict(torch.load(osp.join(fpath,'decoder_reward.pt')))
        self.decoder.state_net.load_state_dict(torch.load(osp.join(fpath,'decoder_state.pt')))
