import sys
try:
    sys.path.remove('/home/huy/Desktop/stuff/spinningup')
    sys.path.append("/home/huy/Desktop/github/RL-Projects/model/varibad_for_game")
except:
    pass
import numpy as np
import torch
from torch.optim import Adam,SGD
import gym
import time


from config.PPO_cfg import PPO_CFG
import model.policy.PPO_spinup_core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95,latent_dim = None):
        self.latent_dim = latent_dim
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        if (latent_dim):
            self.latent_buf = np.zeros(core.combined_shape(size, latent_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.target_value_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp,latent):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        if (self.latent_dim):
            self.latent_buf[self.ptr] = latent
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # calculate target value
        self.target_value_buf[path_slice] = rews[:-1] + self.gamma * vals[1:]


        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf,target_value = self.target_value_buf,
                    val=self.val_buf,rew = self.rew_buf)
        if (self.latent_dim):
            data['latent'] = self.latent_buf
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class PPO():
    def __init__(self, env_fn, CFG, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(), logger_kwargs=dict(), use_latent =False,setup_writer = True):

        setup_pytorch_for_mpi()
        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        if (setup_writer):
            import time
            import os
            start_time = time.time()
            from tensorboardX import SummaryWriter
            os.makedirs('TB_logs', exist_ok = True)
            dir = 'TB_logs/'+str(start_time)
            self.writer = SummaryWriter(log_dir = dir)
        self.use_latent = use_latent
        self.CFG = CFG
        # Random seed
        CFG.seed += 10000 * proc_id()
        torch.manual_seed(CFG.seed)
        np.random.seed(CFG.seed)
        # Instantiate environment
        self.env = env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape
        if (use_latent):
            obs_size = self.CFG.latent_dim + self.env.observation_space.shape[0]
        else:
            obs_size = self.env.observation_space.shape[0]

        # Create actor-critic module
        self.ac = actor_critic(obs_size, self.env.action_space, **ac_kwargs)
        # Sync params across processes
        sync_params(self.ac)
        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
        # Set up experience buffer
        self.local_steps_per_epoch = int(CFG.steps_per_epoch / num_procs())
        latent_dim = None
        if (use_latent):
            latent_dim = self.CFG.latent_dim
        self.buf = PPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, CFG.gamma, CFG.lam,latent_dim)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.CFG.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.CFG.vf_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        self.KL_his = [[] for i in range(self.CFG.train_pi_iters)]
        self.maximum_ent = -np.inf

    def assign_writer(self,writer):
        self.writer = writer

    def tensor(self,x):
        return torch.as_tensor(x, dtype=torch.float32)

    def compute_loss_pi(self,data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        if (self.use_latent):
            latent = data['latent']
            input = torch.cat((self.tensor(obs),self.tensor(latent)),dim = 1)
        else:
            input = obs
        # Policy loss
        pi, logp = self.ac.pi(input, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.CFG.clip_ratio, 1+self.CFG.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.CFG.clip_ratio) | ratio.lt(1-self.CFG.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data,log_ = False,epoch_ = None):
        obs, ret = data['obs'], data['ret']

        if (self.use_latent):
            latent = data['latent']
            input = torch.cat((self.tensor(obs),self.tensor(latent)),dim = 1)
        else:
            input = obs

        target_value = data['target_value']
        value_function = self.ac.v(input)
        variance1 = torch.var(target_value-value_function)
        variance2 = torch.var(target_value)
        v_info = dict(value_residual_variance = variance1/variance2)
        if (log_):
            val = data['val'].detach().numpy()
            self.writer.add_scalar('value function/network value mean',np.mean(val),epoch_)
            self.writer.add_scalar('value function/network value std',np.std(val),epoch_)
            self.writer.add_scalar('value function/network value max',np.max(val),epoch_)
            self.writer.add_scalar('value function/network value min',np.min(val),epoch_)
            self.writer.add_histogram('histogram/network value',val,epoch_)

            adv = data['adv'].detach().numpy()
            self.writer.add_scalar('value function/advantage function mean',np.mean(adv),epoch_)
            self.writer.add_scalar('value function/advantage function std',np.std(adv),epoch_)
            self.writer.add_scalar('value function/advantage function max',np.max(adv),epoch_)
            self.writer.add_scalar('value function/advantage function min',np.min(adv),epoch_)
            self.writer.add_histogram('histogram/advantage function',adv,epoch_)
        return ((value_function - ret)**2).mean(),v_info

    def update(self, epoch_):
        data = self.buf.get()
        target_value = data['target_value'].detach().numpy()
        self.writer.add_scalar('value function/target value mean',np.mean(target_value),epoch_)
        self.writer.add_scalar('value function/target value std',np.std(target_value),epoch_)
        self.writer.add_scalar('value function/target value max',np.max(target_value),epoch_)
        self.writer.add_scalar('value function/target value min',np.min(target_value),epoch_)
        self.writer.add_histogram('histogram/target value',target_value,epoch_)


        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old,_ = self.compute_loss_v(data,True,epoch_)
        v_l_old = v_l_old.item()
        ent_his = []
        # Train policy with multiple steps of gradient descent
        for i in range(self.CFG.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            ent = mpi_avg(pi_info['ent'])
            self.maximum_ent = max(self.maximum_ent,ent)

            self.KL_his[i].append(kl)
            ent_his.append(ent)
            mean_kl = np.mean(self.KL_his[i])
            max_kl = np.max(self.KL_his[i])
            min_kl = np.min(self.KL_his[i])
            std_kl = np.std(self.KL_his[i])
            self.writer.add_scalar('KL/KL divergence_mean',mean_kl,i)
            self.writer.add_scalar('KL/KL divergence_std',std_kl,i)
            self.writer.add_scalar('KL/KL divergence_max',max_kl,i)
            self.writer.add_scalar('KL/KL divergence_min',min_kl,i)

            if kl > 1.5 * self.CFG.target_kl:
                for j in range(i+1,self.CFG.train_pi_iters):
                    self.KL_his[j].append(kl)
                    mean_kl = np.mean(self.KL_his[j])
                    max_kl = np.max(self.KL_his[j])
                    min_kl = np.min(self.KL_his[j])
                    std_kl = np.std(self.KL_his[j])
                    self.writer.add_scalar('KL/KL divergence_mean',mean_kl,j)
                    self.writer.add_scalar('KL/KL divergence_std',std_kl,j)
                    self.writer.add_scalar('KL/KL divergence_max',max_kl,j)
                    self.writer.add_scalar('KL/KL divergence_min',min_kl,j)

                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()
        self.writer.add_scalar('entropy/relative policy entropy',np.mean(ent_his)/self.maximum_ent,epoch_)
        self.writer.add_scalar('entropy/policy entropy',np.mean(ent_his),epoch_)

        self.logger.store(StopIter=i)
        value_residual_variance = []
        # Value function learning
        for i in range(self.CFG.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v,v_info = self.compute_loss_v(data)
            value_residual_variance.append(v_info['value_residual_variance'].detach().numpy())
            loss_v.backward()
            mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()
        self.writer.add_scalar('value function/value_residual_variance',np.mean(value_residual_variance),epoch_)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def train(self):
        start_time = time.time()
        o, ep_ret, ep_len,rewards = self.env.reset(), 0, 0, []
        episode_cnt = 0
        episode_return = []
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.CFG.epochs):
            for t in range(self.local_steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = self.env.step(a)
                rewards.append(r)
                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(o, a, r, v, logp)
                self.logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.CFG.max_ep_len
                terminal = d or timeout
                epoch_ended = t==self.local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
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

                        episode_cnt += 1
                    o, ep_ret, ep_len,rewards = self.env.reset(), 0, 0,[]


            # Save model
            if (epoch % self.CFG.save_freq == 0) or (epoch == self.CFG.epochs-1):
                self.logger.save_state({'env': self.env}, None)

            # Perform PPO update!
            self.update(epoch)

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.CFG.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()
        self.writer.close()

    def load(self, path):
        self.ac = torch.load(path)

    def test(self):
        for ttest in range(5):
            o, ep_ret, ep_len,rewards = self.env.reset(), 0, 0, []
            done = False
            turn = 0
            while( not done):
                self.env.render()
                import time
                time.sleep(0.03)
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                next_o, r, done, _ = self.env.step(a)
                rewards.append(r)
                o = next_o
                turn += 1
                if (turn == 500):
                    done = True
                    print(np.sum(rewards))


def main():
    CFG = PPO_CFG()
    mpi_fork(CFG.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(CFG.exp_name, CFG.seed,'./experiences')
    print(logger_kwargs)
    env = gym.make(CFG.env)

    model = PPO(lambda : gym.make(CFG.env), CFG, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[CFG.hid]*CFG.l),logger_kwargs=logger_kwargs)

    # try:
    #     model.load('./experiences/ppo/ppo_s0/pyt_save/model.pt')
    # except:
    #     print('fail to load model')
    model.train()
    # model.test()

if __name__ == '__main__':
    main()
