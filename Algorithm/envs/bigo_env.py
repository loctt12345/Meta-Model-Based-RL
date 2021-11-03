import numpy as np
import gym
from gym import spaces
from copy import deepcopy

class customEnv_train(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, df, init_round, init_money, len_his):
        super(customEnv_train, self).__init__()
        self.df = df[:-78700].reset_index(drop = True)
        self.init_round = init_round
        self.init_money = init_money
        self.len_his = len_his
        
        Days = self.df['Day'].unique()
        tmp_day = []
        for Day in Days:
            df_tmp = self.df[self.df['Day'] == Day]
            if (df_tmp.shape[0]<init_round):
                continue
            tmp_day.append(Day)

        Days = np.asarray(tmp_day)
        self.Days = Days
        
        self.Hmax = 1000  # max bet
        self.goal = self.init_money * 10 # goal money
        self.reward_factor = 1e-5
        self.penalty = -1e-3
        self.Top5_factor = np.asarray([100000, 150000,250000, 450000, 50000,50000,50000,50000])
        low  = np.float32(np.array([0] + ([0,0,0,0,0,0,0,0] + [0,0,0,0,0,0,0,0] + [0,0,0,0,0] + [-1])*self.len_his))
        high = np.float32(np.array([1] + ([1,1,1,1,1,1,1,1] + [1,1,1,1,1,1,1,1] + [1,1,1,1,1] + [ 1])*self.len_his))
        self.action_space = spaces.Box(low=np.float32(np.full((8), -1)), high=np.float32(np.full((8), 1)))
        self.observation_space = spaces.Box(low=low, high=high)
    
    def reset(self):
        self.Day = np.random.choice(self.Days)
        self.data = self.df[self.df['Day'] == self.Day].reset_index(drop = True)
        self.current_round = self.init_round + np.random.choice(np.arange(len(self.data) - self.init_round))
        self.current_money = self.init_money
        self.create_state()
        return np.asarray(self.state, dtype = np.float32)
    
    def step(self, actions):
        actions = np.asarray(actions)
        actions = actions/2 + 0.5
        scaled_actions = np.asarray(actions * self.Hmax,dtype = int)
        argsort_actions = np.argsort(scaled_actions)
        scaled_actions = scaled_actions - scaled_actions % 100
        results = int(self.data[self.data.index == self.current_round]['Result'])
        if (results == 9):
            results = [0, 1, 2, 3]
        elif (results == 10):
            results = [4, 5, 6, 7]
        else:
            results = [results - 1]
        argsort_actions = argsort_actions[2:]
        before = self.current_money
        for action in argsort_actions:
            self.current_money -= scaled_actions[action]
            if (action not in results):
                continue
            self.current_money += self.cal_win(action, scaled_actions[action])
        after = self.current_money
        reward = (after - before) * self.reward_factor + self.penalty
        
        self.update_state(actions, results, reward)
        self.current_round += 1
        
        done = False
        info = {}
        if (self.current_round == self.data.shape[0]):
            done = True
            info['finish'] = 'end of day'
            
        if (self.current_money <= 0):
            done = True
            info['finish'] = 'out of money'
        
        return np.asarray(self.state,dtype = np.float32), reward, done, info
        
        
    def create_one_hot(self,results):
        res = [0,0,0,0,0,0,0,0]
        for result in results:
            res[result] = 1
        return res
    
    def get_top5(self, idx):
        
        if (self.data.iloc[idx]['Result'] > 8):
            return [0,0,0,0,0]
        
        Top5 = []
        for i in range(1,6):
            top = f'Top{i}'
            string = self.data.iloc[idx][top]
            string = string.replace('[','')
            string = string.replace(']','')
            value = float(string.split(',')[-1])
            Top5.append(value)
        Top5 = list(Top5/ self.Top5_factor[self.data.iloc[idx]['Result'] - 1])
        for i in range(len(Top5)):
            if Top5[i]>1:
                Top5[i] = 1.0
        return Top5
    
    def create_state(self):
        self.state = [self.current_money/self.goal]
        for idx in range(self.current_round-10, self.current_round):
            results = self.data.iloc[idx]['Result']
            if (results == 9):
                results = [0, 1, 2, 3]
            elif (results == 10):
                results = [4, 5, 6, 7]
            else:
                results = [results - 1]
                
            one_hot_result = self.create_one_hot(results)
            actions = [0,0,0,0,0,0,0,0]
            Top5 = self.get_top5(idx)
            data = one_hot_result+actions+Top5+[self.penalty]
            self.state = self.state + data
        
    
    def update_state(self,actions, results, reward):
        self.state[0] = self.current_money / self.goal
        self.state[1:-22] = self.state[23:]
        self.state[-22:-14] = self.create_one_hot(results)
        self.state[-14:-6] = actions
        self.state[-6:-1] = self.get_top5(self.current_round)
        self.state[-1:] = [reward]
        
    
    def cal_win(self, action, money):
        if (action + 1 in [5, 6, 7, 8]):
            return money * 5
        if (action + 1 == 1):
            return money * 10
        if (action + 1 == 2):
            return money * 15
        if (action + 1 == 3):
            return money * 25
        if (action + 1 == 4):
            return money * 45