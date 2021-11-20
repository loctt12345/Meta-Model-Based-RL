import sys

try:
    sys.path.remove('/home/huy/Desktop/stuff/spinningup')
    sys.path.append("/home/huy/Desktop/github/RL-Projects/model/varibad_for_game")
except:
    pass

from envs.half_cheetah_hfield_env import HalfCheetahHFieldEnv
from envs.normalized_env import normalize
import torch
import numpy as np
env = HalfCheetahHFieldEnv()
print(env.observation_space)
o = env.reset()
# for i in range(1000):
#     print(np.round_(o,2))
#
#     o,_,_,_=env.step(env.action_space.sample())
#     env.render()