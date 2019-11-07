'''
#################################################################################################
# author wudong
# date 20190813
# 功能 
#   测试DQN算法,状态空间连续，行为空间离散
#################################################################################################
'''

import gym
from puckworld import PuckWorldEnv
from agent import DQNAgent
from utils import learning_curve
from gridworld import *

env = PuckWorldEnv()
agent = DQNAgent(env)

data = agent.learning(gamma=0.99,
                    epsilon=1,
                    decaying_epsilon=True,
                    alpha= 1e-3,
                    max_episode_num=100,
                    display=True)

learning_curve(data,2,1,title="DQN",x_name="episodes",
                y_name="rewards")