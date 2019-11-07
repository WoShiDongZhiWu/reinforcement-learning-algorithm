'''
############################################
# date 20190805
# 功能
#   实现一个Agent类
#   实现SARSA(0)算法，model-free 
#       on-policy control 
#   针对离散行为空间和离散状态空间
# https://zhuanlan.zhihu.com/p/28133594
############################################
'''

from random import random
from gym import Env
import gym
from gridworld import *


class Agent():
    def __init__(self,env:Env):
        self.env = env #个体所在的环境
        self.Q = {} # 个体维护的一个action-value 表
        self.state = None #个体当前的观测(状态)
    
    def performPolicy(self, state): pass # 执行一个策略

    def act(self, a): #执行一个action
        return self.env.step(a)

    def learning(self): pass #学习过程
    
    def _get_state_name(self, state):
        return str(state)
    
    def _is_state_in_Q(self, s): #判断s的Q值是否存在
        return self.Q.get(s) is not None
    
    def _init_state_value(self, s_name, randomized= True): #初始化某状态的Q值
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n): #针对所有可能行为
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v
    
    def _assert_state_in_Q(self,s,randomized=True): #确保某个状态的Q存在
        if not self._is_state_in_Q(s):
            self._init_state_value(s,randomized)
    
    def _get_Q(self,s,a): #获取Q(s,a)
        self._assert_state_in_Q(s,randomized=True)
        return self.Q[s][a]
    
    def _set_Q(self, s,a,value): #设置Q(s,a)
        self._assert_state_in_Q(s,randomized=True)
        self.Q[s][a] = value

    def performPolicy(self,s,episode_num,use_epsilon): #策略函数
        epsilon = 1.00 / (episode_num + 1) #GLIE策略，使用衰减的ε-greedy算法 
        Q_s = self.Q[s]
        str_act = "unknown"
        rand_value = random()
        action = None
        if use_epsilon and rand_value <epsilon: #epsilon的概率随机探索
            action = self.env.action_space.sample()
        else: #使用greedy，选取最优（使Q最大的action）
            str_act = max(Q_s,key=Q_s.get)
            action = int(str_act)
        return action


    # sarsa(0) learning-针对离散观测空间和离散行为空间
    def learning(self, gamma, alpha, max_episode_num):
        total_time, time_in_episode, num_episode = 0,0,0
        while num_episode< max_episode_num: #设置终止条件
            self.state = self.env.reset() #环境初始化
            s0 = self._get_state_name(self.state) # 获取个体对观测的命名
            self._assert_state_in_Q(s0,randomized=True)
            self.env.render() #显示UI界面
            a0 = self.performPolicy(s0,num_episode,use_epsilon=True) #得到动作

            time_in_episode = 0
            is_done = False
            while not is_done: #针对一个episode
                s1, r1, is_done, info = self.act(a0) #执行一个action
                self.env.render() #更新UI界面
                s1 = self._get_state_name(s1) #获取新状态的命名
                self._assert_state_in_Q(s1,randomized=True)
                # 获取action
                a1 = self.performPolicy(s1, num_episode, use_epsilon=True)
                old_q = self._get_Q(s0,a0)
                q_prime = self._get_Q(s1,a1) 
                td_target = r1 + gamma * q_prime #td目标
                new_q = old_q + alpha * (td_target - old_q) # 更新Q值
                self._set_Q(s0, a0, new_q)  #更新Q值

                if (num_episode+1) ==max_episode_num: #在终端显示最后Episode的信息
                    print("t:%a: s: %a, a: %a, s1 &a"%(time_in_episode,s0,a0,s1))
                
                s0, a0 = s1, a1
                time_in_episode = time_in_episode + 1

            print("episode %a takes %a steps"%(num_episode, time_in_episode))
            total_time = total_time + time_in_episode
            num_episode = num_episode + 1
        return
    
def main():
    # 创建一个格子环境
    env = GridWorldEnv(n_width=12, # 水平方向格子数量
                    n_height=12, # 垂直方向格子数量
                    u_size=60, #格子的尺寸
                    default_reward=0, #设置所有默认的即时奖励为0
                    default_type=0) #默认的格子是可进入的
    env.action_space = spaces.Discrete(4) # 设置action的数量  格子世界环境默认使用0表示左，1右，2上，3下，4567斜向
    env.start = (0,0) #起始位置
    env.ends=[(11,11)] #终止位置
    #设置终止位置,以及其余位置的即时奖励
    env.rewards=[(5,1,-100),(5,6,-100),(9,1,-100),(9,6,-100),
                (11,11,100)]  
    env.types = [(6,1,1),(6,2,1),(6,3,1),(6,4,1),(6,5,1),
                (7,10,1),(2,5,1),(3,8,1),(10,6,1),(11,5,1)
                    ] # 设置障碍格子，不可进入
    env.refresh_setting()
    # env = SimpleGridWorld()
    agent = Agent(env) 
    env.reset()
    print("learning....")
    agent.learning(gamma=0.9,
                    alpha=0.1,
                    max_episode_num=800)

if __name__ == "__main__":
    main()
