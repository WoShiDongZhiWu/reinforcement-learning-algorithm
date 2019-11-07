'''
############################################
# date 20190806
# 功能
#   实现一个Agent类
#   实现反向SARSA(λ)算法，model-free 
#       on-policy control 
#   针对离散行为空间和离散状态空间
# https://zhuanlan.zhihu.com/p/28180443
############################################
'''

from random import random
from gym import Env
import gym
from gridworld import *


class SarsaLambdaAgent():
    def __init__(self,env:Env):
        self.env = env #个体所在的环境
        self.Q = {} # 个体维护的一个action-value 表
        self.E = {} # Eligibility Trace
        self.state = None #个体当前的观测(状态)
        self._init_agent()
        return
    def _init_agent(self):
        self.state = self.env.reset()
        s_name = self._name_state(self.state)
        self._assert_state_in_QE(s_name, randomized = False)

    def _curPolicy(self,s,episode_num,use_epsilon): #策略函数
        epsilon = 1.00 / (episode_num + 1) #GLIE策略，使用衰减的ε-greedy算法 
        Q_s = self.Q[s]
        rand_value = random()
        action = None
        if use_epsilon and rand_value <epsilon: #epsilon的概率随机探索
           return self.env.action_space.sample()
        else: #使用greedy，选取最优（使Q最大的action）
            return int(max(Q_s,key=Q_s.get))
            

    def performPolicy(self,s,episode_num,use_epsilon=True):  # 执行一个策略
        return self._curPolicy(s,episode_num,use_epsilon)

    def act(self, a): #执行一个action
        return self.env.step(a)

    # 反向sarsa(λ) learning-针对离散观测空间和离散行为空间
    def learning(self,lambda_, gamma, alpha, max_episode_num):
        total_time, time_in_episode, num_episode = 0,0,0
        while num_episode< max_episode_num: #设置终止条件
            self._resetEValue() #E只在一个episode中起作用
            self.state = self.env.reset() #环境初始化
            s0 = self._name_state(self.state) # 获取个体对观测的命名
            a0 = self.performPolicy(s0,num_episode) #得到动作
            self.env.render()

            time_in_episode = 0
            is_done = False
            while not is_done: #针对一个episode
                s1, r1, is_done, info = self.act(a0) #执行一个action
                s1 = self._name_state(s1) #获取新状态的命名
                self._assert_state_in_QE(s1,randomized=True)
                # 获取action
                a1 = self.performPolicy(s1, num_episode)
                self.env.render()
                q = self._get_(self.Q,s0,a0)
                q_prime = self._get_(self.Q,s1,a1) 
                td_target = r1 + gamma * q_prime #td目标
                delta = td_target - q

                e = self._get_(self.E,s0,a0) # 获取e的值
                e = e+1 #只给当前状态的E加1
                self._set_(self.E, s0, a0, e) #在更新E之前设置E

                state_action_list = list(zip(self.E.keys(),self.E.values()))
                for s, a_es in state_action_list:
                    for a in range(self.env.action_space.n): #遍历动作空间的所有动作，更新所有的e和q值
                        e_value = a_es[a]
                        old_q = self._get_(self.Q, s, a)
                        new_q = old_q +alpha * delta * e_value
                        new_e = gamma * lambda_ * e_value
                        self._set_(self.Q, s,a,new_q) # 更新q， 对所有的s-a对应的q进行更新
                        self._set_(self.E,s,a,new_e) #更新e，每个动作后，更新所有动作的E值

                if (num_episode+1) ==max_episode_num: #在终端显示最后Episode的信息
                    print("t:%a: s: %a, a: %a, s1 &a"%(time_in_episode,s0,a0,s1))
                
                s0, a0 = s1, a1
                time_in_episode = time_in_episode + 1

            print("episode %a takes %a steps"%(num_episode, time_in_episode))
            total_time = total_time + time_in_episode
            num_episode = num_episode + 1
        return
    
    def _name_state(self, state):
        return str(state)
    
    def _is_state_in_Q(self, s): #判断s的Q值是否存在
        return self.Q.get(s) is not None
    
    def _init_state_value(self, s_name, randomized= True): #初始化某状态的Q值,E
        if not self._is_state_in_Q(s_name):
            self.Q[s_name],self.E[s_name] = {},{}
            for action in range(self.env.action_space.n): #针对所有可能行为
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v
                self.E[s_name][action] = 0.0
    
    def _assert_state_in_QE(self,s,randomized=True): #确保某个状态的Q存在
        if not self._is_state_in_Q(s):
            self._init_state_value(s,randomized)
    
    def _get_(self,QorE,s,a): #获取Q(s,a)
        self._assert_state_in_QE(s,randomized=True)
        return QorE[s][a]
    
    def _set_(self, QorE,s,a,value): #设置Q(s,a),E
        self._assert_state_in_QE(s,randomized=True)
        QorE[s][a] = value

    def _resetEValue(self):
        for value_dic in self.E.values():
            for action in range(self.env.action_space.n):
                value_dic[action] = 0.00

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
    agent = SarsaLambdaAgent(env) 
    env.reset()
    print("learning....")
    agent.learning(lambda_ = 0.01,
                    gamma=0.9,
                    alpha=0.1,
                    max_episode_num=800)

if __name__ == "__main__":
    main()
