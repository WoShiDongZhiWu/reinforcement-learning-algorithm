'''
#################################################################################################
# author wudong
# date 20190812
# 功能 
#   实现抽象的Agent基类:符合gym接口规范
#################################################################################################
'''

from random import random, choice
import gym
from gym import Env
import numpy as np
from typing import List
import random
from tqdm import tqdm

class Agent(object):
    '''
    个体基类，没有学习能力
    具备执行策略、执行行为、学习、
    记忆一定数量的经历过的状态转换对象
    以及从记忆中随机获取一定数量的状态转换对象以供批量学习的功能
    '''
    def __init__(self, env: Env = None, 
                       capacity = 10000):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env # 建立对环境对象的引用
        self.obs_space = env.observation_space if env is not None else None
        self.action_space = env.action_space if env is not None else None
        self.S = None
        self.A = None
        if type(self.obs_space) in [gym.spaces.Discrete]:
            self.S = [i for i in range(self.obs_space.n)]
        if type(self.action_space) in [gym.spaces.Discrete]:
            self.A = [i for i in range(self.action_space.n)]
        self.experience = Experience(capacity = capacity)
        # 有一个变量记录agent当前的state相对来说还是比较方便的。要注意对该变量的维护、更新
        self.state = None   # 个体的当前状态
    
    def policy(self, A, s = None, Q = None, epsilon = None):
        '''均一随机策略(不具有学习能力)
        '''
        return random.sample(self.A, k=1)[0] #从A中随机获取1个元素
    
    def perform_policy(self, s, Q = None, epsilon = 0.05):
        '''
        执行策略
        '''
        action = self.policy(self.A, s, Q, epsilon)
        return int(action) #返回动作
    
    def act(self, a0):
        '''
        执行action，并观测
        '''
        s0 = self.state #当前状态
        s1, r1, is_done, info = self.env.step(a0) #执行动作，观测
        # TODO add extra code here
        trans = Transition(s0, a0, r1, is_done, s1) #存储当前序列
        total_reward = self.experience.push(trans) #将当前序列存在experience中，在experience replace时用
        self.state = s1 #更新当前状态
        return s1, r1, is_done, info, total_reward

    def learning_method(self,lambda_ = 0.9, gamma = 0.9, alpha = 0.5, epsilon = 0.2, display = False):
        '''
        一个没有学习能力的学习方法
        具体针对某算法的学习方法，返回值需是一个二维元组：(一个状态序列的时间步、该状态序列的总奖励)
        '''
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0, epsilon)
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            # add code here
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, epsilon)
            # add your extra code here

            s0, a0 = s1, a1
            time_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return time_in_episode, total_reward  
     
    
    def _decayed_epsilon(self,cur_episode: int, 
                              min_epsilon: float, 
                              max_epsilon: float, 
                              target_episode: int) -> float: # 该episode及以后的episode均使用min_epsilon
        '''
        获得一个在一定范围内的epsilon
        '''
        slope = (min_epsilon - max_epsilon) / (target_episode)
        intercept = max_epsilon
        return max(min_epsilon, slope * cur_episode + intercept)        
                        
    def learning(self, lambda_ = 0.9, epsilon = None, decaying_epsilon = True, gamma = 0.9, 
                 alpha = 0.1, max_episode_num = 800, display = False, min_epsilon = 1e-2, min_epsilon_ratio = 0.8):
        '''学习算法，遍历所有episode进行学习
        '''
        total_time,  episode_reward, num_episode = 0,0,0
        total_times, episode_rewards, num_episodes = [], [], []
        for i in tqdm(range(max_episode_num)): #遍历所有episode
            if epsilon is None:
                epsilon = 1e-10
            elif decaying_epsilon: #使用衰减的epsilon策略
                #epsilon = 1.0 / (1 + num_episode)
                epsilon = self._decayed_epsilon(cur_episode = num_episode+1,
                                                min_epsilon = min_epsilon,
                                                max_epsilon = 1.0,
                                                target_episode = int(max_episode_num * min_epsilon_ratio))
            time_in_episode, episode_reward = self.learning_method(lambda_ = lambda_, \
                  gamma = gamma, alpha = alpha, epsilon = epsilon, display = display)
            total_time += time_in_episode
            num_episode += 1
            total_times.append(total_time)
            episode_rewards.append(episode_reward)
            num_episodes.append(num_episode)
        #self.experience.last_episode.print_detail()
        return  total_times, episode_rewards, num_episodes

    def sample(self, batch_size = 64):
        '''
        从记忆中随机取样
        '''
        return self.experience.sample(batch_size)

    @property
    def total_trans(self):
        '''
        得到Experience里记录的总的状态转换数量
        '''
        return self.experience.total_trans
    
    def last_episode_detail(self):
        self.experience.last_episode.print_detail()


class State(object):
    def __init__(self,name):
        self.name = name

  
class Transition(object):
    '''
    状态转换类
    记录个体的当前状态s0，行为a0,执行a0后的即时奖励reward以及新状态s1
    is_done用来判断s1是否为终止状态
    '''
    def __init__(self, s0, a0, reward:float, is_done:bool, s1):
        self.data = [s0, a0, reward, is_done, s1]

    def __iter__(self):
        return iter(self.data)
    
    def __str__(self):
        return 's:%a a:%a r:%a is_end:%a s1:%a'%(self.data[0], self.data[1], self.data[2],
                   self.data[3], self.data[4])

    @property
    def s0(self): return self.data[0]

    @property
    def a0(self): return self.data[1]

    @property
    def reward(self): return self.data[2]
    
    @property
    def is_done(self): return self.data[3]

    @property
    def s1(self): return self.data[4]
    

class Episode(object):
    '''
    场景类
    记录一系列的Episode，这些Episode由一系列的有序Transition对象组成
    '''
    def __init__(self, e_id:int = 0) -> None:
        self.total_reward = 0   # 总的获得的奖励
        self.trans_list = []    # 状态转移列表
        self.name = str(e_id)   # 可以给Episode起个名字："成功闯关,黯然失败？"

    def push(self, trans:Transition) -> float: #定义返回值的type
        '''
        将一个状态转换送入状态序列中，并返回该序列当前总的奖励值
        '''
        self.trans_list.append(trans)
        self.total_reward += trans.reward # 不计衰减的总奖励
        return self.total_reward

    @property
    def len(self):
        return len(self.trans_list)

    def __str__(self):
        return "episode %a %a steps,total reward:%a" %(self.name, self.len,self.total_reward)

    def print_detail(self):
        print("detail of ({0}):".format(self))
        for i,trans in enumerate(self.trans_list):
            print("step{0:<4} ".format(i),end=" ")
            print(trans)

    def pop(self) -> Transition:
        '''normally this method shouldn't be invoked.
        '''
        if self.len > 1:
            trans = self.trans_list.pop()
            self.total_reward -= trans.reward
            return trans
        else:
            return None

    def is_complete(self) -> bool:
        '''
        判断当前状态序列是否是一个完整的状态序列
        '''
        if self.len == 0: 
            return False 
        return self.trans_list[self.len-1].is_done

    def sample(self, batch_size = 1):   
        '''
        从当前状态序列中随机产生一定数量的不连续的状态转换
        '''
        return random.sample(self.trans_list, k = batch_size)

    def __len__(self) -> int:
        return self.len


class Experience(object):
    '''
    该类用来记录整个experience，由一个agent的episode列表组成
    agent可以从experience中随机采样transitions或episodes 
    '''
    def __init__(self, capacity:int = 20000):
        self.capacity = capacity    # 容量：指的是trans总数量
        self.episodes = []          # episode列表
        self.next_id = 0            # 下一个episode的Id
        self.total_trans = 0        # 总的状态转换数量
        
    def __str__(self):
        return "exp info:{0:5} episodes, memory usage {1}/{2}".\
                format(self.len, self.total_trans, self.capacity)

    def __len__(self):
        return self.len

    @property
    def len(self):
        return len(self.episodes)

    def _remove(self, index = 0):      
        '''扔掉一个Episode，默认第一个。
           args: 
               要移除的episode的索引
           return:
               if exists return the episode else return None
        '''
        if index > self.len - 1:
            raise(Exception("invalid index"))
        if self.len > 0:
            episode = self.episodes[index]
            self.episodes.remove(episode)
            self.total_trans -= episode.len
            return episode
        else:
            return None

    def _remove_first(self):
        self._remove(index = 0)

    def push(self, trans): 
        '''
        压入一个状态转换
        根据当前状态序列是否已经完整将trans加入现有状态序列或开启新的状态序列
        '''
        if self.capacity <= 0:
            return
        while self.total_trans >= self.capacity: # 可能会有空episode吗？
            episode = self._remove_first()
        cur_episode = None
        if self.len == 0 or self.episodes[self.len-1].is_complete():
            cur_episode = Episode(self.next_id)
            self.next_id += 1
            self.episodes.append(cur_episode)
        else:
            cur_episode = self.episodes[self.len-1]
        self.total_trans += 1
        return cur_episode.push(trans)      #return  total reward of an episode

    def sample(self, batch_size=1): # sample transition
        '''randomly sample some transitions from agent's experience.abs
        随机获取一定数量的状态转化对象Transition
        args:
            采样的transitions的数量
        return:
            采样得到的transition列表
        '''
        sample_trans = []
        for _ in range(batch_size):
            index = int(random.random() * self.len)
            sample_trans += self.episodes[index].sample()
        return sample_trans

    def sample_episode(self, episode_num = 1):  # sample episode
        '''随机获取一定数量完整的Episode
        '''
        return random.sample(self.episodes, k = episode_num)

    @property
    def last_episode(self):
        '''
        得到当前最新的一个状态序列
        '''
        if self.len > 0:
            return self.episodes[self.len-1]
        return None