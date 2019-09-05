'''
####################################################################
# 参考地址：https://github.com/vy007vikas/PyTorch-ActorCriticRL
# Author: Qiang Ye
# Date: April 27, 2018
# author wudong
# date 20190816
# 实现DDPG算法，状态空间和行为空间连续，基于experience replay
######################################################################
'''

from random import random, choice
from gym import Env, spaces
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from core import Transition, Experience, Agent
from utils import soft_update, hard_update
from utils import OrnsteinUhlenbeckActionNoise
from approximator import Actor, Critic

class DDPGAgent(Agent):
    '''使用Actor-Critic算法结合深度学习的个体
        接收一个环境对象，同时接受相关的学习参数
        从env得到state和action的特征数目
        继承自基类Agent
    '''
    def __init__(self, env: Env = None,
                       capacity = 2e6,
                       batch_size = 128,
                       action_lim = 1,
                       learning_rate = 0.001,
                       gamma = 0.999,
                       epochs = 2):
        if env is None:
            raise "agent 缺少环境env"
        super(DDPGAgent, self).__init__(env, capacity)
        self.state_dim = env.observation_space.shape[0] # 状态连续,获取状态维度
        self.action_dim = env.action_space.shape[0] # 行为连续，获取行为维度
        self.action_lim = action_lim # 行为值限制
        self.batch_size = batch_size  # 批学习一次状态转换数量
        self.learning_rate = learning_rate # 学习率
        self.gamma = 0.999 # 衰减因子
        self.epochs = epochs # 统一批状态转换学习的次数
        self.tau = 0.01 # 软拷贝的系数
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim) 

        self.actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                self.learning_rate)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 self.learning_rate)
        # 初始化
        # 将critic和actor网络的参数硬拷贝到目标网络target_critic和target_actor
        hard_update(self.target_actor, self.actor) # 硬拷贝
        hard_update(self.target_critic, self.critic) # 硬拷贝
        return
        
    def get_exploitation_action(self, state):
        '''得到给定状态下依据target_actor网络计算出的行为，不探索
        Args:
            state numpy数组
        Returns:
            action numpy 数组
        '''        
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()
    

    def get_exploration_action(self, state):
        '''得到给定状态下根据actor计算出的带噪声的行为，模拟一定的探索
        Args:
            state numpy数组
        Returns:
            action numpy 数组
        '''
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        new_action = new_action.clip(min = -1*self.action_lim, 
                                     max = self.action_lim)
        return new_action
    
       
    def _learn_from_memory(self):
        '''从experience学习，更新两个网络的参数；
        '''
        # 随机获取记忆里的Transmition
        trans_pieces = self.sample(self.batch_size)  
        s0 = np.vstack([x.s0 for x in trans_pieces])
        a0 = np.array([x.a0 for x in trans_pieces])
        r1 = np.array([x.reward for x in trans_pieces])
        s1 = np.vstack([x.s1 for x in trans_pieces])

        # 优化critic网络参数，最小化loss
        a1 = self.target_actor.forward(s1).detach()
        next_val = torch.squeeze(self.target_critic.forward(s1, a1).detach())
        r1 = torch.from_numpy(r1)
        # r1 = r1.type(torch.DoubleTensor)
        next_val = next_val.type(torch.DoubleTensor)
        y_expected = r1 + self.gamma * next_val
        y_expected = y_expected.type(torch.FloatTensor)

        a0 = torch.from_numpy(a0) # 转换成Tensor
        y_predicted = torch.squeeze(self.critic.forward(s0, a0))

        # 最小化loss，更新critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # 优化actor网络参数，优化的目标是使得Q增大
        pred_a0 = self.actor.forward(s0) # 直接使用a0会不收敛
        #反向梯度下降(梯度上升)，以某状态的价值估计为策略目标函数
        loss_actor = -1 * torch.sum(self.critic.forward(s0, pred_a0))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # 软更新参数，跟新target网络
        soft_update(self.target_actor, self.actor, self.tau) 
        soft_update(self.target_critic, self.critic, self.tau)
        return (loss_critic.item(), loss_actor.item())
    

    def learning_method(self, display = False, explore = True):
        '''
        负责agent与env的交互
        实现一个完整的episode，遍历

        返回
            time_in_episode 当前episode的步数
            total_reward 当前episode的总奖励
        '''
        self.state = np.float64(self.env.reset()) #初始化episode的状态
        time_in_episode, total_reward = 0, 0
        is_done = False #episode的结束标志
        loss_critic, loss_actor = 0.0, 0.0
        while not is_done:
            # add code here
            s0 = self.state #获取当前状态
            if explore:
                a0 = self.get_exploration_action(s0)
            else:
                a0 = self.actor.forward(s0).detach().data.numpy()
                
            s1, r1, is_done, info, total_reward = self.act(a0) #执行动作，并获得观测值，并将当前trans序列s0,a0,r0,s1存储
            if display:
                self.env.render()
            
            if self.total_trans > self.batch_size: #若当前的trans的数量大于batch size，开始基于experience replay的学习
                loss_c, loss_a = self._learn_from_memory()
                loss_critic += loss_c
                loss_actor += loss_a

            time_in_episode += 1
        loss_critic /= time_in_episode
        loss_actor /= time_in_episode
        if display:
            print("{}".format(self.experience.last_episode))
        return time_in_episode, total_reward, loss_critic, loss_actor
    
    
    def learning(self,max_episode_num = 800, display = False, explore = True):
        '''
        迭代所有的episode
        并在训练过程中保存训练结果
        '''
        total_time,  episode_reward, num_episode = 0,0,0
        total_times, episode_rewards, num_episodes = [], [], []
        for i in tqdm(range(max_episode_num)): #遍历所有episode
            time_in_episode, episode_reward, loss_critic, loss_actor = \
                self.learning_method(display = display, explore = explore) #对该episode进行学习
            total_time += time_in_episode #总的步数
            num_episode += 1 #总的episode数
            total_times.append(total_time) #存储每个episode的times步数
            episode_rewards.append(episode_reward) #存储每个episode的reward
            num_episodes.append(num_episode) #存储每个episode的索引
            print("episode:{:3}：loss critic:{:4.3f}, J_actor:{:4.3f}".\
                  format(num_episode-1, loss_critic, -loss_actor))
            if explore and num_episode % 100 == 0: # 保存模型
                self.save_models(num_episode)
        return  total_times, episode_rewards, num_episodes
    
    
    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print("Models saved successfully")

        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print("Models loaded succesfully")
 