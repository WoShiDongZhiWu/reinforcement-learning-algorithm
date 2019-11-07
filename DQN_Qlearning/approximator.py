'''
#################################################################################################
# author wudong
# date 20190812
# 功能 
#   近似函数，对Q值进行估计；利用DL-CNN实现
#################################################################################################
'''

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy

class NetApproximator(nn.Module):
    def __init__(self, input_dim = 1, output_dim = 1, hidden_dim = 32):
        '''近似价值函数
        Args:
            input_dim: 输入层的特征数 int
            output_dim: 输出层的特征数 int
        '''
        super(NetApproximator, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim) #全连接层
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim) #全连接层
    
    
    def _prepare_data(self, x, requires_grad = False):
        '''将numpy格式的数据转化为Torch的Variable
        '''
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x) #从numpy数组中构建张量
        if isinstance(x, int): # x为单个数据，即状态的特征只有一个
            x = torch.Tensor([[x]])
        x.requires_grad_ = requires_grad
        x = x.float()   # 从from_numpy()转换过来的数据是DoubleTensor形式
        if x.data.dim() == 1: #将1维的x转换为2维的，torch nn输入最少为2维
            x = x.unsqueeze(0)
        return x


    def forward(self, x):
        '''前向运算，根据网络输入得到网络输出
            args
                x 描述状态的输入参数x
        '''
        x = self._prepare_data(x) #对描述状态的参数x做处理
        h_relu = F.relu(self.linear1(x)) #
        y_pred = self.linear2(h_relu) #网络预测的输出
        return y_pred

    
    def __call__(self, x):
        '''调用该类时，自动调用该函数
        '''
        y_pred = self.forward(x)
        return y_pred.data.numpy()
        
    def fit(self, x, y, criterion=None, optimizer=None, 
                  epochs=1, learning_rate=1e-4):
        '''通过训练更新网络参数来拟合给定的输入x和输出y
        '''
        if criterion is None: #计算loss，使用MSELoss（均方差损失）
            criterion = torch.nn.MSELoss(size_average = False)
        if optimizer is None: #优化器
            optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        if epochs < 1:
            epochs = 1

        y = self._prepare_data(y, requires_grad = False) #目标数据不需要梯度

        for t in range(epochs):
            y_pred = self.forward(x) # 前向传播
            loss = criterion(y_pred, y) # 计算损失
            optimizer.zero_grad() # 梯度重置，准备接受新梯度值
            loss.backward() # 反向传播
            optimizer.step() # 更新权重
        return loss
    
    
    def clone(self):
        '''返回当前模型的深度拷贝对象
        '''
        return copy.deepcopy(self)
    
