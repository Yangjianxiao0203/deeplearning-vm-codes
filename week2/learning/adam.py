'''
adam : 每次的grad下降用的是前面所有的grad的差值
'''

import numpy as np
import torch.nn as nn
import torch

def adam(weight,grad,t,mt,vt, alpha=1e-3,beta1=0.9,beta2=0.999,eps=1e-8):
    '''
    :param weight: 时间步为t的权重
    :param grad: 时间步为t的时候的梯度
    :param t: 时间步
    :param mt: 时间步为t的时候的梯度们的差值
    :param vt: 时间步为t的时候的梯度们的平方的差值
    :param alpha: 学习率
    :param beta1: 超参数
    :param beta2: 超参数
    :param eps: 超参数
    '''

    t= t+1 #时间步+1

    mtt = beta1 * mt + (1-beta1) * grad #mt 为之前的grad的差值，后面为这次的grad，叠加就是差值，反映了之前所有的grad的差值
    vtt = beta2 * vt + (1-beta2) * grad**2 #vt 为之前的grad的平方的差值，后面为这次的grad，叠加就是差值，反映了之前所有的grad的平方的差值

    mth = mtt / (1 - beta1**t) #mt的修正
    vth = vtt / (1 - beta2**t) #vt的修正

    weight_next = weight - (alpha / (np.sqrt(vth) + eps)) * mth #更新权重

    return weight_next,mtt,vtt #返回更新后的权重，mtt，vtt

#测试理论值和实际torch框架的值
x = np.array([1, 2, 3, 4])  #输入
y = np.array([0.1,-0.1,0.01,-0.01])  #预期输出

class TorchModel(nn.Module):
    def __init__(self, hidden_size):
        super(TorchModel, self).__init__()
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss  #loss采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.layer(x)
        y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


#自定义模型，接受一个参数矩阵作为入参
class DiyModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        y_pred = np.dot(self.weight, x)
        y_pred = self.diy_sigmoid(y_pred)
        if y is not None:
            return self.diy_mse_loss(y_pred, y)
        else:
            return y_pred

    #sigmoid
    def diy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #手动实现mse，均方差loss
    def diy_mse_loss(self, y_pred, y_true):
        return np.sum(np.square(y_pred - y_true)) / len(y_pred)

    #手动实现梯度计算
    def calculate_grad(self, y_pred, y_true, x):
        #前向过程
        # wx = np.dot(self.weight, x)
        # sigmoid_wx = self.diy_sigmoid(wx)
        # loss = self.diy_mse_loss(sigmoid_wx, y_true)
        #反向过程
        # 均方差函数 (y_pred - y_true) ^ 2 / n 的导数 = 2 * (y_pred - y_true) / n
        grad_loss_sigmoid_wx = 2/len(x) * (y_pred - y_true) # (n, 1)
        # sigmoid函数 y = 1/(1+e^(-x)) 的导数 = y * (1 - y)
        grad_sigmoid_wx_wx = y_pred * (1 - y_pred) # (n, 1)
        # wx对w求导 = x
        grad_wx_w = x   # (n, m)
        #导数链式相乘
        grad = grad_loss_sigmoid_wx * grad_sigmoid_wx_wx # (n, 1)
        grad = np.dot(grad.reshape(len(x),1), grad_wx_w.reshape(1,len(x)))
        return grad

def diy_sgd(grad, weight, learning_rate):
    return weight - learning_rate * grad