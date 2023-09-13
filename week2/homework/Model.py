import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dim,num_class):
        super(Net, self).__init__()
        self.linear = nn.Linear(input_dim, num_class)   
        self.loss = nn.CrossEntropyLoss()
    
    # 当输入真实标签，返回loss值；无真实标签，返回预测值    
    def forward(self, x,y=None):
        y_pred = self.linear(x)
        print("x.shape:",x.shape)
        print("y_pred.shape:",y_pred.shape)
        #dime=0 是batch_size， dim=1是类别数
        if y is not None:
            print("y.shape:",y.shape)
            res = self.loss(y_pred, y)
        else:
            res = y_pred
        
        print("res:",res)
        return res