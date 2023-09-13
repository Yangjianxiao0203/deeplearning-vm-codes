'''
池化的目的：
原来一个词："abcde"
经过embedding层，假设词表长度1000后，embeding 每个词为100维的向量，变成了一个矩阵，每个字符都有一个向量表示,即5*100
经过池化层，变成了一个向量，即1*100,这个向量就是这个词的向量表示
注意，必须对代表词长度那一维pooling，若多词表长度维求平均，信息就混乱了，每个词不管长度，最后出来的向量表示维度应该是一样的
'''
import torch
import torch.nn as nn

#after embedding
x= torch.rand([3,4,5]) #3条，每条4个词，每个词5维

avg = nn.AvgPool1d(4) # 默认对最后一维求平均
trans_x = x.transpose(1,2)
print("trans_x",trans_x.shape)
y = avg(trans_x)
print("after pooling",y.shape)
y= y.squeeze()
print("after squeezing",y.shape)