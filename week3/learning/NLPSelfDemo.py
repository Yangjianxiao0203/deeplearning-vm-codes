import random

import torch
import torch.nn as nn
import numpy as np

'''
prepare step:
'''

def build_vocab():
    key = 0
    base_char = 'a'
    vocab = {}
    while key < 26:
        vocab[chr(ord(base_char) + key)] = key
        key += 1
    vocab['unk'] = key
    return vocab

'''
构造的索引矩阵x必须是个int，或者longint，代表词表中的映射
'''
def build_datasets(vocab:dict,sentence_length,size):
    vocab_list = list(vocab.keys())
    if vocab_list.__contains__('unk'):
        vocab_list.remove('unk')
    xs=[]
    ys=[]
    for _ in range(size):
        cur_str = ""
        for i in range(sentence_length):
            c = random.choice(vocab_list)
            cur_str+=c
        xs.append([vocab.get(s) for s in cur_str])
        if set(cur_str) & set('abc'):
            ys.append(1)
        else:
            ys.append(0)
    return torch.LongTensor(xs),torch.FloatTensor(ys)

'''
nlp model.py
'''
class NLP(nn.Module):
    def __init__(self,vector_dim,sentence_length,vocab):
        super(NLP, self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim) #把原本的每个词都拿vector代替，变成 sen_len * dim
        self.pooling = nn.AvgPool1d(sentence_length)
        self.linear = nn.Linear(vector_dim,1) # 向量到具体的一个值，二分类问题
        self.activation = torch.sigmoid

    def forward(self,x:torch.LongTensor):
        '''
        x 是一个torch tensor long int： (batch_size,sen_len)
        '''
        x=self.embedding(x) # (batch_size,sen_len) -> (batch_size,sen_len, vector_dim)
        x = x.transpose(1,2)
        x= self.pooling(x) # (batch_size,vector_dim,1)
        x=x.squeeze() # (batch_size,vector_dim)

        x= self.linear(x) #(batch_size,1)
        y=self.activation(x)
        return y

def evaluate(model:nn.Module,vocab,sentence_length):
    model.eval()
    # use test data
    x_test,y_test = build_datasets(vocab,sentence_length,size=200)
    print("本次测试集的正样本个数：{}，负样本个数:{}".format(y_test.sum(),200-y_test.sum()))
    right,wrong=0,0
    with torch.no_grad():
        y_pred=model(x_test)
        y_pred=y_pred.squeeze()
        for y_p,y_t in zip(y_pred,y_test):
            if float(y_p) < 0.5 and int(y_t) == 0:
                right += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                right += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(right, right/(right+wrong)))
    return right/(right+wrong)

def plot_log(log):
    import matplotlib.pyplot as plt
    x=range(len(log))
    acc = [l[0] for l in log]
    loss = [l[1] for l in log]
    plt.subplot(211)
    plt.plot(x,acc,label='acc')
    plt.legend()
    plt.subplot(212)
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.show()




def main():
    epochs = 100
    sentence_length = 6
    vector_dim=20
    trainning_size_each_epoch = 500
    batch_size = 32
    learning_rate=0.001
    vocab = build_vocab()

    model = NLP(vector_dim=vector_dim,vocab=vocab,sentence_length=sentence_length)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)

    log = []

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for i in range(trainning_size_each_epoch// batch_size):
            start = i*batch_size
            end = (i+1)* batch_size
            if end > trainning_size_each_epoch:
                end = trainning_size_each_epoch
            x,y=build_datasets(vocab,sentence_length,end-start)
            optim.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.squeeze()

            loss = criterion(y_pred,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("epoch:{}, loss:{}".format(epoch+1,np.mean(watch_loss)))
        acc = evaluate(model,vocab,sentence_length)
        log.append([acc,np.mean(watch_loss)])
    plot_log(log)


if __name__ =='__main__':
    main()