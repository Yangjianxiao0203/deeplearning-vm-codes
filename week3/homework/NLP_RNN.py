'''
homework for week3:
build a model.py with rnn to predict the sentence in a vocab
'''
from data import build_datasets,build_vocab
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


'''
这个模型是取最后一个时间步的输出作为预测值
'''
class NLP(nn.Module):
    def __init__(self,vector_dim,sentence_length,vocab):
        super(NLP, self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim)
        #rnn : batch_size,seq_len,input_size, 她会把seq_len 变成一个个时间步
        self.rnn = nn.RNN(num_layers=1, input_size=vector_dim,hidden_size=vector_dim,batch_first=True)
        # 输出为类别个数
        self.linear = nn.Linear(vector_dim,3)
        self.activation = torch.sigmoid

    def forward(self,x):
        # x: batch_size,sen_len
        x = self.embedding(x) # batch_size,sen_len,vector_dim
        _,x = self.rnn(x) # x: 1,batch_size,vector_dim
        x = x.squeeze() # batch_size,vector_dim
        x = self.linear(x) # batch_size,1
        y = self.activation(x) # batch_size,1
        return y

def draw_log(log):
    x=range(len(log))
    acc = [l[0] for l in log]
    loss = [l[1] for l in log]
    plt.subplot(211)
    plt.plot(x,acc,label='acc')
    plt.legend()
    plt.subplot(212)
    plt.plot(x,loss,label = 'loss')
    plt.legend()
    plt.show()

def main(vocab,x_train,y_train,x_test,y_test):
    epochs = 100
    sentence_length = 10
    vector_dim=20

    lr = 0.005
    batch_size = 128
    N = x_train.shape[0]
    T = x_test.shape[0]

    model = NLP(vector_dim,sentence_length,vocab)
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    # cross entropy 要求true类别必须是int或者long, shape: (N,)
    # 但是y_pred是float，shape: (N,class_num)
    criterion = nn.functional.cross_entropy
    log = []

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for i in range(N//batch_size):
            start = i * batch_size
            end = (i+1) * batch_size
            if end >N:
                end = N
            x_cur = x_train[start:end]
            y_cur = y_train[start:end]

            optim.zero_grad()
            y_pred = model(x_cur)

            loss = criterion(y_pred,y_cur)
            loss.backward()
            optim.step()

            watch_loss.append(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(T//batch_size):
                start = i * batch_size
                end = (i+1) * batch_size
                if end >N:
                    end = N
                x_cur = x_test[start:end]
                y_cur = y_test[start:end]
                y_pred = model(x_cur) # n * class_num
                y_pred_arg = torch.argmax(y_pred,dim=1) # n*1
                correct+= (y_cur == y_pred_arg).sum().item()
                total+= end-start
        acc = correct / total
        log.append([acc,np.mean(watch_loss)])
        print("epoch: {}, loss: {}, correct: {}".format(epoch+1,np.mean(watch_loss),acc))
    draw_log(log)




if __name__ == '__main__':
    vocab = build_vocab()
    x_train, y_train = build_datasets(vocab, 10, 10000) #x_train: 10000,10
    # 看y_train 的类型
    x_test, y_test = build_datasets(vocab, 10, 1000)
    main(vocab,x_train,y_train,x_test,y_test)