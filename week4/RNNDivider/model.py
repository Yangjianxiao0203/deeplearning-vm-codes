import torch
import torch.nn as nn
import numpy as np
from week4.learning.dataLoading import load_data


class RNNModel(nn.Module):
    def __init__(self,num_layers,vocab,input_dim,hidden_dim):
        super(RNNModel, self).__init__()
        #padding idx=0 表示后续embedding训练中，索引为0对应的向量不会训练，而索引为其他的向量会训练
        self.embedding = nn.Embedding(len(vocab)+1,input_dim,padding_idx=0)
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.classify = nn.Linear(hidden_dim,2) # 2分类: 分词/不分词
    def forward(self,x):
        x= self.embedding(x) #(batch_size,seq_len)->(batch_size,seq_len,input_dim)
        x,_ = self.rnn(x) # (batch_size,seq_len,input_dim)->(batch_size,seq_len,hidden_dim)
        # _ 表示hidden state, (batch_size,hidden_dim)
        y = self.classify(x) # (batch_size,seq_len,hidden_dim)->(batch_size,seq_len,2)
        return y

def main():
    epochs = 50
    batch_size = 64
    char_dim = 50
    hidden_dim = 150
    num_layers = 2
    max_length = 20
    learning_rate = 1e-3

    model_config = {
        'epochs':epochs,
        'batch_size':batch_size,
        'char_dim':char_dim,
        'hidden_dim':hidden_dim,
        'num_layers':num_layers,
        'max_length':max_length,
        'learning_rate':learning_rate
    }

    data_loader,vocab = load_data(batch_size,max_length)

    model = RNNModel(num_layers,vocab,char_dim,hidden_dim)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    log = []
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for x,y in data_loader:
            '''
            x: (batch_size,seq_len)： 每一行的值代表该字在字表中的索引
            y: (batch_size,seq_len): 每一行的值代表是否该分词,y_true
            '''
            optim.zero_grad()
            y_pred = model(x) # y: (batch_size,seq_len,2)，代表每一行每一个词的分词与不分词的概率
            y_pred = y_pred.view(-1,2) # (batch_size*seq_len,2) 合并，把每一行这个维度去掉，看分词不分词的概率
            y_true = y.view(-1) # (batch_size*seq_len) 合并，把每一行这个维度去掉，看分词不分词的真实值
            loss = criterion(y_pred,y_true)
            loss.backward()
            optim.step()

            watch_loss.append(loss.item())
        # 跑一遍样本，看看模型的效果
        model.eval()
        with torch.no_grad():
            correct=0
            total = 0
            for x,y in data_loader:
                y_pred = model(x)
                y_pred = y_pred.view(-1,2)
                y_true = y.view(-1)
                y_pred = torch.argmax(y_pred,dim=1)
                correct += torch.sum(torch.eq(y_pred,y_true)).item()
                total += len(y_true)
        acc = correct/total
        log.append([acc,np.mean(watch_loss)])
        print('epoch: {}, acc: {}, loss: {}'.format(epoch+1,acc,np.mean(watch_loss)))

    # 保存模型
    torch.save(model.state_dict(),'./model.pth')
    # 保存模型配置
    np.save('./model_config.npy',model_config)
    return

if __name__ == "__main__":
    main()


