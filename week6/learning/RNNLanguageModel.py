import json
import os

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding='utf-8') as f:
        index = 0
        for line in f:
            char = line.strip() #去掉结尾换行符
            if char not in vocab:
                vocab[char] = index + 1
                index += 1
    if '\n' not in vocab:
        vocab['\n'] = len(vocab) + 1
    if '<UNK>' not in vocab:
        vocab['<UNK>'] = len(vocab) + 1
    return vocab

def load_corpus(corpus_path):
    return open(corpus_path, encoding='utf-8').read()

#随机生成一个长度为window_size+1的样本，前n个字作为输入，最后一个字作为输出
def build_sample(corpus, window_size,vocab):
    #随机取到这个window
    start = np.random.randint(0, len(corpus) - window_size)
    end = start + window_size
    # x,y 都转化成序号
    x= []
    y = vocab.get(corpus[end], vocab['<UNK>'])
    for i in range(start,end):
        x.append(vocab.get(corpus[i], vocab['<UNK>']))
    return x,y

def build_dataset(num,vocab,window_size,corpus):
    dataset_x = []
    dataset_y = []
    for i in range(num):
        x,y = build_sample(corpus,window_size,vocab)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

class LanguageModel(nn.Module):
    def __init__(self,vocab,vector_dim):
        super(LanguageModel,self).__init__()
        self.vocab = vocab
        self.vector_dim = vector_dim
        self.embedding = nn.Embedding(len(vocab)+1,vector_dim) #0 for pad, padding的位置不会更新
        self.rnn = nn.RNN(vector_dim,vector_dim,num_layers=2,batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(vector_dim,len(vocab)+1)
    def forward(self,x):
        '''
        x: batch_size * sen_len
        '''
        x = self.embedding(x) # batch_size * sen_len * vector_dim
        x,h = self.rnn(x) # batch_size * sen_len * vector_dim, batch_size * num_layers * vector_dim
        x = x[:,-1,:] # batch_size * vector_dim, it is the same as h[-1]
        x = self.dropout(x)
        y = self.classify(x) # batch_size * len(vocab)
        return y
def build_model(vocab,vector_dim):
    model = LanguageModel(vocab,vector_dim)
    return model

def evaluate(model, criterion, eval_dataset, batch_size):
    model.eval()  # Set the model.py to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Ensure no gradients are calculated
        for i in range(0, len(eval_dataset), batch_size):
            x, y = eval_dataset[i: i + batch_size]
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(eval_dataset)
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity

def train(corpus_path,save_path=None):
    vocab = build_vocab("../data/vocab.txt")
    print("vocab size:",len(vocab))
    corpus = load_corpus(corpus_path)
    epoch_num = 10         #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 128        #每个字的维度
    window_size = 6       #样本文本长度

    model = LanguageModel(vocab,char_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    log=[]
    if torch.cuda.is_available():
        model = model.cuda()
    print("start training...")
    for epoch in range(epoch_num):
        watch_loss = []
        model.train()
        for i in range(train_sample // batch_size):
            x,y = build_dataset(batch_size,vocab,window_size,corpus)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            assert x.max().item() < len(
                vocab) + 1, f"Max index in x: {x.max().item()}, Embedding size: {len(vocab) + 1}"
            optimizer.zero_grad()
            y_pred = model(x)
            l = criterion(y_pred,y)
            l.backward()
            optimizer.step()
            watch_loss.append(l.item())
        print("epoch:{},loss:{}".format(epoch,np.mean(watch_loss)))
        eval_dataset_x,eval_dataset_y = build_dataset(1000,vocab,window_size,corpus)
        loss,ppl = evaluate(model,criterion,(eval_dataset_x,eval_dataset_y),batch_size)
        log.append((loss,ppl))
    print("training finished!")
    if save_path is not None:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join(save_path, base_name)
        torch.save(model.state_dict(),model_path)
        #相关参数写成个字典，保存到一个json文件中
        param = {"char_dim":char_dim,"window_size":window_size,"vocab_path":"../data/vocab.txt"}
        param_path = os.path.join(save_path,base_name.replace("pth","json"))
        json.dump(param,open(param_path,"w"))
    return model,log

def train_all(paths):
    for path in os.listdir(paths):
        if path.endswith("txt"):
            corpus_path = os.path.join(paths, path)
            train(corpus_path,save_path=paths)

def log_result(log):
    import matplotlib.pyplot as plt
    loss = [i[0] for i in log]
    ppl = [i[1] for i in log]
    plt.subplot(211)
    plt.plot(loss)
    plt.legend(["loss"])
    plt.subplot(212)
    plt.plot(ppl)
    plt.legend(['ppl'])
    plt.show()


if __name__ == "__main__":
    # model.py,log = train("../data/corpus.txt")
    # log_result(log)
    train_all("../data/corpus")