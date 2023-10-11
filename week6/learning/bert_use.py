import math
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
import random

def build_vocab():
    chars="abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)+1
    return vocab
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #A类样本
    if set("abc") & set(x) and not set("xyz") & set(x):
        y = 0
    #B类样本
    elif not set("abc") & set(x) and set("xyz") & set(x):
        y = 1
    #C类样本
    else:
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

def build_dataset(sample_len,sen_len,vocab):
    dataset_x = []
    dataset_y= []
    for i in range(sample_len):
        x,y = build_sample(vocab,sen_len)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)


class TorchModel(nn.Module):
    def __init__(self, char_dim):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"/Users/jianxiaoyang/Documents/models_hugging_face/bert-base-chinese",
                                              return_dict=False)
        self.classify = nn.Linear(char_dim, 3)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        sequence_out, pooler_out = self.bert(x)
        x = self.classify(pooler_out)
        y = self.activation(x)
        return y
def build_model():
    model = TorchModel(768)
    return model

def evaluate(model, vocab,sample_length):
    model.eval()
    test_x,test_y = build_dataset(200,sample_length,vocab)
    with torch.no_grad():
        y_pred = model(test_x)
        y_pred = torch.argmax(y_pred, dim=1).numpy()
        test_y = test_y.squeeze().numpy()
        acc = np.mean(y_pred == test_y)
        print("acc:{}".format(acc))
    return acc
    # model.eval()
    # total = 200 #测试样本数量
    # x, y = build_dataset(total, sample_length,vocab)   #建立200个用于测试的样本
    # y = y.squeeze()
    # print("A类样本数量：%d, B类样本数量：%d, C类样本数量：%d"%(y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    # correct, wrong = 0, 0
    # with torch.no_grad():
    #     y_pred = model(x)      #模型预测
    #     for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
    #         if int(torch.argmax(y_p)) == int(y_t):
    #             correct += 1   #正样本判断正确
    #         else:
    #             wrong += 1
    # print("正确预测个数：%d / %d, 正确率：%f"%(correct, total, correct/(correct+wrong)))
    # return correct/(correct+wrong)

def main():
    epochs = 15
    batch_size = 20
    train_sample = 1000
    sentence_length = 6
    vocab = build_vocab()
    model = build_model()
    loss_fn = nn.functional.cross_entropy

    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for epoch in range(epochs):
        watch_loss = []
        model.train()
        for batch_index in range(math.ceil(train_sample / batch_size)):
            start = batch_index * batch_size
            end = min((batch_index + 1) * batch_size, train_sample)
            x,y = build_dataset(end - start, sentence_length, vocab)
            optimizer.zero_grad()
            ypred = model(x)
            loss = loss_fn(ypred, y.squeeze(1))
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("epoch:{},loss:{}".format(epoch, sum(watch_loss) / len(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([np.mean(watch_loss),acc])
    return log

if __name__ == '__main__':
    main()