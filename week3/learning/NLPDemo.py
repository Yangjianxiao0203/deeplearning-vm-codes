import random
import numpy as np
import torch
import torch.nn as nn


class NLP(nn.Module):
    '''
    二分类问题
    vocab:词表
    sentence_length: 句子长度
    vector_dim: 词向量维度
    '''

    def __init__(self, vector_dim, sentence_length, vocab):
        super(NLP, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层，将词转换成向量，embedding权重的维度为（词表大小，词向量维度）
        self.pooling = nn.AvgPool1d(sentence_length)  # 池化层，按句子长度的维度进行平均，得到一个词向量，不管句子的长度，最后维度都是 1*词向量维度
        #线性层这里可以换成RNN
        self.linear = nn.Linear(vector_dim, 1)  # 线性层，将词向量维度转换成1维，即判断是否有某个词出现
        self.activation = torch.sigmoid  # 激活函数，将线性层的输出转换成0-1之间的数值
        # self.loss = nn.functional.mse_loss # 损失函数，均方差损失,分类任务只有标签是0或1时才能用这个损失函数

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pooling(x)  # (batch_size, vector_dim, sen_len) -> (batch_size, vector_dim, 1)
        x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.linear(x)  # (batch_size, vector_dim) -> (batch_size, 1)
        return self.activation(x)  # (batch_size, 1) -> (batch_size, 1)


def build_vocab():
    '''
    构建词表
    :return:
    '''
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index
    vocab['unk'] = len(vocab)  # 第26是unknown
    return vocab


def build_str_sample(vocab, sentence_length):
    x = ""
    vocab_keys = list(vocab.keys())
    if vocab_keys.__contains__('unk'):
        vocab_keys.remove('unk')
    for _ in range(sentence_length):
        x += random.choice(vocab_keys)
    return x


def str_to_sequence(vocab, str):
    suquence = [vocab.get(c, vocab['unk']) for c in str]
    return suquence


def build_sample(vocab, sentence_length):
    str = build_str_sample(vocab, sentence_length)
    x = str_to_sequence(vocab, str)
    y = 0
    if set("abc") & set(str):
        y = 1
    return str, x, y


def build_dataset(vocab, sentence_length, size):
    '''
    x: size x sen_len
    y: size
    '''
    xs = []
    ys = []
    for _ in range(size):
        str, x, y = build_sample(vocab, sentence_length)
        xs.append(x)
        ys.append(y)
    return torch.LongTensor(xs), torch.FloatTensor(ys)


def evaluate(model: nn.Module, vocab, sentence_length):
    model.eval()
    x,y = build_dataset(vocab, sentence_length, 200)
    print("本次测试集的正样本个数：{}，负样本个数:{}".format(y.sum(),200-y.sum()))
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred,y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def draw_plot(log):
    import matplotlib.pyplot as plt
    x=range(len(log))
    acc = [l[2] for l in log]
    loss = [l[1] for l in log]
    plt.subplot(211)
    plt.plot(x,acc,label='acc')
    plt.legend()
    plt.subplot(212)
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.show()

def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.01  # 学习率
    vocab = build_vocab()

    # build model.py
    model = NLP(char_dim, sentence_length, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.functional.mse_loss

    log = []

    # train
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(0, train_sample // batch_size):
            x, y = build_dataset(vocab, sentence_length, batch_size)
            optim.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.squeeze()
            loss_val = loss(y_pred, y)
            loss_val.backward()
            optim.step()
            watch_loss.append(loss_val.item())
        print("epoch:{},loss:{}".format(epoch+1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([epoch, np.mean(watch_loss), acc])
        print("="*30)
    draw_plot(log)


if __name__ == '__main__':
    main()
