import torch
import torch.nn as nn
import logging
import random
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1
    vocab["<UNK>"] = len(vocab)
    #print vocab max index and len
    logging.info("vocab max index: %d, vocab size: %d" % (max(vocab.values()), len(vocab)))
    return vocab
def build_corpus(corpus_path):
    corpus = ""
    with open(corpus_path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def build_sample(sen_len,vocab,corpus):
    #choose start point
    start = random.randint(0,len(corpus)-1-sen_len) #[start, len(corpus)-1-sen_len]
    end = start + sen_len
    window = corpus[start:end]
    target = corpus[end]
    x = [vocab.get(word,vocab["<UNK>"]) for word in window]
    y = vocab.get(target,vocab["<UNK>"])
    return x,y
def build_dataset(sen_len,vocab,corpus,sample_size):
    x_dataset = []
    y_dataset = []
    for i in range(sample_size):
        x,y = build_sample(sen_len,vocab,corpus)
        x_dataset.append(x)
        y_dataset.append(y)
    return torch.LongTensor(x_dataset),torch.LongTensor(y_dataset)

class LanguageModel(nn.Module):
    '''
    x: batch * sen_len
    '''
    def __init__(self,vector_dim,vocab):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+1,vector_dim)
        self.lstm = nn.LSTM(vector_dim,vector_dim,num_layers=1,bidirectional=True,batch_first=True)
        self.classify = nn.Linear(vector_dim*2,len(vocab)+1)
        self.dropout = nn.Dropout(0.1)

    def forward(self,x):
        x = self.embedding(x) # batch * sen_len * vector_dim
        x, h = self.lstm(x) # batch * sen_len * (2*vector_dim)
        x = x[:,-1,:] # batch * (2*vector_dim) 取最后一个词的输出
        # x = h         # batch * 1 * (2*vector_dim)
        # x = self.dropout(x)
        y_pred = self.classify(x) # batch * vocab_size
        return y_pred

def generate_sentence(model,vocab,window_size, max_len,words):
    model.eval()
    reverse_vocab = {v:k for k,v in vocab.items()}
    with torch.no_grad():
        pred_char = ""
        while pred_char !='\n' and len(words)<max_len:
            x = [vocab.get(word,vocab["<UNK>"]) for word in words[-window_size:-1]]
            x = torch.LongTensor(x).view(1,-1) # 1*window_size
            if torch.cuda.is_available():
                x = x.cuda()
            y_pred = model(x) # 1*vocab_size
            y_pred_probs = torch.softmax(y_pred,dim=-1)
            pred_index = sampling_strategy(y_pred_probs)
            pred_char = reverse_vocab[pred_index]
            words+=pred_char
    return words

def sampling_strategy(probs):
    '''
    :param probs: 1*vocab_size: sum = 1
    :return: index : to choose the char index
    '''
    if random.random()>0.1:
        # greedy： 取最大的概率对应的index
        return int(torch.argmax(probs).item())
    else:
        # random: 按照概率分布进行采样
        return int(torch.multinomial(probs,1).item())



logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
def train(corpus_path,vocab_path, prompt="让他在半年之前，就不能做出", save_weight=True):
    corpus = build_corpus(corpus_path)
    vocab = build_vocab(vocab_path)

    epochs = 20
    batch_size = 64
    train_sample = 5000
    char_dim = 256
    window_size = 10
    lr = 0.01
    model = LanguageModel(char_dim,vocab)
    if torch.cuda.is_available():
        print("using gpu")
        model = model.cuda()
    loss_fn = nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    print("start training")
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        x, y = build_dataset(window_size,vocab,corpus,train_sample)
        for batch_index in range(train_sample // batch_size + 1):
            start = batch_index * batch_size
            end = start + batch_size
            if end > train_sample:
                end = train_sample
            x_train = x[start:end]
            y_train = y[start:end]
            if torch.cuda.is_available():
                x_train = x_train.cuda()
                y_train = y_train.cuda()
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = loss_fn(y_pred,y_train)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        #get generate sentence
        result = generate_sentence(model,vocab,window_size,20,prompt)
        #get its ppl in train dataset
        logging.info("epoch: {}, loss: {}, next line: {}".format(epoch, sum(watch_loss) / len(watch_loss), result))

    return

if __name__ == "__main__":
    corpus_path = "../data/corpus.txt"
    vocab_path = "../data/vocab.txt"
    train(corpus_path,vocab_path)
