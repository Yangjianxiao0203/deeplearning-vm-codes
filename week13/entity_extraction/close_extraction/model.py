import torch
import torch.nn as nn
from torch.optim import Adam, SGD

class TextClassifyBIO(nn.Module):
    def __init__(self,config):
        '''
        用于同时进行文本分类和序列标注问题，统一算损失
        '''
        super(TextClassifyBIO, self).__init__()
        self.config = config
        self.vocab_size = config["vocab_size"] + 1 # padding
        self.hidden_size = config["hidden_size"]
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(self.hidden_size,self.hidden_size,bidirectional=True,bias=False,batch_first=True)
        self.dropout = nn.Dropout(config["dropout"])
        self.bio_classifier = nn.Linear(self.hidden_size*2,config["bio_count"])
        self.attribute_classifier = nn.Linear(self.hidden_size*2,config["attribute_count"])

    def forward(self,x):
        '''
        x: batch_size, seq_len
        '''
        x = self.embedding(x) # batch_size, seq_len, hidden_size
        # text network
        x, _ = self.lstm(x) # batch_size, seq_len, hidden_size*2
        x = self.dropout(x)
        # bio network: 对每个token进行bio分类，所以不需要pooling
        bio_predict = self.bio_classifier(x) # batch_size, seq_len, bio_count
        # pooling
        self.pooling = nn.AvgPool1d(x.shape[1])
        x = self.pooling(x.transpose(1,2)).squeeze() # batch_size, hidden_size*2
        # attribute network
        attribute_predict = self.attribute_classifier(x) # batch_size, attribute_count

        return attribute_predict, bio_predict