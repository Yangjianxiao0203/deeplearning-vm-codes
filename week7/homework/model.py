import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self,config,vocab_size,class_num):
        '''
            "model_type": "lstm",
            "hidden_size": 128,
            "kernel_size": 3,
            "num_layers": 2,
            "pooling_style": "max",
        '''
        super(TorchModel,self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.pooling_style = config["pooling_style"]
        self.model_type = config["model_type"]
        self.class_num = class_num

        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size,self.hidden_size, padding_idx=0)

        if self.model_type == 'lstm':
            self.encoder = nn.LSTM(self.hidden_size,self.hidden_size,num_layers=self.num_layers)
        elif self.model_type =='bert':
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"])
            self.hidden_size = self.encoder.config.hidden_size
        elif self.model_type == 'bert_lstm':
            self.use_bert = True
            self.encoder = BertLSTM(config)
            self.hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(self.hidden_size,self.class_num)

    def forward(self,x):
        if not self.use_bert:
            x = self.embedding(x)
        x = self.encoder(x)
        if isinstance(x,tuple):
            x = x[0] # (batch_size, seq_len, hidden_size)
        elif not isinstance(x,torch.Tensor):
            try:
                x = x.last_hidden_state # for bert
            except:
                raise ValueError('x must be tuple or tensor')
        x = x.transpose(1,2) # (batch_size, hidden_size, seq_len)
        x = self.pooling_layer(x,self.pooling_style) # (batch_size, hidden_size,1)
        x = x.squeeze() # (batch_size, hidden_size)

        y_pred = self.classify(x) # (batch_size, class_num)
        return y_pred


    def pooling_layer(self,x,pool_type):
        if pool_type == 'max':
            return nn.MaxPool1d(x.shape[-1])(x)
        elif pool_type == 'avg':
            return nn.AvgPool1d(x.shape[-1])(x)
        else:
            raise ValueError('pooling type is not supported')

class BertLSTM(nn.Module):
    def __init__(self,config):
        super(BertLSTM,self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        hidden_size = self.bert.config.hidden_size
        self.lstm = nn.LSTM(hidden_size,hidden_size,num_layers=config["num_layers"])

    def forward(self,x):
        x = self.bert(x).last_hidden_state
        x,_ = self.lstm(x)
        return x

if __name__ == "__main__":
    from config import Config
    Config["class_num"] = 3
    Config["vocab_size"] = 20
    Config["max_length"] = 5
    Config["model_type"] = "lstm"
    model = TorchModel(Config,Config["vocab_size"],Config["class_num"])
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    y_pred = model(x)
    print(y_pred)