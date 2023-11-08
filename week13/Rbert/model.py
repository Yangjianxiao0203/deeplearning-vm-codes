import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

class Rbert(nn.Module):
    def __init__(self,config,logger):
        '''
        model的目的： 给定一个句子和句子中的两个实体，预测句子中两个实体的关系: entity, entity2 -> predict attribute
        '''
        super(Rbert,self).__init__()
        self.config = config
        self.logger = logger
        self.encoder = BertModel.from_pretrained(config['bert_path'])
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config['dropout'])

        self.e1_fc = nn.Linear(self.hidden_size,self.hidden_size)
        self.e2_fc = nn.Linear(self.hidden_size,self.hidden_size)
        self.cls_fc = nn.Linear(self.hidden_size,self.hidden_size)
        self.classify = nn.Linear(self.hidden_size*3,config['num_labels'])

    def forward(self,x,e1_mask,e2_mask):
        '''
        x: batch * sen_len
        e1_mask: batch * sen_len
        e2_mask: batch * sen_len
        '''

        # bert encoder
        x_encode = self.encoder(x)
        x = x_encode.last_hidden_state # batch, sen_len, hidden_size, for whole sequence
        h = x_encode.pooler_output # batch, hidden_size, for [CLS]

        x1_avg = self._entity_average(x,e1_mask) # batch, hidden_size, for entity1
        x2_avg = self._entity_average(x,e2_mask) # batch, hidden_size, for entity2
        #dropout
        h = self.dropout(h)
        x1_avg = self.dropout(x1_avg)
        x2_avg = self.dropout(x2_avg)

        # pass through linear layer
        h = self.cls_fc(h) # batch, hidden_size
        x1_avg = self.e1_fc(x1_avg) # batch, hidden_size
        x2_avg = self.e2_fc(x2_avg) # batch, hidden_size
        #concat
        concat = torch.cat([h,x1_avg,x2_avg],dim=-1) # batch, hidden_size*3
        logits = self.classify(concat) # batch, num_labels
        return logits

    def _entity_average(self,x,mask):
        '''
        Args:
            x: batch * sen_len * hidden_size
            mask: batch * sen_len  all 0,1
        return:
            avg: batch * hidden_size 代表平均下来的实体向量，只有一个token
        '''
        mask = mask.float()

        sum_vector = torch.einsum("ijk,ij->ik",x,mask) # batch * hidden_size
        count = torch.sum(mask,dim=-1,keepdim=True) # batch * 1

        avg = sum_vector / count # batch * hidden_size
        return avg