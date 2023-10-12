import torch
import torch.nn as nn
import numpy as np
'''
BERT is a transformers model.py pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:

Masked language modeling (MLM): taking a sentence, the model.py randomly masks 15% of the words in the input then run the entire masked sentence through the model.py and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally masks the future tokens. It allows the model.py to learn a bidirectional representation of the sentence.
Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model.py then has to predict if the two sentences were following each other or not.
'''

class DiyBert(nn.Module):
    '''
    hidden_size: 词向量维度
    attention_heads: 多头注意力的头数
    '''
    def __init__(self,hidden_size,attention_heads):
        super(DiyBert, self).__init__()
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
    def forward(self,x):
        pass

