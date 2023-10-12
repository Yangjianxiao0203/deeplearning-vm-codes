
'''
prepare step:
'''
import random

import torch


def build_vocab():
    key = 0
    base_char = 'a'
    vocab = {}
    while key < 26:
        vocab[chr(ord(base_char) + key)] = key
        key += 1
    vocab['unk'] = key
    return vocab

'''
构造的索引矩阵x必须是个int，或者longint，代表词表中的映射
'''
def build_datasets(vocab:dict,sentence_length,size):
    vocab_list = list(vocab.keys())
    if vocab_list.__contains__('unk'):
        vocab_list.remove('unk')
    xs=[]
    ys=[]
    for _ in range(size):
        cur_str = ""
        for i in range(sentence_length):
            c = random.choice(vocab_list)
            cur_str+=c
        xs.append([vocab.get(s) for s in cur_str])
        if set(cur_str) & set('abc'):
            ys.append(1)
        elif set(cur_str) & set('xyz'):
            ys.append(2)
        else:
            ys.append(0)
    return torch.LongTensor(xs),torch.LongTensor(ys)

'''
nlp model.py
'''
