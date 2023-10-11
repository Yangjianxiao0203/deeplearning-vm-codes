import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np
from config import Config


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        if config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()
        return

    def load(self):
        data = pd.read_csv(self.data_path)
        # transfrom to numpy
        self.data = np.array(data)
        #encode all texts
        for i in range(len(self.data)):
            self.data[i][1] = self.encode_sentence(self.data[i][1])
        # switch two columns
        self.data = self.data[:, [1, 0]]
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["<unk>"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            vocab[token] = index + 1
    vocab['<unk>'] = len(vocab)
    return vocab

def load_data(data_path,config,shuffle=True):
    dg = DataGenerator(data_path,config)
    dl = DataLoader(dg,batch_size=config["batch_size"],shuffle=shuffle)
    return dl

if __name__ == "__main__":
    dg = DataGenerator("./data/dataset.csv",Config)
    for X,y in dg:
        print(len(X))
        print(y)
        break