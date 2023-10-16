import torch
from torch.utils.data import Dataset, DataLoader
import json

def load_vocab(vocab_path):
    vocab = {}
    with open (vocab_path, encoding="utf8") as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i
    return vocab

class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.path = data_path
        self.config = config
        self.logger = logger
        self.vocab = load_vocab(config["vocab_path"])

        self.config["vocab_size"] = len(self.vocab)
        self.config["pad_idx"] = self.vocab["[PAD]"]
        self.config["start_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]

        self.load()

    def load(self):
        '''
        load data
        '''
        self.data = []
        with open (self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self._prepare_data(title,content)

    def _prepare_data(self,title,content):
        # title: decoder, content:encoder
        '''
        target: [CLS] title : 代表模型要依照他一个个生成的，一开始就一个CLS进去，然后生成第一个字，再生成第二个字，以此类推，最后生成的和gold算误差
        '''
        self.input = self._encode_sentence(content, self.config["input_max_length"], False, False)
        self.target = self._encode_sentence(title, self.config["output_max_length"], True, False)
        self.gold = self._encode_sentence(title, self.config["output_max_length"], False, True)
        self.data.append([torch.LongTensor(self.input), torch.LongTensor(self.target), torch.LongTensor(self.gold)])

    def _encode_sentence(self, text, max_len, cls = True, sep = False):
        input_encode = []
        if cls:
            input_encode.append(self.vocab["[CLS]"])
        for char in text:
            input_encode.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if sep:
            input_encode.append(self.vocab["[SEP]"])
        input_encode = self._padding(input_encode, max_len)
        return input_encode

    def _padding(self,input_encode,max_len):
        input_encode = input_encode[:max_len]
        input_encode += [self.vocab["[PAD]"]] * (max_len - len(input_encode))
        return input_encode

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path,config,logger,shuffle=True):
    dg = DataGenerator(data_path,config,logger)
    data_loader = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return data_loader

if __name__ =='__main__':
    from config import Config
    dl = load_data(Config["train_data_path"],Config,None)
    for i, (input, target, gold) in enumerate(dl):
        print(input.shape)
        print(target.shape)
        print(gold.shape)
        break