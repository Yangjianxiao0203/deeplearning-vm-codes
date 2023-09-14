from model import RNNModel
from dataLoading import build_vocab, sentence_to_sequence
import numpy as np
import torch
def predict(model_path,model_config_path,vocab_path,input_strings):
    model_config = np.load(model_config_path,allow_pickle=True).item()
    vocab = build_vocab(vocab_path)
    model = RNNModel(
        input_dim= model_config['char_dim'],
        hidden_dim= model_config['hidden_dim'],
        num_layers= model_config['num_layers'],
        vocab=vocab
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for input_string in input_strings:
            x = sentence_to_sequence(input_string,vocab) # (seq_len,)
            x= torch.LongTensor(x).unsqueeze(dim=0) # (1,seq_len)
            x = model.forward(x) # (1,seq_len,2)
            result = x.squeeze() # (seq_len,2)
            result = torch.argmax(result,dim=-1)
            for index,p in enumerate(result):
                if p == 1:
                    print(input_string[index],end=" ")
                else:
                    print(input_string[index],end="")
            print()

if __name__ =='__main__':
    model_path = 'model.pth'
    model_config_path = 'model_config.npy'
    vocab_path = '../data/chars.txt'
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    predict(model_path,model_config_path,vocab_path,input_strings)