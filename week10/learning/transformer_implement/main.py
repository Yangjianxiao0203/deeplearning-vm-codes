import logging
import os

from config import Config
import random
import numpy as np
import torch
# from transformer.Models import Transformer
from transformer_self.Models import Transformer
from loader import load_data,load_vocab
from optimizer import choose_optimizer
from evaluate import Evaluator

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed():
    seed = Config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(config,save=False):
    save_path = config["model_path"]
    writer = SummaryWriter()
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    #load vocab
    vocab = load_vocab(config["vocab_path"])
    #load model
    # model = Transformer(n_src_vocab=len(vocab),n_trg_vocab = len(vocab),src_pad_idx=vocab["[PAD]"],trg_pad_idx=vocab["[PAD]"])
    model = Transformer(config["vocab_size"], config["vocab_size"], 0, 0,
                        d_word_vec=128, d_model=128, d_inner=256,
                        n_layers=1, n_head=2, d_k=64, d_v=64,trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=False
                        )
    #use cuda
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #optimizer
    optimizer = choose_optimizer(config,model)
    #train data
    train_data = load_data(config["train_data_path"],config,logger)
    #evaluator
    evaluator = Evaluator(config,model,logger)
    #loss_fn
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab["[PAD]"])
    #train
    for epoch in range(config["epoch"]):
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info("第{}个epoch".format(epoch+1))
        train_loss = []
        for batch in train_data:
            if cuda_flag:
                batch = [b.cuda() for b in batch]
            input,target,gold = batch
            #input : [batch_size,input_seq_len]
            #target : [batch_size,out_seq_len]
            #gold : [batch_size,out_seq_len]
            optimizer.zero_grad()
            output = model(input,target) #output: [batch x out_seq_len, n_vocab]
            loss = loss_fn(output,gold.view(-1))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        logger.info("第{}个epoch的平均loss为{}".format(epoch+1,np.mean(train_loss)))
        writer.add_scalar("train_loss",np.mean(train_loss),epoch)
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"],"model.pth")
    writer.close()
    if save:
        torch.save(model.state_dict(),model_path)
    return

if __name__ =='__main__':
    main(Config)




