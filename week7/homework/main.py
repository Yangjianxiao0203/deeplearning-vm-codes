import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel
from loader import load_data, load_vocab
import torch
import torch.nn as nn
from optimizer import choose_optimizer,choose_loss
from evaluate import Evaluator

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config, verbose=True,save=False):
    # save output
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # load data
    train_data = load_data(config["data_path"], config)
    # load vocab
    vocab = load_vocab(config["vocab_path"])
    # load model
    model = TorchModel(config, vocab_size=len(vocab), class_num=config['class_num'])
    # use gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu is available, moving model to cuda")
        model = model.cuda()
    #choose optimizer
    optimizer = choose_optimizer(config, model)
    #choose loss
    loss_fn = choose_loss(config)
    #evaluate
    evaluator = Evaluator(config, model, logger)
    acc_pre = 0
    #main part
    for epoch in range(config["epoch"]):
        model.train()
        if verbose:
            logger.info("epoch %d begin" % epoch)
        train_loss = []
        for i, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            x,y = batch_data
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y.squeeze())
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if i % int(len(train_data) / 2) == 0 and verbose:
                logger.info("epoch %d batch %d loss %f" % (epoch, i, loss.item()))
        if verbose:
            logger.info("epoch %d mean loss %f" % (epoch, np.mean(train_loss)))
        # evaluate
        acc = evaluator.eval(epoch)
        # save model
        if save and acc > acc_pre:
            torch.save(model.state_dict(), os.path.join(config["model_path"], config["model_type"]))
            acc_pre = acc
    return acc


if __name__ == '__main__':
    main(Config, verbose=True,save=False)
