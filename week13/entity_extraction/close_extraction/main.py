import os
import numpy as np
import logging
from config import Config
from loader import load_data
from model import *
from evaluator import Evaluator
import torch
from config import Config

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Debug = Config["Debug"]

def choose_optimizer(config,model):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"])
    else:
        raise Exception("optimizer not found")

def train(config,logger,verbose=True):
    train_data = load_data(config["train_path"], config,logger)
    model = TextClassifyBIO(config)
    optimizer = choose_optimizer(config,model)
    evaluator = Evaluator(config, model, logger)
    cuda_flag = torch.cuda.is_available()
    epochs = config["epoch"]
    alpha = config["loss_alpha"]
    loss_fn = torch.functional.F.cross_entropy
    if Debug:
        epochs = 1
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    for epoch in range(epochs):
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch in enumerate(train_data):
            batch_data, text_data = batch
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, attribute_label,bio_label = batch_data # attribute_label: (batch_size, 1) # bio_label: (batch_size,seq_len)
            optimizer.zero_grad()
            attr_pred, bio_pred = model(input_id) # attr: (batch_size, attr_num) # bio: (batch_size,seq_len,bio_len)

            #transform
            attr_pred = attr_pred.view(-1,attr_pred.shape[-1])
            bio_pred = bio_pred.view(-1,bio_pred.shape[-1]) # (batch_size*seq_len,bio_len)
            attribute_label = attribute_label.view(-1) # (batch_size, )
            bio_label = bio_label.view(-1) # (batch_size*seq_len, )

            attr_loss = loss_fn(attr_pred,attribute_label)
            bio_loss = loss_fn(bio_pred,bio_label)
            loss = alpha * attr_loss + (1-alpha) * bio_loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
            if Debug:
                break
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    return model




if __name__ == "__main__":
    train(Config,logger)