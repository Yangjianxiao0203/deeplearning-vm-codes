import torch
from model import Rbert
from config import Config
from loader import load_data
from torch.optim import Adam,SGD
from evaluator import Evaluator
import numpy as np
import logging
import os
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
Debug = Config["Debug"]

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["lr"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

def train(config,logger):
    train_data = load_data(config["train_path"], config,logger)
    model = Rbert(config,logger)
    optimizer = choose_optimizer(config,model)
    evaluator = Evaluator(config,model,logger)
    logger.info("train data loaded, start training")
    epochs = config["epoch"]
    if Debug:
        epochs = 1
    loss_fn = torch.nn.CrossEntropyLoss()
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu can be used, transfer model to gpu")
        model = model.cuda()
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id,e1_mask,e2_mask,attr = batch_data
            attr_pred = model(input_id,e1_mask,e2_mask)
            loss = loss_fn(attr_pred,attr.view(-1))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch index %d, loss %f" % (index, loss))
            if Debug:
                break
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        #evaluate
        evaluator.eval(epoch)

    return model

if __name__ == '__main__':
    model = train(config=Config,logger=logger)
    model_path = os.path.join(Config["model_path"], "epoch_%d.pth" % Config["epoch"])
    torch.save(model.state_dict(), model_path)