from torch.optim import Adam,SGD
import torch.nn as nn
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

def choose_loss(config):
    loss = config["loss"]
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss == "mse":
        return nn.MSELoss()
    elif loss == "bce":
        return nn.BCELoss()
    else:
        raise Exception("No such loss function")