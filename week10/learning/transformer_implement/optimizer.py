import torch
def choose_optimizer(Config,model):
    if Config["optimizer"] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=Config["learning_rate"])
    elif Config["optimizer"] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=Config["learning_rate"])
    else:
        raise Exception("optimizer type error")