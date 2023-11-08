import os
if 'SAGEMAKER_TRAINING_MODULE' in os.environ:
    data_path = "/opt/ml/input/data/train"
    vocab_path = "/opt/ml/input/data/vocab"
    pretrained_model_path = "/opt/ml/input/data/pretrained_model"
else:
    data_path = "./data/data/dataset.csv"
    vocab_path = "./data/data/chars.txt"
    pretrained_model_path = r"bert-base-chinese"

Config = {
    "model_path": "output",
    "data_path" : data_path,
    "vocab_path":vocab_path,
    "max_length": 20,
    "class_num": 2,

    "model_type": "cnn",
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "pooling_style": "max",

    "epoch": 15,
    "batch_size": 64,
    "learning_rate": 1e-3,

    "optimizer": "adam",
    "loss":"cross_entropy",

    "pretrain_model_path": pretrained_model_path,
    "seed": 987
}