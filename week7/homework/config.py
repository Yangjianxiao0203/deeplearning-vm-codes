
Config = {
    "model_path": "output",
    "data_path" : "./data/data/dataset.csv",
    "vocab_path":"./data/data/chars.txt",
    "max_length": 20,
    "class_num": 2,

    "model_type": "bert_lstm",
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "pooling_style": "max",

    "epoch": 15,
    "batch_size": 64,
    "learning_rate": 1e-3,

    "optimizer": "adam",
    "loss":"cross_entropy",

    "pretrain_model_path":r"./bert-base-chinese",
    "seed": 987
}