
Config = {
    "model_path": "output",
    "data_path" : "./data/dataset.csv",
    "vocab_path":"./data/chars.txt",
    "model_type": "lstm",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"/Users/jianxiaoyang/Documents/models_hugging_face/bert-base-chinese",
    "seed": 987
}