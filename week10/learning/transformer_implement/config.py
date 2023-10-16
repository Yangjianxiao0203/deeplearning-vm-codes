Config = {
    "model_path": "output",

    "input_max_length": 120,
    "output_max_length": 30,
    "vocab_size":6219,
    "vocab_path":"vocab.txt",

    "epoch":200,
    "batch_size": 32,
    "learning_rate":1e-3,
    "optimizer": "adam",

    "seed":42,

    "train_data_path": r"sample_data.json",
    "valid_data_path": r"sample_data.json",

    "beam_size":5,

}