data_dir = "./s3/data/Rbert/"
Config = {
    "data_path": data_dir + "triplet_data.json",
    "train_path": data_dir+"train_triplet_data.json",
    "valid_path": data_dir + "valid_triplet_data.json",
    "vocab_path" : data_dir + "chars.txt",
    "schema_path": "schema.json",
    "bert_path": "bert-base-chinese",
    "bert_config_path": "bert_config.json",
    "model_path": "model_output",

    "max_length": 100,
    "batch_size":32,
    "hidden_size":256,
    "lr":5e-5,
    "optimizer":"adam",
    "dropout":0.1,

    "epoch": 15,

    "Debug":False,
}