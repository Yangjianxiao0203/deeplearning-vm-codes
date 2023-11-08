
data_dir = "../../triplet_data/"
Config = {
    "data_path": data_dir + "triplet_data.json",
    "train_path": data_dir+"train_triplet_data.json",
    "valid_path": data_dir + "valid_triplet_data.json",
    "vocab_path" : "chars.txt",
    "schema_path": "schema.json",

    "max_length": 200,
    "batch_size":128,
    "hidden_size":256,
    "lr":1e-4,
    "loss_alpha":0.5,
    "optimizer":"adam",
    "dropout":0.1,

    "epoch": 15,

    "Debug":True,
}