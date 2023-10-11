import pandas as pd
import numpy as np
from config import Config
data = pd.read_csv(Config["data_path"])
print(data)