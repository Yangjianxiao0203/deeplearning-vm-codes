import torch
import numpy as np
import random
import torch.nn as nn
# 设置种子
def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # when using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

linear1 = nn.Linear(2,2)
linear2 = nn.Linear(2,2)

print("linear1 weight",linear1.weight)
print("linear2 weight",linear2.weight)