import os
import random
import numpy as np
import torch

def set_global_seed(seed: int):
    """
    Set random seeds across random, numpy, torch, and CUDA for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Global seed set to {seed}")
