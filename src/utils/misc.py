import numpy as np
import torch
import random
import os

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def makepath(desired_path,isfile=False):
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path),exist_ok=True)
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path,exist_ok=True)
    return desired_path

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor

def to_tensor(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray).float()
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray