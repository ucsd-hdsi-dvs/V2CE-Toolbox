import torch
import random
import numpy as np
import pandas as pd

# Split the data into train, validation, and test sets
def train_val_test_split(data, train_size=0.8, val_size=0.1, test_size=0.1, seed=2333):
    assert train_size + val_size + test_size == 1, 'Train, validation, and test sizes must sum to 1'
    # Shuffle the data using a fixed seed
    data_type = type(data)
    if data_type == np.ndarray:
        np.random.seed(seed)
        data = np.random.permutation(data)
    elif data_type == list:
        random.seed(seed)
        random.shuffle(data)
    elif data_type == pd.DataFrame:
        data = data.sample(frac=1, random_state=seed)
    elif data_type == torch.Tensor:
        torch.manual_seed(seed)
        data = data[torch.randperm(data.size()[0])]
    else:
        raise TypeError('Data type not supported')
    
    # Split the data
    train_idx = int(train_size * len(data))
    val_idx = int((train_size + val_size) * len(data))
    train, val, test = data[:train_idx], data[train_idx:val_idx], data[val_idx:]
    return train, val, test


# Split the data into train, validation, and test sets
def train_val_split(data, train_size=0.8, val_size=0.2, seed=2333):
    assert train_size + val_size  == 1, 'Train and validation sizes must sum to 1'
    # Shuffle the data using a fixed seed
    data_type = type(data)
    if data_type == np.ndarray:
        np.random.seed(seed)
        data = np.random.permutation(data)
    elif data_type == list:
        random.seed(seed)
        random.shuffle(data)
    elif data_type == pd.DataFrame:
        data = data.sample(frac=1, random_state=seed)
    elif data_type == torch.Tensor:
        torch.manual_seed(seed)
        data = data[torch.randperm(data.size()[0])]
    else:
        raise TypeError('Data type not supported')
    
    # Split the data
    train_idx = int(train_size * len(data))
    train, val = data[:train_idx], data[train_idx:]
    return train, val

def num_trainable_parameters(module):
    """Return the number of trainable parameters in the module"""
    
    trainable_parameters = filter(lambda p: p.requires_grad,
                                  module.parameters())
    return sum([np.prod(p.size()) for p in trainable_parameters])