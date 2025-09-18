import glob
import yaml
import json
import random
import torch
import numpy as np
import os.path as op
from pathlib2 import Path

def seq_random_flip(image, event_volume, imu, flows, flip_x_prob=0.5, flip_y_prob=0):
    """ Randomly flip the image, event_volume, and imu.
    Args:
        image: torch.Tensor, shape: (L, 2, H, W).
        event_volume: torch.Tensor, shape: (L, 2num_bin, H, W).
        imu: torch.Tensor, shape: (L, 6).
        flip_x_prob: float, the probability to flip the image and event_volume
            horizontally.
        flip_y_prob: float, the probability to flip the image and event_volume
            vertically.
    Return:
        image: torch.Tensor, shape: (L, 2, H, W).
        event_volume: torch.Tensor, shape: (L, 2num_bin, H, W).
        imu: torch.Tensor, shape: (L, 6).
    """
    if np.random.rand() < flip_x_prob:
        event_volume = torch.flip(event_volume, dims=[-1])
        image = torch.flip(image, dims=[-1])
        flows = torch.flip(flows, dims=[-1])
        imu[:, 0] = -imu[:, 0]
        imu[:, 4] = -imu[:, 4]
        imu[:, 5] = -imu[:, 5]
    if np.random.rand() < flip_y_prob:
        event_volume = torch.flip(event_volume, dims=[-2])
        image = torch.flip(image, dims=[-2])
        flows = torch.flip(flows, dims=[-2])
        imu[:, 1] = -imu[:, 1]
        imu[:, 3] = -imu[:, 3]
        imu[:, 5] = -imu[:, 5]
    return image, event_volume, imu, flows

def apply_illum_augmentation(image, gain_min=0.8, gain_max=1.2, gamma_min=0.8, gamma_max=1.2):
    random_gamma = gamma_min + random.random() * (gamma_max - gamma_min)
    random_gain = gain_min + random.random() * (gain_max - gain_min)
    image_aug = random_gain * torch.pow(image, random_gamma)
    return torch.clamp(image_aug, 0, 1.)

# def transform_gamma_gain_np(image, gamma, gain):
#     # apply gamma change and image gain.
#     #! THIS MIGHT NOT BE NEEDED, AS THE IMAGE IS ALREADY IN 0~1 RANGE
#     # image = (1. + image) / 2.
#     image = gain * torch.pow(image, gamma) 
#     # image = (image - 0.5) * 2.
#     return torch.clamp(image, -1., 1.) 

