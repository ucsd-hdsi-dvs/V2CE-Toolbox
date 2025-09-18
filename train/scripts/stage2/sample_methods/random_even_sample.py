import h5py
import torch
import importlib
import numpy as np
import pandas as pd
import os.path as op
from torch import Tensor,tensor
from typing import List, Tuple
import numpy 
import logging
from typing import List, Tuple

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def pick_elements(ts: Tensor, num_elements: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """ Pick the first `num_elements` events from `ts`, and add their x and y index
    Args:
        ts(H,W,M): timestamps of M events
        num_elements(H,W): number of element to keep in each 1-d array of timestamps of the voxel
    """
    H, W = num_elements.shape
    device = ts.device
    # The selection mask for each pixel
    selection = torch.arange(ts.shape[-1], device=device).unsqueeze(0).unsqueeze(1) < num_elements.unsqueeze(2)
    # Create the row and column indices for events generated
    x_index = torch.arange(W, device=device, dtype=torch.int16).unsqueeze(1).expand(H, W, ts.shape[-1])
    y_index = torch.arange(H, device=device, dtype=torch.int16).unsqueeze(1).unsqueeze(1).expand(H, W, ts.shape[-1])

    return ts[selection], x_index[selection], y_index[selection]
    # return torch.stack([ts[selection], row[selection], column[selection]], -1)

def pick_elements_bn(ts: Tensor, num_elements: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """ Pick the first `num_elements` events from `ts`, and add their x and y index
    Args:
        ts(H,W,M): timestamps of M events
        num_elements(H,W): number of element to keep in each 1-d array of timestamps of the voxel
    """
    # global saved_selection
    H, W = num_elements.shape
    device = ts.device

    selection = torch.bernoulli(num_elements).bool()
    saved_selection = selection
    # Create the row and column indices for events generated
    x_index = torch.arange(W, device=device, dtype=torch.int16).expand(H, W)
    y_index = torch.arange(H, device=device, dtype=torch.int16).unsqueeze(1).expand(H, W)
    
    return ts[selection], x_index[selection], y_index[selection]
    # return torch.stack([ts[selection], row[selection], column[selection]], -1)


def pick_and_sort(ts: Tensor, num_elements: Tensor,bn = False) -> List[numpy.recarray]:
    """ Pick the first `num_elements` events from `ts`, and add their x and y index, output as dvs events
    Args:
        ts(B,P,H,W,M): timestamps of M events
        num_elements(B,P,H,W): number of element to keep in each 1-d array of timestamps of the voxel
    """
    num_elements = torch.swapaxes(num_elements, 1, 2)
    ts = torch.swapaxes(ts, 1, 2)

    logger.debug(f"ts.shape: {ts.shape}")
    logger.debug(f"y.shape: {num_elements.shape}")

    # these two tensors need to be sliced B times, convert into tuple for faster accessing
    ts_by_batch = tuple(ts)
    selection_by_batch = tuple(num_elements)

    result_all = []
    for batch_idx in range(ts.shape[0]):
        # four arrays store the four columns of output in this batch
        ts_all = []
        row_all = []
        column_all = []
        p_all = []

        ts_by_voxel = tuple(ts_by_batch[batch_idx])
        selection_by_voxel = tuple(selection_by_batch[batch_idx])

        for voxel_index in range(ts.shape[1]):
            ts_voxel = ts_by_voxel[voxel_index]
            selection_voxel = selection_by_voxel[voxel_index]
            device = ts_voxel.device
            if bn:
                # process negative events
                ts_n, row_n, column_n = pick_elements_bn(ts_voxel[1], selection_voxel[1])
                # process positive events
                ts_p, row_p, column_p = pick_elements_bn(ts_voxel[0], selection_voxel[0])
            else:
                # process negative events
                ts_n, row_n, column_n = pick_elements(ts_voxel[1], selection_voxel[1])
                # process positive events
                ts_p, row_p, column_p = pick_elements(ts_voxel[0], selection_voxel[0])

            # stack them together, sort them
            ts_np = torch.hstack((ts_n, ts_p))
            sorting = ts_np.argsort()

            ts_all.append(ts_np[sorting])
            row_all.append(torch.hstack((row_n, row_p))[sorting])
            column_all.append(torch.hstack((column_n, column_p))[sorting])
            p_all.append(torch.hstack((torch.zeros((ts_n.shape[0]), device=device, dtype=torch.int8),
                                       torch.ones((ts_p.shape[0]), device=device, dtype=torch.int8)))[sorting])

        result = torch.hstack(ts_all).cpu(), torch.hstack(row_all).cpu(), torch.hstack(column_all).cpu(), torch.hstack(
            p_all).cpu()

        # convert to numpy recarray
        result_all.append(numpy.core.records.fromarrays(result, names=['timestamp', 'x', 'y', 'polarity']))
        # [('timestamp', '<i8'), ('x', '<i2'), ('y', '<i2'), ('polarity', 'i1')]
    return result_all


def sample_voxel_baseline(y, t0=0, fps=30, even=False, random = False):
    assert(even or random)
    B, P, C, H, W = y.shape
    device = y.device
    # y = torch.einsum('blphw->bplhw', y).reshape(B* P, L, H, W)
    y = y.reshape(B * P, C, H, W)
    delta = 1 / (fps * C)
    # Do a avg pooling on xy axis of y
    y=y.float()

    int_part = y.floor()
    decimal_part = y - int_part

    # Generate events for int part
    M = int(int_part.max())
    # Generate uniformly distributed timestamps
    ts_shape = torch.Size(list(int_part.shape) + [M])  # (B*P, C, H, W, M)

    if random:
        ts_u = torch.rand(ts_shape, device=device)*delta
    
        ts = ts_u.reshape(B, P, C, H, W, M)  # (B, P, C, H, W, M)
    if even:
        ts = torch.arange(0,M,1,device=device).expand(B,P,C,H,W,M)/(int_part.reshape(B,P,C,H,W,1)+1)
      
        ts *= delta
    int_part = int_part.reshape(B, P, C, H, W)  # (B, P, C, H, W)

    # Add the starting timestamp of each voxel to the timestamps
    ts += torch.arange(0, 1 / fps, 1 / fps / C, device=device).reshape(1, 1, C, 1, 1, 1) + t0
    ts *= 1e6
    ts = ts.to(torch.long)
    int_events= pick_and_sort(ts, int_part)

    if random:
        ts_u = torch.rand(decimal_part.shape, device=device)*delta
        ts = ts_u.reshape(B, P, C, H, W)  # (B, P, C, H, W)
    if even:
        ts = int_part.reshape(B,P,C,H,W)/(int_part.reshape(B,P,C,H,W)+1)
        ts *= delta
    
    decimal_part = decimal_part.reshape(B, P, C, H, W)  # (B, P, C, H, W)
    ts += torch.arange(0, 1 / fps, 1 / fps / C, device=device).reshape(1, 1, C, 1, 1) + t0
    ts *= 1e6
    ts = ts.to(torch.long)
    
    dec_events= pick_and_sort(ts, decimal_part, bn=True)
    results = []
    for i in range(B):
        result = np.concatenate((int_events[i],dec_events[i]))
        result = np.sort(result, order='timestamp')
        results.append(result)
    return results