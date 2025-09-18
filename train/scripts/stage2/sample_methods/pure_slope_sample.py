import logging
from typing import List, Tuple

import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)

def calculate_statistical_linear_params_for_stage2(y, bias=0):
    """
    Calculate the statistical linear parameters for stage 2
    Args:
        y: input tensor of shape (B, L, H, W)
        bias: bias for the x values
    Returns:
        k: slope of the linear function, shape (B, H, W)
    """
    B, L, H, W = y.shape
    # Padding the input
    y_padded = F.pad(y, (0, 0, 0, 0, 1, 1), mode='reflect')
    device = y.device
    # Create convolution kernels
    ones_kernel = torch.ones((1, 1, 3), device=device)
    xy_kernel = torch.tensor([-1.0, 0.0, 1.0], device=device).repeat(1, 1, 1)

    # Reshape padded tensor
    y_padded_reshaped = torch.einsum('bkhw->bhwk', y_padded).reshape(B * H * W, 1, L + 2)

    # Compute sum_y
    sum_y = F.conv1d(y_padded_reshaped, ones_kernel, padding=0).view(B, H, W, L).permute(0, 3, 1, 2)

    # Compute sum_xy
    sum_xy = F.conv1d(y_padded_reshaped, xy_kernel, padding=0).view(B, H, W, L).permute(0, 3, 1, 2)

    # # Create corresponding x values
    # x = torch.tensor([-1.0, 0.0, 1.0],device=device)
    # x += bias
    #
    # # Compute sums
    # sum_x = torch.sum(x)  # Σx
    # sum_x2 = torch.sum(x ** 2)  # Σx²
    # N = x.shape[0]  # Number of points

    sum_x = 0
    sum_x2 = 2
    N = 3
    # Using the least squares formula to calculate the slope m and the intercept b
    k = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)

    logger.debug(f"Slope k's shape: {k.shape}")
    return k

def sample_voxel_statistical(y, t0=0, fps=30, pooling_type='none', pooling_kernel_size=3, additional_events_strategy='slope'):

    """ Sample voxel from y, and add noise to it
    Args:
        y: input tensor of shape (B, P, C, H, W), where P=2, C=10
        alpha: A constant to control the μ/k ratio
        t0: The starting timestamp of the event sequence
        fps: Frames per second
        time_bins: Number of time bins
    """
    assert pooling_type in ['avg', 'weighted', 'none']
    assert additional_events_strategy in ['none', 'random', 'slope']

    logger.debug(f"pooling_type: {pooling_type}")
    logger.debug(f"pooling_kernel_size: {pooling_kernel_size}")
    B, P, C, H, W = y.shape
    
    device = y.device
    # y = torch.einsum('blphw->bplhw', y).reshape(B* P, L, H, W)
    y = y.reshape(B * P, C, H, W)
    voxel_step = 1 / (fps * C)
    
    if pooling_type == 'weighted':
        pooling_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=device, dtype=torch.float) / 16
        pooling_kernel = pooling_kernel.unsqueeze(0).unsqueeze(0)
        y_pooled = F.conv2d(y.reshape(B * P * C, 1, H, W).float(), pooling_kernel, padding=1, groups=1).reshape(B * P, C, H, W)
    elif pooling_type == 'avg':
        y_pooled = nn.AvgPool2d(kernel_size=pooling_kernel_size, stride=1, padding=pooling_kernel_size//2)(y.reshape(B * P, C, H, W).float())
    elif pooling_type == 'none':
        y_pooled = y.float()
        
    # Calculate slope for slope distribution, divide by y so the area under is 1
    k = calculate_statistical_linear_params_for_stage2(y_pooled) / (voxel_step**2) / (y_pooled+1e-8)
    # k = calculate_statistical_linear_params_for_stage2(y_pooled) / y_pooled / delta #?
    
    b = 1 / voxel_step - voxel_step * k / 2
    y[:,8,:,:]+= y[:,9,:,:]
    y[:,9,:,:]=0

    int_part = y.floor().int()
    decimal_part = y - int_part
    
    # Generate uniformly distributed timestamps
    ts_shape = torch.Size(list(y.shape))  # (B*P, C, H, W)
    ts_u = torch.rand(ts_shape, device=device)
    # Repeat the k values to match the shape of ts
    # k = k.unsqueeze(-1).repeat(1, 1, 1)
    # b = b.unsqueeze(-1).repeat(1, 1, 1)
    # Transform the timestamps to the corresponding values in a slope distribution
    ts = (-b + torch.sqrt((b ** 2 + 2 * k * ts_u))) / k
    ts = torch.where(k==0, ts_u / fps / C, ts)

    # Reshape the timestamps and y to get ready for pick&sort
    ts = ts.reshape(B, P, C, H, W)  # (B, P, C, H, W)
    decimal_part = decimal_part.reshape(B, P, C, H, W)  # (B, P, C, H, W)

    # Add the starting timestamp of each voxel to the timestamps
    ts += torch.arange(0, 1 / fps, 1 / fps / C, device=device).reshape(1, 1, C, 1, 1) + t0
    ts *= 1e6
    ts = ts.to(torch.long)
    logger.debug(f"k.shape: {k.shape}")
    dec_events = pick_and_sort(ts, decimal_part, bn=True)

    # Generate events for int part
    M = int(int_part.max())

    # Generate uniformly distributed timestamps
    ts_shape = torch.Size(list(int_part.shape) + [M])  # (B*P, C, H, W, M)
    ts_u = torch.rand(ts_shape, device=device)

    # Repeat the k values to match the shape of ts
    k = k.unsqueeze(-1).repeat(1, 1, 1, 1, 1, M)
    b = b.unsqueeze(-1).repeat(1, 1, 1, 1, 1, M)

    # Transform the timestamps to the corresponding values in a slope distribution
    ts = (-b + torch.sqrt((b ** 2 + 2 * k * ts_u))) / k
    ts = torch.where(k==0, ts_u / fps / C, ts)

    # Reshape the timestamps and y to get ready for pick&sort
    ts = ts.reshape(B, P, C, H, W, M)  # (B, P, C, H, W, M)
    int_part = int_part.reshape(B, P, C, H, W)  # (B, P, C, H, W)

    # Add the starting timestamp of each voxel to the timestamps
    ts += torch.arange(0, 1 / fps, 1 / fps / C, device=device).reshape(1, 1, C, 1, 1, 1) + t0
    ts *= 1e6
    ts = ts.to(torch.long)

    int_events = pick_and_sort(ts, int_part)


    results = []
    for i in range(B):
        result = np.concatenate((int_events[i],dec_events[i]))
        result = np.sort(result, order='timestamp')
        results.append(result)
    return results


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


def timer(func):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    res = func()
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    return start.elapsed_time(end), res


if __name__ == "__main__":
    # torch.seed(42)
    numpy.random.seed(42)
    torch.random.manual_seed(42)
    logging.basicConfig(level=logging.DEBUG)
    B, P, C, H, W = 10, 2, 10, 260, 346
    # y = torch.randint(0, 10, (B, P, C, H, W), device='cuda', dtype=torch.int16)
    # y = torch.tensor([0,1,2,3,4,5])
    # y = y.unsqueeze(1).unsqueeze(1).unsqueeze(0).unsqueeze(0)
    runtime, res = timer(
        lambda: sample_voxel_statistical(torch.rand((B, P, C, H, W), device='cuda', dtype=torch.float)))
    # res = sample_voxel_statistical(y)
    # runtime = time.time() - start
    total = 0
    for i in range(B):
        total += res[i].shape[0]
    print(runtime / total)
    print(runtime)
    print(torch.cuda.max_memory_allocated(device='cuda') / 1024 ** 3)
    print(torch.cuda.max_memory_allocated(device='cuda') / total)
    # res = sample_voxel_statistical(y)
    print(len(res), res[0].shape)
    print(res)
    y = torch.randint(0, 10, (B, P, C, H, W), device='cuda', dtype=torch.int16)

    runtime, res = timer(
        lambda: sample_voxel_statistical(torch.randint(0, 10, (B, P, C, H, W), device='cuda', dtype=torch.int16)))
    total = 0
    for i in range(B):
        total += res[i].shape[0]
    print(runtime / total)

