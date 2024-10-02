import logging
from typing import List, Tuple

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# @tic_toc
def calculate_statistical_linear_params_for_stage2(y):
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

    sum_x = 0
    sum_x2 = 2
    N = 3
    # Using the least squares formula to calculate the slope m and the intercept b
    k = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)

    logger.debug(f"Slope k's shape: {k.shape}")
    logger.debug(f"nonzero k: {torch.sum(k>0)}")
    logger.debug(f"max raw k: {torch.max(k)}")
    logger.debug(f"min raw k: {torch.min(k)}")
    return k

def y_relocate_adapt(y):
    B, C, H, W = y.shape
    new_y = torch.zeros((B, C-1, H, W), device=y.device, dtype=int)
    rand_y = torch.rand((B, C-1, H, W), device=y.device, dtype=float)
    tendency = torch.zeros((B, C-1, H, W), device=y.device, dtype=float)
    
    from_left_until = C-1
    debt = torch.zeros_like(y[:,0,:,:])

    for i in range(from_left_until): 
        yslice = y[:,i,:,:]
        _new_y_slice = yslice - debt 

        within_one_mask = (yslice>0)&(yslice<1)
        new_y[:,i,:,:][within_one_mask] = torch.bernoulli(yslice[within_one_mask])
        tendency[:,i,:,:][within_one_mask] = rand_y[:,i,:,:][within_one_mask]

        new_y_slice = torch.ceil(_new_y_slice-1e-6) 
        debt = new_y_slice - _new_y_slice
        new_y[:,i,:,:] = new_y_slice
        tendency[:,i,:,:] = debt
    
    new_y[:,-1,:,:] += (y[:,-1,:,:]-debt).int()
    print("ratio",torch.sum(new_y)/torch.sum(y))
    return new_y, tendency

# @tic_toc
def y_relocate(y, bidirectional=False, erase_beginning=False):
    B, C, H, W = y.shape
    new_y = torch.zeros((B, C-1, H, W), device=y.device, dtype=int)
    tendency = torch.zeros((B, C-1, H, W), device=y.device, dtype=float)

    # for values that are smaller than 0.01 in the original y, we will set them to 0 
    if erase_beginning:
        y = torch.where(y<0.001, torch.zeros_like(y), y)
    
    if not bidirectional:
        from_left_until = C-1
    else:
        from_left_until = (C-1)//2
    
    debt = torch.zeros_like(y[:,0,:,:])

    for i in range(from_left_until): 
        yslice = y[:,i,:,:]
        _new_y_slice = yslice - debt 
        new_y_slice = torch.ceil(_new_y_slice-1e-6) 
        debt = new_y_slice - _new_y_slice
        new_y[:,i,:,:] = new_y_slice

        tendency[:,i,:,:] = debt    
    
    if not bidirectional:
        new_y[:,-1,:,:] += (y[:,-1,:,:]-debt).int()
    else:
        bless = y[:,C-1,:,:]
        for i in range(C-2, C//2, -1):
            yslice = y[:,i,:,:]
            tendency[:,i,:,:] = bless
            _y_slice = yslice + bless
            _y_slice = torch.floor(_y_slice+1e-6)

            bless = yslice-_y_slice + bless
            bless = torch.clamp(bless, min=0)
            new_y[:,i,:,:] = _y_slice

        i = C//2
        yslice = y[:,i,:,:]
        tendency[:,i,:,:] = bless-debt 
        new_y[:,i,:,:] = torch.ceil(yslice+bless-debt) 
    return new_y, tendency

# @tic_toc
def sample_voxel_statistical(y, t0=0, fps=30, pooling_type='none', pooling_kernel_size=3, additional_events_strategy='slope', bidirectional=False):
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
    logger.debug(f"additional_events_strategy: {additional_events_strategy}")
    B, P, C, H, W = y.shape
    
    device = y.device
    y = y.reshape(B * P, C, H, W).float()
    frame_step = 1 / fps
    voxel_step = 1 / fps / (C-1)
    #print("before",torch.sum(y))
    # Reloaate y based on the generation method of event voxel
    y, y_tendency = y_relocate(y, bidirectional=bidirectional)
    #print("after",torch.sum(y))
    logger.debug(f"None-zero y_tendency: {torch.sum(y_tendency>0)}")
    logger.debug(f"None-zero y: {torch.sum(y>0)}")
    C = C-1

    # Adapt the y_tendency whose unit is a unit voxel time to actual time in seconds
    ts = y_tendency / fps / C

    # Reshape the timestamps and y to get ready for pick&sort
    ts = ts.reshape(B, P, C, H, W)  # (B, P, C, H, W)
    y = y.reshape(B, P, C, H, W) # (B, P, C, H, W)

    # Add the starting timestamp of each voxel to the timestamps
    ts += torch.arange(0, frame_step, voxel_step, device=device).reshape(1, 1, C, 1, 1) + t0
    ts *= 1e6
    ts = ts.to(torch.long)

    ######## DEAL WITH ADDITIONAL EVENTS WHERE THE VOXEL VALUE IS LARGER THAN 1 ########
    # Generate uniformly distributed timestamps
    max_event_num_per_voxel = torch.max(y)
    additional_ts_shape = torch.Size(list(y.shape) + [max_event_num_per_voxel])  # (B*P, C, H, W, max_event_num_per_voxel-1)
    raw_additional_ts = torch.rand(additional_ts_shape, device=device)

    if additional_events_strategy == 'random':
        additional_ts = raw_additional_ts
    elif additional_events_strategy == 'slope':
        # Do a avg pooling on xy axis of y
        if pooling_type == 'weighted':
            pooling_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=device, dtype=torch.float) / 16
            pooling_kernel = pooling_kernel.unsqueeze(0).unsqueeze(0)
            y_pooled = F.conv2d(y.reshape(B * P * C, 1, H, W).float(), pooling_kernel, padding=1, groups=1).reshape(B * P, C, H, W)
        elif pooling_type == 'avg':
            y_pooled = nn.AvgPool2d(kernel_size=pooling_kernel_size, stride=1, padding=pooling_kernel_size//2)(y.reshape(B * P, C, H, W).float())
        elif pooling_type == 'none':
            y_pooled = y.float()
            
        # Calculate slope for slope distribution, divide by y so the area under is 1
        y_pooled = y_pooled.reshape(B * P, C, H, W)
        k = calculate_statistical_linear_params_for_stage2(y_pooled) / (voxel_step**2) / (y_pooled+1e-8)
        
        b = 1 / voxel_step - voxel_step * k / 2
        b = b.unsqueeze(-1).repeat(1, 1, 1, 1, max_event_num_per_voxel)
        k = k.unsqueeze(-1).repeat(1, 1, 1, 1, max_event_num_per_voxel)
        
        raw_additional_ts = raw_additional_ts.reshape(B*P, C, H, W, max_event_num_per_voxel)
        additional_ts = (-b + torch.sqrt((b ** 2 + 2 * k * raw_additional_ts))) / k
        additional_ts = torch.where(k==0, raw_additional_ts / fps / C, additional_ts)
        if additional_ts.numel() > 0:
            logger.debug(f"max of additional_ts: {torch.max(additional_ts)}")
            logger.debug(f"min of additional_ts: {torch.min(additional_ts)}")
        logger.debug(f"max of raw_additional_ts: {torch.max(raw_additional_ts)}")
        logger.debug(f"min of raw_additional_ts: {torch.min(raw_additional_ts)}")
        logger.debug(f"max of k: {torch.max(k)}")
        logger.debug(f"min of k: {torch.min(k)}")
        logger.debug(f"max of b: {torch.max(b)}")
        logger.debug(f"min of b: {torch.min(b)}")
    else:
        additional_ts = torch.zeros_like(raw_additional_ts)
        
    additional_ts = additional_ts.reshape(B, P, C, H, W, max_event_num_per_voxel) 
    additional_ts += torch.arange(0, frame_step, voxel_step, device=device).reshape(1, 1, C, 1, 1, 1) + t0
    additional_ts *= 1e6
    additional_ts = additional_ts.to(torch.long)

    return pick_and_sort(ts, y, additional_ts, additional_events_strategy=additional_events_strategy)

# @tic_toc
def pick_elements(ts: Tensor, num_elements: Tensor, additional_ts:Tensor, additional_events_strategy='none') -> Tuple[Tensor, Tensor, Tensor]:
    """ Pick the first `num_elements` events from `ts`, and add their x and y index
    Args:
        ts(H,W): the exact location of the last event within the voxel
        num_elements(H,W): event number at each pixel 
        additional_ts(H,W,max_event_num_per_voxel): timestamps of additional events (where the voxel value is larger than 1)
    """
    H, W = num_elements.shape
    device = ts.device

    # if a voxel in num_elements has a value which is larger than 0 and smaller than 1, select it
    selection = (num_elements == 1) #(num_elements > 0) & (num_elements <= 1)
    # Create the row and column indices for events generated
    x_index = torch.arange(W, device=device, dtype=torch.int16).expand(H, W)
    y_index = torch.arange(H, device=device, dtype=torch.int16).unsqueeze(1).expand(H, W)
    ts_selected = ts[selection]
    x_index_selected = x_index[selection]
    y_index_selected = y_index[selection]
    
    num_elements = torch.where((num_elements==1), torch.zeros_like(num_elements), num_elements)
    
    max_event_num_per_voxel = additional_ts.shape[-1]
    selection_additional = torch.arange(additional_ts.shape[-1], device=device).unsqueeze(0).unsqueeze(1) < num_elements.unsqueeze(2)

    if additional_events_strategy != 'none':
        ts_selected = torch.cat((ts_selected, additional_ts[selection_additional]))
        x_index_selected = torch.cat((x_index_selected, x_index.unsqueeze(-1).expand(H, W, max_event_num_per_voxel)[selection_additional]))
        y_index_selected = torch.cat((y_index_selected, y_index.unsqueeze(-1).expand(H, W, max_event_num_per_voxel)[selection_additional]))
    return ts_selected, x_index_selected, y_index_selected

# @tic_toc
def pick_and_sort(ts, num_elements, additional_ts=None, additional_events_strategy='none'):
    """ Pick the first `num_elements` events from `ts`, and add their x and y index, output as dvs events
    Args:
        ts(B,P,C,H,W): timestamp of the last event within the voxel
        num_elements(B,P,C,H,W): number of element to keep in each 1-d array of timestamps of the voxel
        additional_ts(B,P,C,H,W): timestamps of additional events (where the voxel value is larger than 1)
    """
    B, P, C, H, W = ts.shape
    device = ts.device
    
    num_elements = torch.swapaxes(num_elements, 1, 2)
    ts = torch.swapaxes(ts, 1, 2) # (B, C, P, H, W)
    if additional_ts is not None:
        additional_ts = torch.swapaxes(additional_ts, 1, 2)
    else:
        additional_ts = torch.zeros(ts.shape+(1,), device=device, dtype=ts.dtype)

    logger.debug(f"ts.shape: {ts.shape}")
    logger.debug(f"y.shape: {num_elements.shape}")

    # these two tensors need to be sliced B times, convert into tuple for faster accessing
    ts_by_batch = tuple(ts)
    selection_by_batch = tuple(num_elements)
    additional_ts_by_batch = tuple(additional_ts)

    result_all = []
    # Iterate through each batch
    for batch_idx in range(B): # B
        # four arrays store the four columns of output in this batch
        ts_all, row_all, column_all, p_all = [], [], [], []

        ts_by_voxel = tuple(ts_by_batch[batch_idx]) # (C, P, H, W)
        selection_by_voxel = tuple(selection_by_batch[batch_idx])
        additional_ts_by_voxel = tuple(additional_ts_by_batch[batch_idx]) 

        # Iterate through each channel
        for channel_index in range(C): # C
            ts_channel = ts_by_voxel[channel_index] # (P, H, W)
            selection_channel = selection_by_voxel[channel_index] # (P, H, W)
            additional_ts_channel = additional_ts_by_voxel[channel_index] # (P, H, W)

            # process negative events
            ts_n, row_n, column_n = pick_elements(ts_channel[1], selection_channel[1], additional_ts_channel[1], additional_events_strategy=additional_events_strategy)

            # process positive events
            ts_p, row_p, column_p = pick_elements(ts_channel[0], selection_channel[0], additional_ts_channel[0], additional_events_strategy=additional_events_strategy)

            # stack them together, sort them
            ts_np = torch.hstack((ts_n, ts_p))
            sorting = ts_np.argsort()

            ts_all.append(ts_np[sorting])
            row_all.append(torch.hstack((row_n, row_p))[sorting])
            column_all.append(torch.hstack((column_n, column_p))[sorting])
            p_all.append(torch.hstack((torch.zeros((ts_n.shape[0]), device=device, dtype=torch.int8),
                                       torch.ones((ts_p.shape[0]), device=device, dtype=torch.int8)))[sorting])
            
        ts_all, row_all, column_all, p_all = [torch.hstack(x).cpu() for x in [ts_all, row_all, column_all, p_all]]
        
        # convert to numpy recarray
        result_all.append(numpy.core.records.fromarrays([ts_all, row_all, column_all, p_all], names=['timestamp', 'x', 'y', 'polarity']))
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
    numpy.random.seed(42)
    torch.random.manual_seed(42)
    logging.basicConfig(level=logging.DEBUG)
    B, P, C, H, W = 10, 2, 10, 260, 346

    runtime, res = timer(
        lambda: sample_voxel_statistical(torch.rand((B, P, C, H, W), device='cuda', dtype=torch.float)))

    total = 0
    for i in range(B):
        total += res[i].shape[0]
    print(runtime / total)
    print(runtime)
    print(torch.cuda.max_memory_allocated(device='cuda') / 1024 ** 3)
    print(torch.cuda.max_memory_allocated(device='cuda') / total)
    print(len(res), res[0].shape)
    print(res)
    y = torch.randint(0, 10, (B, P, C, H, W), device='cuda', dtype=torch.int16)

    runtime, res = timer(
        lambda: sample_voxel_statistical(torch.randint(0, 10, (B, P, C, H, W), device='cuda', dtype=torch.int16)))
    total = 0
    for i in range(B):
        total += res[i].shape[0]
    print(runtime / total)

