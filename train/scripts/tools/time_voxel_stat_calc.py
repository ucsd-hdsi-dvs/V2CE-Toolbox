import os
import sys
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.events_utils import gen_discretized_event_volume

root = r'/tsukimi/datasets/MVSEC/event_chunks_processed'
file_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pkl')]

buffer = []
for idx, file_path in tqdm(enumerate(file_paths), total=len(file_paths)):
    with open(file_path, 'rb') as f:
        data_packet = pkl.load(f)
    events = data_packet['events']
    volume = [gen_discretized_event_volume(e, [20, 260, 346]) for e in events]
    temp = np.concatenate([v.flatten() for v in volume])
    # remove 0 values in temp
    temp = temp[temp != 0]
    buffer.append(temp)
    # if idx == 2:
    #     break
buffer = np.concatenate(buffer, axis=0)

# calculate mean and std
mean = np.mean(buffer)
std = np.std(buffer)
print(mean, std)

# write results to file
with open('mean_std.txt', 'w') as f:
    f.write('mean: {}\n'.format(mean))
    f.write('std: {}\n'.format(std))
    f.write('max: {}\n'.format(np.max(buffer)))
    f.write('min: {}\n'.format(np.min(buffer)))
    # top 10% percentile
    f.write('top 10% percentile: {}\n'.format(np.percentile(buffer, 90)))
    # top 5% percentile
    f.write('top 5% percentile: {}\n'.format(np.percentile(buffer, 95)))
    # top 2% percentile
    f.write('top 2% percentile: {}\n'.format(np.percentile(buffer, 98)))
    # top 1% percentile
    f.write('top 1% percentile: {}\n'.format(np.percentile(buffer, 99)))
    # top 0.1% percentile
    f.write('top 0.1% percentile: {}\n'.format(np.percentile(buffer, 99.9)))
    
    # bottom 10% percentile
    f.write('bottom 10% percentile: {}\n'.format(np.percentile(buffer, 10)))
    # bottom 5% percentile
    f.write('bottom 5% percentile: {}\n'.format(np.percentile(buffer, 5)))
    # bottom 2% percentile
    f.write('bottom 2% percentile: {}\n'.format(np.percentile(buffer, 2)))
    # bottom 1% percentile
    f.write('bottom 1% percentile: {}\n'.format(np.percentile(buffer, 1)))
    # bottom 0.1% percentile
    f.write('bottom 0.1% percentile: {}\n'.format(np.percentile(buffer, 0.1)))
    
    f.write('voxel number: {}\n'.format(len(buffer)))
    f.write('file number: {}\n'.format(len(file_paths)))