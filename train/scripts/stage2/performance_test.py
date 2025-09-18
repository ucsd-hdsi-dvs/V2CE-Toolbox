import os
import time
import torch
import pickle as pkl
import numpy as np
import os.path as op
from tqdm import tqdm

from sample_methods.LDATI import sample_voxel_statistical as our_sample

# data_root = "/mnt/Kristina/V2CE/Datasets/MVSEC/best_model_log/recorder"
data_root = "/tsukimi/v2ce-project/best_model_log/recorder"

file_paths = [op.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".pkl")]
file_paths.sort()
print(len(file_paths))

def temp(y):
    y = y.cuda()
    tic = time.time()
    _ = our_sample(y, fps=30, additional_events_strategy='slope', bidirectional=False)[0]
    del y
    return time.time() - tic

def main():
    time_acc = []
    with torch.no_grad():
        for file_path in tqdm(file_paths):
            v2ce_recorder = pkl.load(open(file_path, 'rb'))

            event_voxels = []
            for i in range(16):
                # load info we are interested in from v2ce_recorder
                batch = v2ce_recorder['batch']
                outputs = v2ce_recorder['outputs']
                gt_voxel = batch['voxels'][i]

                event_voxels.append(gt_voxel)
            event_voxels = torch.from_numpy(np.stack(event_voxels, axis=0))
            event_voxels = event_voxels.reshape(16, 2, 10, 260, 346)

            try:
                time_acc.append(temp(event_voxels))
                print(f"Averaged time: {np.mean(time_acc)/16*1000} ms")
            except:
                print(file_path)
                continue

    print(f"Averaged time: {np.mean(time_acc)/16*1000} ms")
    
if __name__ == "__main__":
    main()