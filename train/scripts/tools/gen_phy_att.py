import os
import sys
import glob
import os.path as op
import numpy as np
import pickle as pkl
import multiprocessing as mp

sys.path.append("..")

from utils.physical_att import physical_attention_batch_generation

data_root = r'/tsukimi/datasets/MVSEC/event_chunks_processed'
data_paths = glob.glob(op.join(data_root, '*.pkl'))
print(f"The number of data packets: {len(data_paths)}")

def correct_phy_att(data_path):
    print(f"Processing {data_path}")
    with open(data_path, 'rb') as f:
        data_packet = pkl.load(f)

    events = data_packet['events']
    image_units = np.stack([data_packet['images'][:-1], data_packet['images'][1:]], axis=1) # 16, 2, 260, 346

    physical_att = physical_attention_batch_generation(events, image_units, 8, advanced=True, ceiling=25)

    data_packet['physical_att'] = physical_att
    # write back to the original file
    with open(data_path, 'wb') as f:
        pkl.dump(data_packet, f)

    # write this processed data_path into 'processed.txt', be aware of the multi-processing lock
    with open('processed.txt', 'a') as f:
        f.write(data_path + '\n')


# Feed the data paths to the multiprocessing pool with a tqdm wrapper
pool = mp.Pool(processes=mp.cpu_count())
pool.map(correct_phy_att, data_paths)
pool.close()
pool.join()
