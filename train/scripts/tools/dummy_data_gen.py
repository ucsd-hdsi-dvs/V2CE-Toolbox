import os
import os.path as op
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange

data_folder = '../../dummy_data'
os.makedirs(data_folder, exist_ok=True)

data_packet_num = 256

for i in trange(data_packet_num):
    data_packet = {}
    data_packet['images'] = np.random.randint(0, 255, (17, 260, 346), dtype=np.uint8)
    data_packet['gyroscopes'] = np.random.rand(17, 3)
    data_packet['accelerometers'] = np.random.rand(17, 3)
    data_packet['physical_att'] = np.random.rand(16, 260, 346)
    data_packet['optical_flow'] = np.random.rand(16, 2, 260, 346)
    data_packet['acc_flow'] = np.random.rand(16, 2, 260, 346)
    data_packet['timestamps'] = np.random.randint(0, 1000000, (17,))
    data_packet['timestamps'].sort()

    event_packets = []
    for j in range(16):
        # gen a dummy `events` object which is a structured array with 4 fields: `timestamp`, `x`, `y`, `polarity`
        events = np.zeros((1000,), dtype=[('timestamp', '<i8'), ('x', '<i2'), ('y', '<i2'), ('polarity', 'i1')])
        # `ts` is the timestamp of the event in microseconds in ascending order
        events['timestamp'] = np.random.randint(0, 1000000, (1000,))
        events['timestamp'].sort()
        events['x'] = np.random.randint(0, 346, (1000,))
        events['y'] = np.random.randint(0, 260, (1000,))
        events['polarity'] = np.random.randint(0, 2, (1000,))
        event_packets.append(events)
    data_packet['events'] = event_packets

    out_path = op.join(data_folder, f'{i:05d}.pkl')
    with open(out_path, 'wb') as f:
        pkl.dump(data_packet, f)
    # break
