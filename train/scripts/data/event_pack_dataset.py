import os
import os.path as op
import pickle as pkl
import numpy as np
import logging
import torch
from torchvision import transforms

from torch.utils.data import Dataset
from ..utils.dl_utils import train_val_test_split
from ..utils.data_utils import seq_random_flip
from ..utils.events_utils import structured_events_to_voxel_grid, gen_discretized_event_volume
from ..utils.physical_att import gen_log_frame_residual_batch, physical_attention_generation, physical_attention_batch_generation
from ..utils.image_derivative import get_batch_double_blurred_image_gradient

logger = logging.getLogger(__name__)

class EventPackDataset(Dataset):
    def __init__(self, mode, data_dir, partial_dataset=1, transform=None, seq_len=16, frame_size=(260, 346), num_bins=10, ef_collapse_seq=True, advanced_physical_att=False, apply_image_grad=False, random_flip=False, flip_x_prob=0.5, flip_y_prob=0,
                 phyatt_grid_size=8, seed=2333, ceiling_att=5, **kwargs):
        self.mode = mode
        self.data_root = data_dir
        self.transform = transform
        self.num_bins = num_bins
        self.phyatt_grid_size = phyatt_grid_size
        self.partial_dataset = partial_dataset
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.ef_collapse_seq = ef_collapse_seq
        self.advanced_physical_att = advanced_physical_att
        self.ceiling_att = ceiling_att
        self.apply_image_grad = apply_image_grad
        self.random_flip = random_flip
        self.flip_x_prob = flip_x_prob
        self.flip_y_prob = flip_y_prob

        #Normalization for image units and flows
        self.frame_normalize = transforms.Compose([
                transforms.Normalize([0.153, 0.153], [0.165, 0.165])])
        self.normalize_optical_flow = transforms.Compose([
                transforms.Normalize([-0.0673,  0.0192], [1.7283, 1.8886])])
        self.normalize_flows = transforms.Compose([
                transforms.Normalize([ 420.4524, -3841.5618], [6386.6489, 4546.8569])])

        with open('/tsukimi/datasets/MVSEC/data_paths_new.pkl', 'rb') as f:
            self.paths_pack = pkl.load(f)
        
        if mode == 'train':
            self.data_paths = self.paths_pack['train']
        elif mode == 'val':
            self.data_paths = self.paths_pack['val']
        elif mode == 'test':
            self.data_paths = self.paths_pack['test']

    def __len__(self):
        return int(self.partial_dataset*len(self.data_paths))
    
    def __getitem__(self, idx):
        data_path = op.join(self.data_root, self.data_paths[idx])
        with open(data_path, 'rb') as f:
            data_packet = pkl.load(f)

        events = data_packet['events']
        lfr = gen_log_frame_residual_batch(data_packet['images'])
        lfr = torch.from_numpy(lfr).float()
        image_units = np.stack([data_packet['images'][:-1], data_packet['images'][1:]], axis=1) # 16, 2, 260, 346

        image_units = torch.from_numpy(image_units).float() / 255
        if self.apply_image_grad:
            image_gradient_blur = get_batch_double_blurred_image_gradient(image_units[:, 0:1], image_units[:, 1:2])
            image_gradient_blur = image_gradient_blur / image_gradient_blur.max()
            image_units = self.frame_normalize(image_units)
            image_units = torch.cat([image_units, image_gradient_blur], dim=1)
        else:
            image_units = self.frame_normalize(image_units)
            
        gyroscopes = torch.from_numpy(data_packet['gyroscopes']).float()
        accelerometers = torch.from_numpy(data_packet['accelerometers']).float()
        optical_flow = torch.from_numpy(data_packet['optical_flow']).float()
        optical_flow = self.normalize_optical_flow(optical_flow)
        acc_flow = torch.from_numpy(data_packet['acc_flow']).float()
        acc_flow = self.normalize_flows(acc_flow)
        flows = torch.cat([optical_flow, acc_flow], axis=1) #16, 4, 260, 346

        # start buuilding the torch for voxels based on events input
        voxels = []
        for i in range(len(events)):
            # time_voxel = structured_events_to_voxel_grid(events[i], num_bins=self.num_bins, width=346, height=260)
            time_voxel = gen_discretized_event_volume(events[i],
                                                    [self.num_bins*2,
                                                     self.frame_size[0],
                                                     self.frame_size[1]])

            voxels.append(time_voxel)

        voxels = torch.stack(voxels, dim=0)
        imu = torch.cat([accelerometers, gyroscopes], axis=1)[1:]

        if 0 < self.seq_len < 16:
            lfr = lfr[:self.seq_len]
            image_units = image_units[:self.seq_len]
            flows = flows[:self.seq_len]
            voxels = voxels[:self.seq_len]
            imu = imu[:self.seq_len]

        if self.mode == 'train' and self.random_flip:
            image_units, voxels, imu, flows = seq_random_flip(image_units, voxels, imu, flows, self.flip_x_prob, self.flip_y_prob)
            
        return {
            'image_units': image_units, # [L, 2, H, W]
            'flows': flows, # [L, 4, H, W]
            'voxels': voxels, # [L, 2*num_bin, H, W]
            'imu': imu, # [L, 6]
            'physical_att': None, # [L, 1, H, W]
            'lfr': lfr, # [L, 1, H, W]
            'data_path': data_path
        }

