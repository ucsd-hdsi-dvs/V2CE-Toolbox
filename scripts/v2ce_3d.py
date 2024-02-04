"""
Event Motion Net: A deep learning model for event-based time voxel generation with motion information 
author: Zhongyang Zhang
"""
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

from .unet_2layer import UNet3D

class V2ce3d(nn.Module):
    def __init__(self, in_channels=2, out_channels=20):
        super().__init__()
        self.UNet=UNet3D(num_input_channels=in_channels,
                     num_output_channels=out_channels,
                     skip_type='concat',
                     activation='relu',
                     num_encoders=4,
                     base_num_channels=32,
                     num_residual_blocks=2,
                     norm='BN',
                     sn=True,
                     multi=False)

    def forward(self, x):
        frames = x.permute(0,2,1,3,4)
        voxels = self.UNet(frames)
        voxels = voxels[-1].permute(0,2,1,3,4)
        return voxels
    