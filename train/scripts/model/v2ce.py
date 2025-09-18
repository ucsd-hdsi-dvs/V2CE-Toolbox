"""
Event Motion Net: A deep learning model for event-based time voxel generation with motion information 
author: Zhongyang Zhang
"""
import importlib
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


class V2ce(nn.Module):
    def __init__(self, in_channels, out_channels, unet_multi=True, real_multi_out=False, unet_all_residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_multi = unet_multi
        self.real_multi_out = real_multi_out

        if unet_all_residual:
            UNet = importlib.import_module('scripts.model.unet_2layer').UNet
        else:
            UNet = importlib.import_module('scripts.model.unet').UNet

        self.UNet=UNet(num_input_channels=self.in_channels,
                     num_output_channels=self.out_channels,
                     skip_type='concat',
                     activation='relu',
                     num_encoders=4,
                     base_num_channels=32,
                     num_residual_blocks=2,
                     norm='BN',
                     sn=True,
                     multi=self.unet_multi)

    def forward(self, x):
        frames = x['image_units'] # [B, L, 2, H, W]
        B, L, C, H, W = frames.shape
        frames=frames.reshape(B*L,C,H,W)
        voxels=self.UNet(frames)
        if self.real_multi_out:
            voxels = [voxel.reshape(B,L,self.out_channels,H,W) for voxel in voxels]
        else:
            voxels = [voxels[-1].reshape(B,L,self.out_channels,H,W)]
        return {'voxels': voxels}
