"""
Losses implementations
"""
import logging
import torch
import torch.nn as nn
import os.path as op
from einops import rearrange

logger = logging.getLogger(__name__)

__all__ = ['Pyramid3dLoss', 'EncoderLoss', 'ChamferDistanceLoss', 'CompensationLoss']



class Pyramid3dLoss(nn.Module):
    def __init__(self, add_base_loss=False):
        super(Pyramid3dLoss, self).__init__()
        self.add_base_loss = add_base_loss
        self.loss = nn.MSELoss()
        self.pool_scales = [2**i for i in range(1, 4)]
        self.pools = [nn.AvgPool3d(i, stride=i) for i in self.pool_scales]

    def forward(self, pred, target):
        loss = self.loss(pred, target) if self.add_base_loss else 0
        for pool in self.pools:
            loss += self.loss(pool(pred), pool(target))
        loss /= len(self.pool_scales)
        logger.debug(f"Pyramid Loss: {loss}")
        return loss

class PyramidTemporalLoss(nn.Module):
    def __init__(self):
        super(PyramidTemporalLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.pools = [
            nn.AvgPool1d(kernel_size=3, stride=3, padding=1),
            nn.AvgPool1d(kernel_size=5, stride=5, padding=0),
        ]
    
    def forward(self, pred, target):
        pred = rearrange(pred, 'b c h w -> b (h w) c')
        target = rearrange(target, 'b c h w -> b (h w) c')
        loss = self.loss(pred, target)
        for pool in self.pools:
            loss += self.loss(pool(pred), pool(target))
        loss /= len(self.pools)
        logger.debug(f"Pyramid Temporal Loss: {loss}")
        return loss

class VoxelEncoder(nn.Module):
    def __init__(self, in_channels=20, out_channels=512, hidden_size=64):
        """ A Transformer Model that Encodes a Voxel Grid to a Vector
        Input shape: (batch_size, seq_len, in_channels, h, w)
        Output shape: (batch_size, seq_len, out_channels)
        """
        super(VoxelEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Define the downsampling layers to reduce the input shape from (B, C, H, W) to (B, C, hidden_size)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_size, hidden_size*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_size*2, hidden_size*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size*4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        # Define the transformer model
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size*4, nhead=2),
            num_layers=2,
        )
        # Define the output layer
        self.output = nn.Linear(hidden_size*4, out_channels)


    def forward(self, x):
        B, L, C, H, W = x.shape
        # Flatten the input
        x = x.view(B*L, C, H, W)
        # Apply the downsampling layers
        x = self.downsample(x).squeeze().view(B, L, -1)
        # Encode the input
        x = self.encoder(x)
        # Apply the output layer
        x = self.output(x)
        return x

class EncoderLoss(nn.Module):
    weight_path = op.join(op.dirname(op.dirname(op.abspath(__file__))), 'weights', 'voxel_encoder.pt')

    def __init__(self):
        super(EncoderLoss, self).__init__()
        self.voxel_encoder = VoxelEncoder()
        self.voxel_encoder.load_state_dict(torch.load(self.weight_path))
        self.voxel_encoder.eval()
        # freeze the parameters of the voxel encoder
        for param in self.voxel_encoder.parameters():
            param.requires_grad = False
        self.loss_func = nn.MSELoss()

    def forward(self, pred, target):
        pred = self.voxel_encoder(pred)
        target = self.voxel_encoder(target)
        loss = self.loss_func(pred, target)
        logger.debug(f"Encode Loss: {loss}")
        return loss
    
class MatchLoss(nn.Module):
    def __init__(self):
        super(MatchLoss, self).__init__()
        self.loss_func = nn.NLLLoss()
    
    def forward(self, pred, target):
        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        log_pred_softmax = torch.log(pred_softmax)

        target = target.argmax(dim=1).long()

        return self.loss_func(log_pred_softmax, target)

class CompensationLoss(nn.Module):
    def __init__(self):
        super(CompensationLoss, self).__init__()
        self.loss_func = nn.MSELoss()
    
    def forward(self, pred, target):
        pred_mask, target_mask = pred > 0.01, target > 0.01
        pred_sum = (pred * pred_mask).sum(dim=(2, 3), keepdim=True)
        target_sum = (target * target_mask).sum(dim=(2, 3), keepdim=True)

        pred_count = torch.clamp(pred_mask.sum(dim=(2, 3), keepdim=True), min=1)
        target_count = torch.clamp(target_mask.sum(dim=(2, 3), keepdim=True), min=1)

        return self.loss_func(pred_sum / pred_count, target_sum / target_count)

        