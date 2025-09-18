import os.path as op
import sys
import torch
import numpy as np
import torch.nn.functional as F

from fastflownet import FastFlowNet
from .flow_vis import flow_to_color

# Comment this line if used by other hosts
#CKPT_PATH = op.join(op.dirname(op.dirname(op.dirname(__file__))), 'weights', 'fastflownet', 'fastflownet_ft_mix.pth')
# rewrite to relative path
# CKPT_PATH = 
# CKPT_PATH = op.join(op.dirname(op.dirname(op.dirname(r'E:\GitHub\V2CE\scripts\utils\optical_flow.py'))), 'weights', 'fastflownet', 'fastflownet_ft_mix.pth')
# CKPT_PATH='weights/fastflownet/fastflownet_ft_mix.pth'
# rewrite to relative path using os.path
CKPT_PATH = op.join(op.dirname(op.abspath(__file__)), 'weights', 'fastflownet', 'fastflownet_ft_mix.pth')


def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

def batch_flow_calc(img1, img2, model, div_flow=20.0, div_size=64):
    """
    Calculate the optical flow between two image batches.
    Args:
        img1: torch.Tensor, shape (b, c, h, w), the first image, in [0, 1] range.
        img2: torch.Tensor, shape (b, c, h, w), the second image, in [0, 1] range.
        model: torch.nn.Module, the optical flow model.
        div_flow: float, the value to divide the flow.
        div_size: int, the value to divide the image size.
    
    Returns:
        flow: torch.Tensor, shape (b, 2, h, w).
    """
    img1, img2, _ = centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)), 
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    input_t = torch.cat([img1, img2], 1).cuda()

    
    with torch.no_grad():
        output_t = model(input_t)

    flow = div_flow * F.interpolate(output_t, size=input_size, mode='bilinear', align_corners=False)

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h
        
    return flow

def single_image_pair_flow_calc(img1, img2, model, div_flow=20.0, div_size=64):
    """
    Calculate the optical flow between two images.
    Args:
        img1: np.ndarray, shape (h, w, c), the first image, in [0, 1] range.
        img2: np.ndarray, shape (h, w, c), the second image, in [0, 1] range.
        model: torch.nn.Module, the optical flow model.
        div_flow: float, the value to divide the flow.
        div_size: int, the value to divide the image size.
        
    Returns:
        flow: np.ndarray, shape (2, h, w).
    """
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0)
    print(img1.shape, img2.shape)
    flow = batch_flow_calc(img1, img2, model, div_flow, div_size)
    return flow

class OpticalFlowCalculator:
    def __init__(self, div_flow=20.0, div_size=64, ckpt_path=CKPT_PATH):
        # self.model = model
        self.div_flow = div_flow
        self.div_size = div_size

        self.model = FastFlowNet().cuda().eval()
        # print(ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path))

    def batch_flow_calc(self, img1, img2):
        return batch_flow_calc(img1, img2, self.model, self.div_flow, self.div_size)
    
    def single_image_pair_color_flow_calc(self, img1, img2):
        return single_image_pair_flow_calc(img1, img2, self.model, self.div_flow, self.div_size)

    def __call__(self, img1, img2):
        if len(img1.shape) == 3:
            return self.single_image_pair_color_flow_calc(img1, img2)
        else:
            return self.batch_flow_calc(img1, img2)
        
    def to_color(self, flow):
        if len(flow.shape) == 4:
            flow = flow[0]
        flow = flow.cpu().permute(1, 2, 0).numpy()
        flow_color = flow_to_color(flow, convert_to_bgr=True)
        return flow_color