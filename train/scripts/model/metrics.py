"""
Metrics implementation for 3D human pose comparisons
"""

import torch
from torch import nn
from einops import rearrange

__all__ = ['Accuracy', 'BinaryMatch', 'BinaryMatchF1', 'MeanRatio']


class BaseMetric(nn.Module):
    def forward(self, pred, gt, gt_mask=None):
        """
        Base forward method for metric evaluation
        Args:
            pred: predicted voxel
            gt: ground truth voxel
            gt_mask: ground truth mask

        Returns:
            Metric as single value
        """
        pass


class Accuracy(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, pred, y):
        y_label = torch.argmax(y, dim=-1)
        pred_label = torch.argmax(pred, dim=-1)

        acc = (y_label == pred_label).double().mean()
        return acc


class BinaryMatch(BaseMetric):
    def __init__(self, op_type='raw', **kwargs):
        super().__init__(**kwargs)
        self.op_type = op_type
        assert self.op_type in ['raw', 'sum_c', 'sum_cp']

    def forward(self, pred, y):
        if self.op_type == 'sum_c':
            pred = rearrange(pred, 'b l (p c) h w -> b l c p h w', p=2)
            y = rearrange(y, 'b l (p c) h w -> b l c p h w', p=2)
            y = torch.sum(y, dim=2)
            pred = torch.sum(pred, dim=2)
        elif self.op_type == 'sum_cp':
            pred = torch.sum(pred, dim=2)
            y = torch.sum(y, dim=2)

        # for all pixels in pred, if pred > 0.01, pred_binary = 1, else 0

        pred_binary = torch.where(pred > 0.01, torch.ones_like(pred), torch.zeros_like(pred))
        label_binary = torch.where(y > 0.01, torch.ones_like(y), torch.zeros_like(y))
        

        acc = (pred_binary == label_binary).double().mean()
        return acc


def f1score(pred, y):
    """
    Compute F1 score for binary classification
    Args:
        pred: predicted voxel
        y: ground truth voxel
    Returns:
        f1: F1 score
    """
    TP = (pred * y).sum()

    # 计算 False Positive (FP)：预测为正类但目标为负类的数量
    FP = (pred * (1 - y)).sum()

    # 计算 False Negative (FN)：预测为负类但目标为正类的数量
    FN = ((1 -pred) * y).sum()

    # TN=((1-pred)*(1-y)).sum()

    # 计算 Precision 和 Recall
    precision = TP / (TP + FP + 1e-8)  # 加上 1e-8 避免除数为 0
    recall = TP / (TP + FN + 1e-8)

    # 计算 F1 分数
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


class BinaryMatchF1(BaseMetric):
    def __init__(self, threshold=0.01, op_type='sum_cp', **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.op_type = op_type
        assert self.op_type in ['raw', 'sum_c', 'sum_cp']

    def forward(self, pred, y):
        if self.op_type == 'sum_c':
            pred = rearrange(pred, 'b l (p c) h w -> b l c p h w', p=2)
            y = rearrange(y, 'b l (p c) h w -> b l c p h w', p=2)
            y = torch.sum(y, dim=2)
            pred = torch.sum(pred, dim=2)
        elif self.op_type == 'sum_cp':
            pred = torch.sum(pred, dim=2)
            y = torch.sum(y, dim=2)
        
        y = torch.where(y > self.threshold, torch.ones_like(y), torch.zeros_like(y)).to(y.device)
        pred = torch.where(pred > self.threshold, torch.ones_like(pred), torch.zeros_like(pred)).to(pred.device)
        
        f1 = f1score(pred, y)
        return f1
        

class PoolMSE(nn.Module):
    def __init__(self, kernel_size=2):
        super(PoolMSE, self).__init__()
        self.score = nn.MSELoss()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool3d(kernel_size, stride=kernel_size)

    def forward(self, pred, target):
        pred = rearrange(pred, 'b l (p c) h w -> (b p) (l c) h w', p=2)
        target = rearrange(target, 'b l (p c) h w -> (b p) (l c) h w', p=2)
        score = self.score(self.pool(pred), self.pool(target))
        return score


class MeanRatio(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, pred, y):
        ratio = (pred + 0.01) / (y + 0.01)
        # if ratio < 1, ratio = 1/ratio
        ratio = torch.where(ratio < 1, 1/ratio, ratio)
        ratio = ratio.mean()
        return ratio