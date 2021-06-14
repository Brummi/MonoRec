import torch

from utils import mask_mean


def completeness_metric(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, roi=None, max_distance=None):
    return torch.mean((depth_prediction != 0).to(dtype=torch.float32))


def covered_gt_metric(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, roi=None, max_distance=None):
    gt_mask = depth_gt != 0
    return mask_mean(((depth_prediction != 0)).to(dtype=torch.float32), gt_mask)
