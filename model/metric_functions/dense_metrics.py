import torch

from utils import preprocess_roi, get_positive_depth, get_absolute_depth


def sc_inv_metric(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, roi=None, max_distance=None):
    """
    Computes scale inveriant metric described in (14)
    :param depth_prediction: Depth prediction computed by the network
    :param depth_gt: GT Depth
    :param roi: Specify a region of interest on which the metric should be computed
    :return: metric (mean over batch_size)
    """
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    n = depth_gt.shape[2] * depth_gt.shape[3]
    E = torch.log(depth_prediction) - torch.log(depth_gt)
    E[torch.isnan(E)] = 0
    batch_metric = torch.sqrt(1 / n * torch.sum(E**2, dim=[2, 3]) - 1 / (n**2) * (torch.sum(E, dim=[2, 3])**2))
    batch_metric[torch.isnan(batch_metric)] = 0
    result = torch.mean(batch_metric)
    return result


def l1_rel_metric(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, roi=None, max_distance=None):
    """
    Computes the L1-rel metric described in (15)
    :param depth_prediction: Depth prediction computed by the network
    :param depth_gt: GT Depth
    :param roi: Specify a region of interest on which the metric should be computed
    :return: metric (mean over batch_size)
    """
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    return torch.mean(torch.abs(depth_prediction - depth_gt) / depth_gt)


def l1_inv_metric(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, roi=None, max_distance=None):
    """
    Computes the L1-inv metric described in (16)
    :param depth_prediction: Depth prediction computed by the network
    :param depth_gt: GT Depth
    :param roi: Specify a region of interest on which the metric should be computed
    :return: metric (mean over batch_size)
    """
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)


    return torch.mean(torch.abs(depth_prediction - depth_gt))