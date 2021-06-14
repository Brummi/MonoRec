import torch

from utils import preprocess_roi, get_positive_depth, get_absolute_depth, get_mask, mask_mean


def a1_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt))
    return torch.mean((thresh < 1.25).type(torch.float))


def a2_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt)).type(torch.float)
    return torch.mean((thresh < 1.25 ** 2).type(torch.float))


def a3_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt)).type(torch.float)
    return torch.mean((thresh < 1.25 ** 3).type(torch.float))


def rmse_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    se = (depth_prediction - depth_gt) ** 2
    return torch.mean(torch.sqrt(torch.mean(se, dim=[1, 2, 3])))


def rmse_log_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    sle = (torch.log(depth_prediction) - torch.log(depth_gt)) ** 2
    return torch.mean(torch.sqrt(torch.mean(sle, dim=[1, 2, 3])))


def abs_rel_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    return torch.mean(torch.abs(depth_prediction - depth_gt) / depth_gt)


def sq_rel_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    return torch.mean(((depth_prediction - depth_gt) ** 2) / depth_gt)


def a1_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return a1_base(depth_prediction, depth_gt, mask)


def a2_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return a2_base(depth_prediction, depth_gt, mask)


def a3_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return a3_base(depth_prediction, depth_gt, mask)


def rmse_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return rmse_base(depth_prediction, depth_gt, mask)


def rmse_log_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return rmse_log_base(depth_prediction, depth_gt, mask)


def abs_rel_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return abs_rel_base(depth_prediction, depth_gt, mask)


def sq_rel_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return sq_rel_base(depth_prediction, depth_gt, mask)


def a1_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return a1_sparse_metric(data_dict, roi, max_distance, False)


def a2_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return a2_sparse_metric(data_dict, roi, max_distance, False)


def a3_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return a3_sparse_metric(data_dict, roi, max_distance, False)


def rmse_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return rmse_sparse_metric(data_dict, roi, max_distance, False)


def rmse_log_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return rmse_log_sparse_metric(data_dict, roi, max_distance, False)


def abs_rel_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return abs_rel_sparse_metric(data_dict, roi, max_distance, False)


def sq_rel_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return sq_rel_sparse_metric(data_dict, roi, max_distance, False)


def a1_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return a1_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def a2_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return a2_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def a3_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return a3_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def rmse_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return rmse_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def rmse_log_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return rmse_log_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def abs_rel_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return abs_rel_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def sq_rel_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return sq_rel_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def a1_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt))
    return mask_mean((thresh < 1.25).type(torch.float), mask)


def a2_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    depth_gt[mask] = 1
    depth_prediction[mask] = 1
    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt)).type(torch.float)
    return mask_mean((thresh < 1.25 ** 2).type(torch.float), mask)


def a3_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    depth_gt[mask] = 1
    depth_prediction[mask] = 1
    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt)).type(torch.float)
    return mask_mean((thresh < 1.25 ** 3).type(torch.float), mask)


def rmse_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    depth_gt[mask] = 1
    depth_prediction[mask] = 1
    se = (depth_prediction - depth_gt) ** 2
    return torch.mean(torch.sqrt(mask_mean(se, mask, dim=[1, 2, 3])))


def rmse_log_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    depth_gt[mask] = 1
    depth_prediction[mask] = 1
    sle = (torch.log(depth_prediction) - torch.log(depth_gt)) ** 2
    return torch.mean(torch.sqrt(mask_mean(sle, mask, dim=[1, 2, 3])))


def abs_rel_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    return mask_mean(torch.abs(depth_prediction - depth_gt) / depth_gt, mask)


def sq_rel_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    return mask_mean(((depth_prediction - depth_gt) ** 2) / depth_gt, mask)