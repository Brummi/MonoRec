import torch
from torch.nn import functional as F

from utils import mask_mean
from .common_losses import compute_errors, sparse_depth_loss, edge_aware_smoothness_loss, reprojection_loss, \
    selfsup_loss


def depth_loss(data_dict, alpha=None, roi=None, options=()):
    loss_dict = {}

    use_mono=True
    use_stereo=False
    automasking = True
    combine_frames = "min"
    mask_border = 0
    error_function = compute_errors

    if "stereo" in options:
        use_stereo = True

    depth_gt = data_dict["target"]
    depth_predictions = data_dict["predicted_inverse_depths"]

    depth_gt[depth_gt <= 0.0] = 0.0
    depth_gt[depth_gt >= 100.] = 100

    if alpha is None:
        alpha = 0.5

    loss = 0
    sdl_sum = 0
    md2l_sum = 0
    for i, depth_prediction in enumerate(depth_predictions):
        depth_prediction[depth_prediction <= 0.0] = 0.0
        if depth_prediction.shape[2] != depth_gt.shape[2]:
            depth_prediction = torch.nn.functional.upsample(depth_prediction, (depth_gt.shape[2], depth_gt.shape[3]))
        sdl = sparse_depth_loss(depth_prediction, depth_gt, l2=False)
        md2l = selfsup_loss(depth_prediction, data_dict, scale=i, use_mono=use_mono, use_stereo=use_stereo, automasking=automasking, error_function=error_function, combine_frames=combine_frames, mask_border=mask_border)
        sdl_sum += sdl
        md2l_sum += md2l
        loss_dict[f"md2l_{i}"] = md2l
        loss_dict[f"sdl_{i}"] = sdl
    loss += 2 * alpha * 4 * sdl_sum + 2 * (1 - alpha) * md2l_sum
    loss_dict["loss"] = loss

    return loss_dict


def mask_loss(data_dict, alpha=None, roi=None, options=()):
    gt_mask: torch.Tensor = data_dict["mvobj_mask"]
    cv_mask = data_dict["cv_mask"]

    mvg_pixels = 2741312
    total_pixels = 338034688
    mvg_frames = 2579
    mvg_ratio = 0.008109558
    # mvg_ratio = 0.10

    weight = gt_mask.clone().detach()
    weight[gt_mask > 0] = 1 / mvg_ratio
    weight[gt_mask == 0] = 1 / (1 - mvg_ratio)
    # weight[gt_mask != 0] = (1 - mvg_ratio)
    # weight[gt_mask == 0] = mvg_ratio

    if "multiplicative_weight_mask" in data_dict:
        weight *= data_dict["multiplicative_weight_mask"]

    loss = F.binary_cross_entropy(cv_mask, gt_mask.to(dtype=torch.float32), weight=weight)

    gt_pred = gt_mask > .5
    cv_pred = cv_mask > .5

    intersection = torch.sum(cv_pred & gt_pred, dtype=torch.float32, dim=[1, 2, 3])
    union = torch.sum(cv_pred | gt_pred, dtype=torch.float32, dim=[1, 2, 3])
    gt_sum = torch.sum(gt_pred, dtype=torch.float32, dim=[1, 2, 3])
    cv_sum = torch.sum(cv_pred, dtype=torch.float32, dim=[1, 2, 3])

    acc = torch.mean((cv_pred == gt_pred).to(dtype=torch.float32))
    prec = intersection / cv_sum
    prec[cv_sum == 0] = 1 - intersection.clamp(0, 1)[cv_sum == 0]
    prec = prec.mean()
    rec = intersection / gt_sum
    rec[gt_sum == 0] = 1 - intersection.clamp(0, 1)[gt_sum == 0]
    rec = rec.mean()
    iou = intersection / union
    iou[union == 0] = 1
    iou = iou.mean()

    return {
        "loss": loss,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "iou": iou
    }


def mask_refinement_loss(data_dict, alpha=None, roi=None, options=()):
    loss_dict = {}

    depth_gt = data_dict["target"]
    mono_preds = data_dict["mono_pred"]
    stereo_preds = data_dict["stereo_pred"]
    cv_mask = data_dict["cv_mask"]
    if "mvobj_mask" in data_dict:
        gt_mask = data_dict["mvobj_mask"] > .5
    inv_depth_min = data_dict["inv_depth_min"]
    inv_depth_max = data_dict["inv_depth_max"]
    inv_depth_range = inv_depth_min - inv_depth_max

    bias = 1.0

    depth_gt[depth_gt <= 0.0] = 0.0
    depth_gt[depth_gt >= 100.] = 100

    if alpha is None:
        alpha = 0.5

    loss = 0
    mono_sdl_sum = 0
    stereo_sdl_sum = 0
    sdl_sum = 0

    md2l_sum = 0

    mask_loss_value = 0

    gt_pred = gt_mask > .5
    cv_pred = cv_mask > .5

    intersection = torch.sum(cv_pred & gt_pred, dtype=torch.float32, dim=[1, 2, 3])
    union = torch.sum(cv_pred | gt_pred, dtype=torch.float32, dim=[1, 2, 3])
    gt_sum = torch.sum(gt_pred, dtype=torch.float32, dim=[1, 2, 3])
    cv_sum = torch.sum(cv_pred, dtype=torch.float32, dim=[1, 2, 3])

    acc = torch.mean((cv_pred == gt_pred).to(dtype=torch.float32))
    prec = intersection / cv_sum
    prec[cv_sum == 0] = 1 - intersection.clamp(0, 1)[cv_sum == 0]
    prec = prec.mean()
    rec = intersection / gt_sum
    rec[gt_sum == 0] = 1 - intersection.clamp(0, 1)[gt_sum == 0]
    rec = rec.mean()
    iou = intersection / union
    iou[union == 0] = 1
    iou = iou.mean()

    loss_dict["acc"] = acc
    loss_dict["prec"] = prec
    loss_dict["rec"] = rec
    loss_dict["iou"] = iou

    for scale, (mono_pred, stereo_pred) in enumerate(zip(mono_preds, stereo_preds)):
        if mono_pred.shape[2] != depth_gt.shape[2]:
            mono_pred = torch.nn.functional.upsample(mono_pred, (depth_gt.shape[2], depth_gt.shape[3]))
            stereo_pred = torch.nn.functional.upsample(stereo_pred, (depth_gt.shape[2], depth_gt.shape[3]))

        mono_sdl, mono_sdl_mask = sparse_depth_loss(mono_pred, depth_gt, l2=False, reduce=False)
        stereo_sdl, stereo_sdl_mask = sparse_depth_loss(stereo_pred, depth_gt, l2=False, reduce=False)

        mono_sdl_sum += mask_mean(mono_sdl.detach(), mono_sdl_mask)
        stereo_sdl_sum += mask_mean(stereo_sdl.detach(), stereo_sdl_mask)

        sdl = mask_mean(mono_sdl * (1 - cv_mask), mono_sdl_mask) + mask_mean(stereo_sdl * cv_mask, stereo_sdl_mask) / bias
        sdl_sum += sdl
        loss_dict[f"sdl_{scale}"] = sdl

        if "dist_diff_loss" in options:
            b = 16 // (2 ** scale)
            mono_thresh = (mono_pred.detach() < (inv_depth_range / 32 * 2 + inv_depth_max))
            # stereo_thresh = (stereo_pred.detach() >= (inv_depth_range / 32 * 2 + inv_depth_max))
            dist_diff_mask = mono_thresh & gt_mask
            dist_diff_mask = F.conv2d(dist_diff_mask.to(dtype=torch.float32), dist_diff_mask.new_ones((1, 1, b+1, b+1), requires_grad=False, dtype=torch.float32), padding=b//2) >= ((b+1) ** 2) / 4
            dist_diff_loss = (-torch.log(cv_mask[:, :, b*4:-b, b:-b][dist_diff_mask[:, :, b*4:-b, b:-b]])).sum() / torch.clamp_min(dist_diff_mask[:, :, b*4:-b, b:-b].to(dtype=torch.float32).sum(), 1) * (2**(-3))
            loss_dict[f"dist_diff_{scale}"] = dist_diff_loss
            mask_loss_value += dist_diff_loss
            mult_weight_mask = mono_thresh.new_ones(mono_thresh.shape, dtype=torch.float32)
            mult_weight_mask[mono_thresh & ~gt_mask] = 1e-3
            data_dict["multiplicative_weight_mask"] = mult_weight_mask

        mono_smoothness = edge_aware_smoothness_loss(mono_pred, data_dict, reduce=False)
        stereo_smoothness = edge_aware_smoothness_loss(stereo_pred, data_dict, reduce=False)
        smoothness = (mono_smoothness * (1 - cv_mask) + stereo_smoothness * cv_mask / bias).mean()

        mono_repr_l = reprojection_loss(mono_pred, data_dict, use_mono=True, use_stereo=False,
                                        automasking=False, error_function=compute_errors, reduce=False, combine_frames="min", mono_auto=False).unsqueeze(1)
        stereo_repr_l = reprojection_loss(stereo_pred, data_dict, use_mono=False, use_stereo=True,
                                          automasking=False, error_function=compute_errors, reduce=False, combine_frames="min", mono_auto=False, border=3).unsqueeze(1)

        mono_mask = torch.isinf(mono_repr_l)
        stereo_mask = torch.isinf(stereo_repr_l)
        mono_repr_l[mono_mask] = 0
        stereo_repr_l[stereo_mask] = 0

        loss_dict[f"static_md2l_{scale}"] = mask_mean(mono_repr_l, mono_mask)
        loss_dict[f"dynamic_md2l_{scale}"] = mask_mean(stereo_repr_l, stereo_mask)

        mono_repr_l = mono_repr_l * torch.max(1 - cv_mask, stereo_mask.to(dtype=torch.float32))
        stereo_repr_l = stereo_repr_l * torch.max(cv_mask, mono_mask.to(dtype=torch.float32))

        repr_l = mask_mean(mono_repr_l + stereo_repr_l / bias, mono_mask & stereo_mask)
        md2l = repr_l + smoothness * 1e-3 / (2 ** scale)
        loss_dict[f"md2l_{scale}"] = md2l
        md2l_sum += md2l

    if "mask_loss" in options:
        loss_dict_mask = mask_loss(data_dict)
        mask_loss_value = loss_dict_mask["loss"]
        del loss_dict_mask["loss"]
        loss_dict_mask["mask_loss"] = mask_loss_value * 4
        for k in loss_dict_mask.keys():
            loss_dict[k] = loss_dict_mask[k]
    else:
        mask_loss_value = 0

    loss += 2 * alpha * 4 * sdl_sum + 2 * (1 - alpha) * md2l_sum + mask_loss_value
    loss_dict["loss"] = loss

    return loss_dict


def depth_aux_mask_loss(data_dict, alpha=None, roi=None, options=()):
    # This loss function is never used in the paper

    loss_dict = {}

    depth_gt = data_dict["target"]
    mono_preds = data_dict["mono_pred"]
    cv_mask = data_dict["cv_mask"]
    if "mvobj_mask" in data_dict:
        gt_mask = data_dict["mvobj_mask"] > .5
    inv_depth_min = data_dict["inv_depth_min"]
    inv_depth_max = data_dict["inv_depth_max"]
    inv_depth_range = inv_depth_min - inv_depth_max

    cv_mask = cv_mask.detach() > .5

    depth_gt[depth_gt <= 0.0] = 0.0
    depth_gt[depth_gt >= 100.] = 100

    if alpha is None:
        alpha = 0.5

    loss = 0
    mono_sdl_sum = 0
    sdl_sum = 0

    md2l_sum = 0

    for scale, mono_pred in enumerate(mono_preds):
        if mono_pred.shape[2] != depth_gt.shape[2]:
            mono_pred = torch.nn.functional.upsample(mono_pred, (depth_gt.shape[2], depth_gt.shape[3]))

        mono_sdl, mono_sdl_mask = sparse_depth_loss(mono_pred, depth_gt, l2=False, reduce=False)
        mono_sdl_sum += mask_mean(mono_sdl.detach(), mono_sdl_mask | cv_mask)

        sdl = mask_mean(mono_sdl, mono_sdl_mask | cv_mask)
        sdl_sum += sdl
        loss_dict[f"sdl_{scale}"] = sdl

        mono_smoothness = edge_aware_smoothness_loss(mono_pred, data_dict, reduce=False)
        smoothness = mask_mean(mono_smoothness, cv_mask)

        mono_repr_l = reprojection_loss(mono_pred, data_dict, use_mono=True, use_stereo=False,
                                        automasking=False, error_function=compute_errors, reduce=False, combine_frames="min", mono_auto=False).unsqueeze(1)

        mono_mask = torch.isinf(mono_repr_l)
        mono_repr_l[mono_mask] = 0

        loss_dict[f"static_md2l_{scale}"] = mask_mean(mono_repr_l, mono_mask)

        repr_l = mask_mean(mono_repr_l, mono_mask | cv_mask)
        md2l = repr_l + smoothness * 1e-3 / (2 ** scale)
        loss_dict[f"md2l_{scale}"] = md2l
        md2l_sum += md2l

    loss += 2 * alpha * 4 * sdl_sum + 2 * (1 - alpha) * md2l_sum
    loss_dict["loss"] = loss

    return loss_dict


def depth_refinement_loss(data_dict, alpha=None, roi=None, options=()):
    loss_dict = {}

    use_stereo = "stereo" in options
    use_stereo_reprl = "stereo_repr" in options
    use_mono_stereodl = not ("no_mono_stereodl" in options)

    depth_gt = data_dict["target"]
    mono_preds = data_dict["mono_pred"]
    if use_mono_stereodl:
        stereo_preds = data_dict["stereo_pred"]
    else:
        stereo_preds = len(mono_preds) * [None]
    cv_mask = data_dict["cv_mask"]
    if "mvobj_mask" in data_dict:
        gt_mask = data_dict["mvobj_mask"] > .5
    inv_depth_min = data_dict["inv_depth_min"]
    inv_depth_max = data_dict["inv_depth_max"]
    inv_depth_range = inv_depth_min - inv_depth_max

    bias = 1.0

    depth_gt[depth_gt <= 0.0] = 0.0
    depth_gt[depth_gt >= 100.] = 100

    cv_mask_discrete = (cv_mask > .5).to(dtype=torch.float32)
    ratio = cv_mask_discrete.sum() / cv_mask_discrete.numel()

    if alpha is None:
        alpha = 0.5

    loss = 0
    mono_sdl_sum = 0
    stereo_sdl_sum = 0
    sdl_sum = 0

    md2l_sum = 0

    mask_loss_value = 0

    for scale, (mono_pred, stereo_pred) in enumerate(zip(mono_preds, stereo_preds)):
        if mono_pred.shape[2] != depth_gt.shape[2]:
            mono_pred = torch.nn.functional.upsample(mono_pred, (depth_gt.shape[2], depth_gt.shape[3]))
            if use_mono_stereodl:
                stereo_pred = torch.nn.functional.upsample(stereo_pred, (depth_gt.shape[2], depth_gt.shape[3]))

        if use_mono_stereodl:
            stereo_pred = stereo_pred.detach()

        # Depth losses
        mono_sparsedl, mono_sparsedl_mask = sparse_depth_loss(mono_pred, depth_gt * (1 - cv_mask_discrete), l2=False, reduce=False)
        mono_sdl = mask_mean(mono_sparsedl.detach(), mono_sparsedl_mask)
        mono_sdl_sum += mono_sdl.detach()

        if use_mono_stereodl:
            mono_stereodl, mono_stereodl_mask = sparse_depth_loss(mono_pred, stereo_pred * cv_mask_discrete, l2=False, reduce=False)
            stereo_sdl = mask_mean(mono_stereodl.detach(), mono_stereodl_mask)
            stereo_sdl_sum += stereo_sdl.detach()
        else:
            stereo_sdl = 0

        sdl = mono_sdl * (1 - ratio) + stereo_sdl * ratio * 4

        sdl_sum += sdl
        loss_dict[f"sdl_{scale}"] = sdl

        # Smoothness loss

        smoothness = edge_aware_smoothness_loss(mono_pred, data_dict, reduce=False)

        # Reprojection losses

        mono_repr_l = reprojection_loss(mono_pred, data_dict, use_mono=True, use_stereo=use_stereo, automasking=True, error_function=compute_errors, reduce=False, combine_frames="min", mono_auto=False).unsqueeze(1)
        mono_mask = torch.isinf(mono_repr_l) | (cv_mask_discrete > .5)
        mono_repr_l[mono_mask] = 0
        mono_repr_l = mask_mean(mono_repr_l, mono_mask)

        if use_stereo_reprl:
            stereo_repr_l = reprojection_loss(mono_pred, data_dict, use_mono=False, use_stereo=True, automasking=False, error_function=compute_errors, reduce=False, combine_frames="min", mono_auto=False, border=3).unsqueeze(1)
            stereo_mask = torch.isinf(stereo_repr_l) | (cv_mask_discrete <= .5)
            stereo_repr_l[stereo_mask] = 0
            stereo_repr_l = mask_mean(stereo_repr_l, stereo_mask)
        else:
            stereo_repr_l = mono_repr_l.new_zeros(mono_repr_l.shape)

        loss_dict[f"static_md2l_{scale}"] = mono_repr_l.detach()
        loss_dict[f"dynamic_md2l_{scale}"] = stereo_repr_l

        repr_l = mono_repr_l * (1 - ratio) + stereo_repr_l * ratio * (1 if use_mono_stereodl else 1)
        md2l = repr_l + smoothness * 1e-3 / (2 ** scale)
        loss_dict[f"md2l_{scale}"] = md2l
        md2l_sum += md2l

    loss += 2 * alpha * 4 * sdl_sum + 2 * (1 - alpha) * md2l_sum + mask_loss_value
    loss_dict["loss"] = loss

    return loss_dict