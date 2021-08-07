import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from model.layers import Backprojection, point_projection, ssim
from utils import create_mask, mask_mean


def compute_errors(img0, img1, mask=None):
    errors = .85 * torch.mean(ssim(img0, img1, pad_reflection=False, gaussian_average=True, comp_mode=True), dim=1) + .15 * torch.mean(torch.abs(img0 - img1), dim=1)
    if mask is not None: return errors, mask
    else: return errors


def reprojection_loss(depth_prediction: torch.Tensor, data_dict, automasking=False,
                      error_function=compute_errors, error_function_weight=None, use_mono=True, use_stereo=False,
                      reduce=True, combine_frames="min", mono_auto=False, border=0):
    keyframe = data_dict["keyframe"]
    keyframe_pose = data_dict["keyframe_pose"]
    keyframe_intrinsics = data_dict["keyframe_intrinsics"]

    frames = []
    poses = []
    intrinsics = []

    if use_mono:
        frames += data_dict["frames"]
        poses += data_dict["poses"]
        intrinsics += data_dict["intrinsics"]
    if use_stereo:
        frames += [data_dict["stereoframe"]]
        poses += [data_dict["stereoframe_pose"]]
        intrinsics += [data_dict["stereoframe_intrinsics"]]

    batch_size, channels, height, width = keyframe.shape
    frame_count = len(frames)
    keyframe_extrinsics = torch.inverse(keyframe_pose)
    extrinsics = [torch.inverse(pose) for pose in poses]

    reprojections = []
    if border > 0:
        masks = [create_mask(batch_size, height, width, border, keyframe.device) for _ in range(frame_count)]
        warped_masks = []

    backproject_depth = Backprojection(batch_size, height, width)
    backproject_depth.to(keyframe.device)

    for i, (frame, extrinsic, intrinsic) in enumerate(zip(frames, extrinsics, intrinsics)):
        cam_points = backproject_depth(1 / depth_prediction, torch.inverse(keyframe_intrinsics))
        pix_coords = point_projection(cam_points, batch_size, height, width, intrinsic, extrinsic @ keyframe_pose)
        reprojections.append(F.grid_sample(frame + 1.5, pix_coords, padding_mode="zeros"))
        if border > 0:
            warped_masks.append(F.grid_sample(masks[i], pix_coords, padding_mode="zeros"))

    reprojections = torch.stack(reprojections, dim=1).view(batch_size * frame_count, channels, height, width)
    mask = reprojections[:, 0, :, :] == 0
    reprojections -= 1.0

    if border > 0:
        mask = ~(torch.stack(warped_masks, dim=1).view(batch_size * frame_count, height, width) > .5)

    keyframe_expanded = (keyframe + .5).unsqueeze(1).expand(-1, frame_count, -1, -1, -1).reshape(batch_size * frame_count, channels, height, width)

    loss = 0

    if type(error_function) != list:
        error_function = [error_function]
    if error_function_weight is None:
        error_function_weight = [1 for i in range(len(error_function))]

    for ef, w in zip(error_function, error_function_weight):
        errors, n_mask = ef(reprojections, keyframe_expanded, mask)
        n_height, n_width = n_mask.shape[1:]
        errors = errors.view(batch_size, frame_count, n_height, n_width)

        n_mask = n_mask.view(batch_size, frame_count, n_height, n_width)
        errors[n_mask] = float("inf")

        if automasking:
            frames_stacked = torch.stack(frames, dim=1).view(batch_size * frame_count, channels, height, width) + .5
            errors_nowarp = ef(frames_stacked, keyframe_expanded).view(batch_size, frame_count, n_height, n_width)
            errors[errors_nowarp < errors] = float("inf")

        if mono_auto:
            keyframe_expanded_ = (keyframe + .5).unsqueeze(1).expand(-1, len(data_dict["frames"]), -1, -1, -1).reshape(batch_size * len(data_dict["frames"]), channels, height, width)
            frames_stacked = (torch.stack(data_dict["frames"], dim=1) + .5).view(batch_size * len(data_dict["frames"]), channels, height, width)
            errors_nowarp = ef(frames_stacked, keyframe_expanded_).view(batch_size, len(data_dict["frames"]), n_height, n_width)
            errors_nowarp = torch.mean(errors_nowarp, dim=1, keepdim=True)
            errors_nowarp[torch.all(n_mask, dim=1, keepdim=True)] = float("inf")
            errors = torch.min(errors, errors_nowarp.expand(-1, frame_count, -1, -1))

        if combine_frames == "min":
            errors = torch.min(errors, dim=1)[0]
            n_mask = torch.isinf(errors)
        elif combine_frames == "avg":
            n_mask = torch.isinf(errors)
            hits = torch.sum((~n_mask).to(dtype=torch.float32), dim=1)
            errors[n_mask] = 0
            errors = torch.sum(errors, dim=1) / hits
            n_mask = hits == 0
            errors[n_mask] = float("inf")
        elif combine_frames == "rnd":
            index = torch.randint(frame_count, (batch_size, 1, 1, 1), device=keyframe.device).expand(-1, 1, n_height, n_width)
            errors = torch.gather(errors, dim=1, index=index).squeeze(1)
            n_mask = torch.gather(n_mask, dim=1, index=index).squeeze(1)
        else:
            raise ValueError("Combine frames must be \"min\", \"avg\" or \"rnd\".")

        if reduce:
            loss += w * mask_mean(errors, n_mask)
        else:
            loss += w * errors
    return loss


def edge_aware_smoothness_loss(depth_prediction, input, reduce=True):
    keyframe = input["keyframe"]
    depth_prediction = depth_prediction / torch.mean(depth_prediction, dim=[2, 3], keepdim=True)

    d_dx = torch.abs(depth_prediction[:, :, :, :-1] - depth_prediction[:, :, :, 1:])
    d_dy = torch.abs(depth_prediction[:, :, :-1, :] - depth_prediction[:, :, 1:, :])

    k_dx = torch.mean(torch.abs(keyframe[:, :, :, :-1] - keyframe[:, :, :, 1:]), 1, keepdim=True)
    k_dy = torch.mean(torch.abs(keyframe[:, :, :-1, :] - keyframe[:, :, 1:, :]), 1, keepdim=True)

    d_dx *= torch.exp(-k_dx)
    d_dy *= torch.exp(-k_dy)

    if reduce:
        return d_dx.mean() + d_dy.mean()
    else:
        return  F.pad(d_dx, pad=(0, 1), mode='constant', value=0) + F.pad(d_dy, pad=(0, 0, 0, 1), mode='constant', value=0)


def sparse_depth_loss(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, l2=False, reduce=True):
    """
    :param depth_prediction:
    :param depth_gt: (N, 1, H, W)
    :return:
    """
    n, c, h, w = depth_prediction.shape
    mask = depth_gt == 0
    if not l2:
        errors = torch.abs(depth_prediction - depth_gt)
    else:
        errors = (depth_prediction - depth_gt) ** 2

    if reduce:
        loss = mask_mean(errors, mask)
        loss[torch.isnan(loss)] = 0
        return loss
    else:
        return errors, mask


def selfsup_loss(depth_prediction: torch.Tensor, input=None, scale=0, automasking=True, error_function=None, error_function_weight=None, use_mono=True, use_stereo=False, reduce=True, combine_frames="min", mask_border=0):
    reprojection_l = reprojection_loss(depth_prediction, input, automasking=automasking, error_function=error_function, error_function_weight=error_function_weight, use_mono=use_mono, use_stereo=use_stereo, reduce=reduce, combine_frames=combine_frames, border=mask_border)
    reprojection_l[torch.isnan(reprojection_l)] = 0
    edge_aware_smoothness_l = edge_aware_smoothness_loss(depth_prediction, input)
    edge_aware_smoothness_l[torch.isnan(edge_aware_smoothness_l)] = 0
    loss = reprojection_l + edge_aware_smoothness_l * 1e-3 / (2 ** scale)
    return loss


class PerceptualError(nn.Module):
    def __init__(self, small_features=False):
        super().__init__()
        self.small_features = small_features
        vgg16 = torchvision.models.vgg16(True, True)
        self.feature_extractor = nn.Sequential(*list(vgg16.features.children())[:4 if self.small_features else 9])
        for p in self.feature_extractor.parameters(True):
            p.requires_grad_(False)
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, mask=None):
        n, c, h, w = img0.shape
        c = torchvision.models.vgg.cfgs["D"][1 if self.small_features else 4]
        if not self.small_features:
            h //= 2
            w //= 2

        img0 = (img0 - self.mean) / self.std
        img1 = (img1 - self.mean) / self.std

        if mask is not None:
            img0[mask.unsqueeze(1).expand(-1, 3, -1, -1)] = 0
            img1[mask.unsqueeze(1).expand(-1, 3, -1, -1)] = 0

        input = torch.cat([img0, img1], dim=0)
        features = self.feature_extractor(input)
        features = features.view(2, n, c, h, w)
        errors = torch.mean((features[1] - features[0]) ** 2, dim=1)

        if mask is not None:
            if not self.small_features:
                mask = F.upsample(mask.to(dtype=torch.float).unsqueeze(1), (h, w), mode="bilinear").squeeze(1)
            mask = mask > 0
            return errors, mask
        else:
            return errors
