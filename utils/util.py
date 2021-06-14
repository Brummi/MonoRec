import json
from functools import partial
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict

import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F


def map_fn(batch, fn):
    if isinstance(batch, dict):
        for k in batch.keys():
            batch[k] = map_fn(batch[k], fn)
        return batch
    elif isinstance(batch, list):
        return [map_fn(e, fn) for e in batch]
    else:
        return fn(batch)

def to(data, device):
    if isinstance(data, dict):
        return {k: to(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to(v, device) for v in data]
    else:
        return data.to(device)


eps = 1e-6

def preprocess_roi(depth_prediction, depth_gt: torch.Tensor, roi):
    if roi is not None:
        if isinstance(depth_prediction, list):
            depth_prediction = [dpr[:, :, roi[0]:roi[1], roi[2]:roi[3]] for dpr in depth_prediction]
        else:
            depth_prediction = depth_prediction[:, :, roi[0]:roi[1], roi[2]:roi[3]]
        depth_gt = depth_gt[:, :, roi[0]:roi[1], roi[2]:roi[3]]
    return depth_prediction, depth_gt


def get_absolute_depth(depth_prediction, depth_gt: torch.Tensor, max_distance=None):
    if max_distance is not None:
        if isinstance(depth_prediction, list):
            depth_prediction = [torch.clamp_min(dpr, 1 / max_distance) for dpr in depth_prediction]
        else:
            depth_prediction = torch.clamp_min(depth_prediction, 1 / max_distance)
        depth_gt = torch.clamp_min(depth_gt, 1 / max_distance)
    if isinstance(depth_prediction, list):
        return [1 / dpr for dpr in depth_prediction], 1 / depth_gt
    else:
        return 1 / depth_prediction, 1 / depth_gt


def get_positive_depth(depth_prediction: torch.Tensor, depth_gt: torch.Tensor):
    if isinstance(depth_prediction, list):
        depth_prediction = [torch.nn.functional.relu(dpr) for dpr in depth_prediction]
    else:
        depth_prediction = torch.nn.functional.relu(depth_prediction)
    depth_gt = torch.nn.functional.relu(depth_gt)
    return depth_prediction, depth_gt


def depthmap_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, flatten=False):
    n, c, h, w = depth.shape
    grid = DepthWarper._create_meshgrid(h, w).expand(n, -1, -1, -1).to(depth.device)
    points = pixel2cam(depth, torch.inverse(intrinsics), grid)
    if not flatten:
        return points
    else:
        return points.view(n, h * w, 3)


def save_frame_for_tsdf(dir: Path, index, keyframe, depth, pose, crop=None, min_distance=None, max_distance=None):
    if crop is not None:
        keyframe = keyframe[:, crop[0]:crop[1], crop[2]:crop[3]]
        depth = depth[crop[0]:crop[1], crop[2]:crop[3]]
    keyframe = ((keyframe + .5) * 255).to(torch.uint8).permute(1, 2, 0)
    depth = (1 / depth * 100).to(torch.int16)
    depth[depth < 0] = 0
    if min_distance is not None:
        depth[depth < min_distance * 100] = 0
    if max_distance is not None:
        depth[depth > max_distance * 100] = 0
    Image.fromarray(keyframe.numpy()).save(dir / f"frame-{index:06d}.color.jpg")
    Image.fromarray(depth.numpy()).save(dir / f"frame-{index:06d}.depth.png")
    np.savetxt(dir / f"frame-{index:06d}.pose.txt", torch.inverse(pose).numpy())


def save_intrinsics_for_tsdf(dir: Path, intrinsics, crop=None):
    if crop is not None:
        intrinsics[0, 2] -= crop[2]
        intrinsics[1, 2] -= crop[0]
    np.savetxt(dir / f"camera-intrinsics.txt", intrinsics[:3, :3].numpy())


def get_mask(pred: torch.Tensor, gt: torch.Tensor, max_distance=None, pred_all_valid=True):
    mask = gt == 0
    if max_distance:
        mask |= (gt < 1 / max_distance)
    if not pred_all_valid:
        mask |= pred == 0
    return mask


def mask_mean(t: torch.Tensor, m: torch.Tensor, dim=None):
    t = t.clone()
    t[m] = 0
    els = 1
    if dim is None:
        dim = list(range(len(t.shape)))
    for d in dim:
        els *= t.shape[d]
    return torch.sum(t, dim=dim) / (els - torch.sum(m.to(torch.float), dim=dim))


def conditional_flip(x, condition, inplace=True):
    if inplace:
        x[condition, :, :, :] = x[condition, :, :, :].flip(3)
    else:
        flipped_x = x.clone()
        flipped_x[condition, :, :, :] = x[condition, :, :, :].flip(3)
        return flipped_x


def create_mask(c: int, height: int, width: int, border_radius: int, device):
    mask = torch.ones(c, 1, height - 2 * border_radius, width - 2 * border_radius, device=device)
    return torch.nn.functional.pad(mask, [border_radius, border_radius, border_radius, border_radius])


def median_scaling(data_dict):
    target = data_dict["target"]
    prediction = data_dict["result"]
    mask = target > 0
    ratios = mask.new_tensor([torch.median(target[i, mask[i]]) / torch.median(prediction[i, mask[i]]) for i in range(target.shape[0])], dtype=torch.float32)
    data_dict = dict(data_dict)
    data_dict["result"] = prediction * ratios.view(-1, 1, 1, 1)
    return data_dict


unsqueezer = partial(torch.unsqueeze, dim=0)


class DS_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, start=0, end=-1, every_nth=1):
        super().__init__()
        self.dataset = dataset
        self.start = start
        if end == -1:
            self.end = len(self.dataset)
        else:
            self.end = end
        self.every_nth = every_nth

    def __getitem__(self, index: int):
        return self.dataset.__getitem__(index * self.every_nth + self.start)

    def __len__(self):
        return (self.end - self.start) // self.every_nth + (1 if (self.end - self.start) % self.every_nth != 0 else 0)

class DS_Merger(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, index: int):
        return (ds.__getitem__(index + self.start) for ds in self.datasets)

    def __len__(self):
        return len(self.datasets[0])


class LossWrapper(torch.nn.Module):
    def __init__(self, loss_function, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.loss_function = loss_function
        self.num_devices = 1.0

    def forward(self, data):
        loss_dict = self.loss_function(data, **self.kwargs)
        loss_dict = map_fn(loss_dict, lambda x: (x / self.num_devices))
        if loss_dict["loss"].requires_grad:
            loss_dict["loss"].backward()
        loss_dict["loss"].detach_()
        return data, loss_dict


class ValueFader:
    def __init__(self, steps, values):
        self.steps = steps
        self.values = values
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_value(self, epoch=None):
        if epoch is None:
            epoch = self.epoch
        if epoch >= self.steps[-1]:
            return self.values[-1]

        step_index = 0

        while step_index < len(self.steps)-1 and epoch >= self.steps[step_index+1]:
            step_index += 1

        p = float(epoch - self.steps[step_index]) / float(self.steps[step_index+1] - self.steps[step_index])
        return (1-p) * self.values[step_index] + p * self.values[step_index+1]


def pose_distance_thresh(data_dict, spatial_thresh=.6, rotational_thresh=.05):
    poses = torch.stack([data_dict["keyframe_pose"]] + data_dict["poses"], dim=1)
    forward = poses.new_tensor([0, 0, 1], dtype=torch.float32)
    spatial_expanse = torch.norm(torch.max(poses[..., :3, 3], dim=1)[0] - torch.min(poses[..., :3, 3], dim=1)[0], dim=1)
    rotational_expanse = torch.norm(torch.max(poses[..., :3, :3] @ forward, dim=1)[0] - torch.min(poses[..., :3, :3] @ forward, dim=1)[0], dim=1)
    return (spatial_expanse > spatial_thresh) | (rotational_expanse > rotational_thresh)


def dilate_mask(m: torch.Tensor, size: int = 15):
    k = m.new_ones((1, 1, size, size), dtype=torch.float32)
    dilated_mask = F.conv2d((m >= 0.5).to(dtype=torch.float32), k, padding=(size//2, size//2))
    return dilated_mask > 0


def operator_on_dict(dict_0: dict, dict_1: dict, operator, default=0):
    keys = set(dict_0.keys()).union(set(dict_1.keys()))
    results = {}
    for k in keys:
        v_0 = dict_0[k] if k in dict_0 else default
        v_1 = dict_1[k] if k in dict_1 else default
        results[k] = operator(v_0, v_1)
    return results


numbers = [f"{i:d}" for i in range(1, 10, 1)]


def filter_state_dict(state_dict, data_parallel=False):
    if data_parallel:
        state_dict = {k[7:]: state_dict[k] for k in state_dict}
    state_dict = {(k[2:] if k.startswith("0") else k): state_dict[k] for k in state_dict if not k[0] in numbers}
    return state_dict


def seed_rng(seed):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(0)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
