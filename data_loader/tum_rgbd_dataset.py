from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import os
if os.path.exists(os.path.abspath(os.path.join(__file__, os.pardir, 'oxford_robotcar'))):
    from data_loader.oxford_robotcar.interpolate_poses import interpolate_poses
from utils import map_fn


class TUMRGBDDataset(Dataset):

    # _intrinsics = torch.tensor(
    #     [[517.3, 0, 318.6, 0],
    #      [0, 516.5, 255.3, 0],
    #      [0, 0, 1, 0],
    #      [0, 0, 0, 1]
    #     ], dtype=torch.float32)
    _intrinsics = torch.tensor(
        [[535.4, 0, 320.1, 0],
         [0, 539.2, 247.6, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]
         ], dtype=torch.float32)
    _depth_scale = 1.035 / 5000.

    _swapaxes = torch.tensor([[[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]]], dtype=torch.float32)
    _swapaxes_ = torch.inverse(_swapaxes)

    def __init__(self, dataset_dir, frame_count=2, target_image_size=(480, 640), dilation=1):
        """
        Dataset implementation for TUM RGBD.
        """
        self.dataset_dir = Path(dataset_dir)
        self.frame_count = frame_count
        self.dilation = dilation
        self.target_image_size = target_image_size

        (rgb_times, self._rgb_paths) = self.load_file_times(self.dataset_dir / "rgb.txt")
        (pose_times, self._raw_poses) = self.load_pose_times(self.dataset_dir / "groundtruth.txt")
        (depth_times, self._depth_paths) = self.load_file_times(self.dataset_dir / "depth.txt")

        self._image_index = self.build_image_index(rgb_times, pose_times, depth_times)
        self._poses = self.build_pose(pose_times, self._raw_poses, rgb_times)

        self._offset = (frame_count // 2) * self.dilation
        self._length = self._image_index.shape[0] - frame_count * dilation

    def __getitem__(self, index: int):
        frame_count = self.frame_count
        offset = self._offset

        keyframe_intrinsics = self._intrinsics
        keyframe = self.open_image(index + offset)
        # keyframe_pose = self._raw_poses[self._image_index[index + offset, 0]]
        keyframe_pose = self._poses[index + offset]
        keyframe_depth = self.open_depth(index + offset)

        frames = [self.open_image(index + i) for i in range(0, (frame_count + 1) * self.dilation, self.dilation) if i != offset]
        intrinsics = [self._intrinsics for _ in range(frame_count)]
        # poses = [self._raw_poses[self._image_index[index + i, 0]] for i in range(0, (frame_count + 1) * self.dilation, self.dilation) if i != offset]
        poses = [self._poses[index + i] for i in range(0, (frame_count + 1) * self.dilation, self.dilation) if i != offset]

        data = {
            "keyframe": keyframe,
            "keyframe_pose": keyframe_pose,
            "keyframe_intrinsics": keyframe_intrinsics,
            "frames": frames,
            "poses": poses,
            "intrinsics": intrinsics,
            "sequence": torch.tensor([0]),
            "image_id": torch.tensor([index + offset])
        }
        return data, keyframe_depth

    def __len__(self) -> int:
        return self._length

    def build_pose(self, pose_times, poses, rgb_times):
        return torch.tensor(np.array(interpolate_poses(pose_times.tolist(), list(poses), rgb_times.tolist(), rgb_times[0])), dtype=torch.float32)

    def build_image_index(self, rgb_times, pose_times, depth_times):
        curr_pose_i = 0
        curr_depth_i = 0
        image_index = np.zeros((rgb_times.shape[0], 2), dtype=np.int)
        for i, timestamp in enumerate(rgb_times):
            while (curr_pose_i + 1 < pose_times.shape[0]) and abs(timestamp - pose_times[curr_pose_i]) > abs(timestamp - pose_times[curr_pose_i + 1]):
                curr_pose_i += 1
            while (curr_depth_i + 1 < depth_times.shape[0]) and abs(timestamp - depth_times[curr_depth_i]) > abs(timestamp - depth_times[curr_depth_i + 1]):
                curr_depth_i += 1
            image_index[i, 0] = curr_pose_i
            image_index[i, 1] = curr_depth_i
        return image_index

    def load_file_times(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
            lines = lines[3:]
        pairs = [l.split(" ") for l in lines]
        times = np.array([float(p[0]) for p in pairs])
        paths = [p[1][:-1] for p in pairs]
        return times, paths

    def load_pose_times(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
            lines = lines[3:]
        data = np.genfromtxt(lines, dtype=np.float64)
        times = data[:, 0]
        ts = torch.tensor(data[:, 1:4])
        qs = torch.tensor(data[:, [7, 4, 5, 6]])
        rs = torch.eye(4).unsqueeze(0).repeat(qs.shape[0], 1, 1)
        rs[:, :3, :3] = torch.tensor(Rotation.from_quat(qs).as_matrix())
        rs[:, :3, 3] = ts
        poses = rs.to(torch.float32)
        poses[:, :3, 3] = ts
        poses[:, 3, 3] = 1
        return times, poses

    def open_image(self, index):
        i = torch.tensor(np.asarray(Image.open(self.dataset_dir / self._rgb_paths[index])), dtype=torch.float32)
        i = i / 255 - .5
        i = i.permute(2, 0, 1)
        return i

    def open_depth(self, index):
        d = torch.tensor(np.asarray(Image.open(self.dataset_dir / self._depth_paths[self._image_index[index, 1]])), dtype=torch.float32)
        invalid = d == 0
        d = 1 / (d * self._depth_scale)
        d[invalid] = 0
        return d.unsqueeze(0)