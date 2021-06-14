from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

from utils import map_fn


class TUMMonoVOMultiDataset(Dataset):

    def __init__(self, dataset_dirs, **kwargs):
        if isinstance(dataset_dirs, list):
            self.datasets = [TUMMonoVODataset(dataset_dir, **kwargs) for dataset_dir in dataset_dirs]
        else:
            self.datasets = [TUMMonoVODataset(dataset_dirs, **kwargs)]

    def __getitem__(self, index):
        for dataset in self.datasets:
            l = len(dataset)
            if index >= l:
                index -= l
            else:
                return dataset.__getitem__(index)
        return None

    def __len__(self):
        sum = 0
        for dataset in self.datasets:
            sum += len(dataset)
        return sum


class TUMMonoVODataset(Dataset):

    def __init__(self, dataset_dir, frame_count=2, target_image_size=(480, 640), max_length=None, dilation=1, only_keyframes=False, color_augmentation=True, scale_factor=1):
        """
        Dataset implementation for TUMMonoVO. Requires the images to be rectified first. Support for depth maps is WIP.
        :param dataset_dir: Folder of a single sequence (e.g. .../tummonovo/sequence_50). This folder should contain images/.
        :param frame_count: Number of frames used per sample (excluding the keyframe). (Default=2)
        :param target_image_size: Desired image size. (Default=(480, 640))
        :param max_length: Crop dataset to given length. (Default=None)
        :param dilation: Spacing between the different frames. (Default=1)
        :param only_keyframes: Only use frames that were used as keyframes in DSO. Relies on depth maps -> WIP. (Default=False)
        :param color_augmentation: Use color jitter augmentation. (Default=False)
        :param scale_factor: Scale poses for the sequence. Useful for DSO, which does not necessarily detect the correct world-scale. (Default=1)
        """
        self.dataset_dir = Path(dataset_dir)
        self.frame_count = frame_count
        self.only_keyframes = only_keyframes
        self.dilation = dilation
        self.target_image_size = target_image_size
        self.color_augmentation = color_augmentation
        self.scale_factor = scale_factor

        self._result = np.loadtxt(self.dataset_dir / "result.txt")
        self._times = np.loadtxt(self.dataset_dir / "times.txt")
        self._pcalib = self.invert_pcalib(np.loadtxt(self.dataset_dir / "pcalib.txt"))
        self._image_index = self.build_image_index()

        if self.only_keyframes:
            self._keyframe_index = self.build_keyframe_index()
            self.length = self._keyframe_index.shape[0]
        else:
            self.length = self._result.shape[0] - frame_count * dilation
            if max_length is not None:
                self.length = min(self.length, max_length)

        self._offset = (frame_count // 2) * self.dilation

        self._intrinsics, self._crop_box = self.compute_target_intrinsics()
        self._intrinsics = format_intrinsics(self._intrinsics, self.target_image_size)

        self._poses = self.build_poses()
        self._depth = torch.zeros((1, target_image_size[0], target_image_size[1]), dtype=torch.float32)

        if self.color_augmentation:
            self.color_augmentation_transform = ColorJitterMulti(brightness=.2, contrast=.2, saturation=.2, hue=.1)

    def preprocess_image(self, img: Image.Image, crop_box=None):
        img = img.convert('RGB')
        if crop_box:
            img = img.crop(crop_box)
        if self.target_image_size:
            img = img.resize((self.target_image_size[1], self.target_image_size[0]), resample=Image.BILINEAR)
        if self.color_augmentation:
            img = self.color_augmentation_transform(img)
        image_tensor = torch.tensor(np.array(img)).to(dtype=torch.float32)
        image_tensor = self._pcalib[image_tensor.to(dtype=torch.long)]
        image_tensor = image_tensor / 255 - .5
        if len(image_tensor.shape) == 2:
            image_tensor = torch.stack((image_tensor, image_tensor, image_tensor))
        else:
            image_tensor = image_tensor.permute(2, 0, 1)
        del img
        return image_tensor

    def preprocess_depth(self, depth: Image.Image, crop_box=None):
        if crop_box:
            depth = depth.crop(crop_box)
        if self.target_image_size:
            if self.target_image_size[0] * 2 == depth.size[1]:
                depth_tensor = torch.tensor(np.array(depth).astype(np.float32))
                depth_tensor = torch.nn.functional.max_pool2d(depth_tensor.unsqueeze(0), kernel_size=2)
            else:
                depth = depth.resize((self.target_image_size[1], self.target_image_size[0]), resample=Image.BILINEAR)
                depth_tensor = torch.tensor(np.array(depth).astype(np.float32)).unsqueeze(0)
        depth_tensor[depth_tensor < 0] = 0
        return depth_tensor

    def __getitem__(self, index: int):
        frame_count = self.frame_count
        offset = self._offset

        if self.color_augmentation:
            self.color_augmentation_transform.fix_transform()

        if self.only_keyframes:
            index = self._keyframe_index[index] - offset

        keyframe_intrinsics = self._intrinsics
        keyframe = self.preprocess_image(self.open_image(index + offset), self._crop_box)
        keyframe_pose = self._poses[index + offset]
        keyframe_depth = self.open_depth(index + offset)
        if keyframe_depth is None:
            keyframe_depth = self._depth
        else:
            keyframe_depth = self.preprocess_depth(keyframe_depth, self._crop_box)

        frames = [self.preprocess_image(self.open_image(index + i), self._crop_box) for i in range(0, (frame_count + 1) * self.dilation, self.dilation) if i != offset]
        intrinsics = [self._intrinsics for _ in range(frame_count)]
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
        return self.length

    def build_image_index(self):
        eps = 1e-5
        current_index = 0
        image_index = np.zeros((self._result.shape[0]), dtype=np.int)
        for i in range(self._result.shape[0]):
            timestamp = self._result[i, 0]
            while not timestamp <= self._times[current_index, 1] + eps:
                current_index += 1
            image_index[i] = current_index
        return image_index

    def build_keyframe_index(self):
        keyframe_index = []
        image_index_pos = 0
        for p in sorted((self.dataset_dir / "images_depth").glob("*.exr")):
            index = int(p.stem[:5])
            while self._image_index[image_index_pos] < index:
                image_index_pos += 1
            index = image_index_pos
            if not (index >= len(self._image_index) - (self.frame_count // 2 + 1) * self.dilation or index < (self.frame_count // 2) * self.dilation):
                keyframe_index.append(index)
        return np.array(keyframe_index)

    def load_orig_intrinsics(self):
        camera_file = self.dataset_dir / "camera.txt"
        with open(camera_file) as f:
            intrinsics_use_first_col = ord("0") <= ord(f.readline()[0]) <= ord("9")
        if intrinsics_use_first_col:
            intrinsics_v = np.loadtxt(camera_file, usecols=list(range(4)), max_rows=1)
        else:
            intrinsics_v = np.loadtxt(camera_file, usecols=[1, 2, 3, 4], max_rows=1)
        intrinsics = np.identity(4, dtype=np.float)
        intrinsics[0, 0] = intrinsics_v[0]
        intrinsics[1, 1] = intrinsics_v[1]
        intrinsics[0, 2] = intrinsics_v[2]
        intrinsics[1, 2] = intrinsics_v[3]
        return intrinsics

    def compute_target_intrinsics(self):
        P_cam = self.load_orig_intrinsics()
        orig_size = tuple(reversed(Image.open(self.dataset_dir / "images" / "00000.jpg").size))

        P_cam[0, 0] *= orig_size[1]
        P_cam[1, 1] *= orig_size[0]
        P_cam[0, 2] *= orig_size[1]
        P_cam[1, 2] *= orig_size[0]

        r_orig = orig_size[0] / orig_size[1]
        r_target = self.target_image_size[0] / self.target_image_size[1]

        if r_orig >= r_target:
            new_height = r_target * orig_size[1]
            box = (0, (orig_size[0] - new_height) // 2, orig_size[1], orig_size[0] - (orig_size[0] - new_height) // 2)

            c_x = P_cam[0, 2] / orig_size[1]
            c_y = (P_cam[1, 2] - (orig_size[0] - new_height) / 2) / new_height

            rescale = orig_size[1] / self.target_image_size[1]

        else:
            new_width = orig_size[0] / r_target
            box = ((orig_size[1] - new_width) // 2, 0, orig_size[1] - (orig_size[1] - new_width) // 2, orig_size[0])

            c_x = (P_cam[0, 2] - (orig_size[1] - new_width) / 2) / new_width
            c_y = P_cam[1, 2] / orig_size[0]

            rescale = orig_size[0] / self.target_image_size[0]

        f_x = P_cam[0, 0] / self.target_image_size[1] / rescale
        f_y = P_cam[1, 1] / self.target_image_size[0] / rescale

        intrinsics = (f_x, f_y, c_x, c_y)

        return intrinsics, box

    def build_poses(self):
        ts = torch.tensor(self._result[:, 1:4])
        qs = torch.tensor(self._result[:, [7, 4, 5, 6]])
        rs = torch.eye(4).unsqueeze(0).repeat(qs.shape[0], 1, 1)
        rs[:, :3, :3] = torch.tensor(Rotation.from_quat(qs).as_matrix())
        rs[:, :3, 3] = ts * self.scale_factor
        poses = rs
        return poses.to(torch.float32)

    def open_image(self, index):
        return Image.open(self.dataset_dir / "images" / f"{self._image_index[index]:05d}.jpg")

    def open_depth(self, index):
        p = self.dataset_dir / "images_depth" / f"{self._image_index[index]:05d}_d.exr"
        if p.exists() and p.is_file():
            return Image.fromarray(cv2.imread(str(p), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        else:
            return None

    def invert_pcalib(self, pcalib):
        inv_pcalib = torch.zeros(256, dtype=torch.float32)
        j = 0
        for i in range(256):
            while j < 255 and i + .5 > pcalib[j]:
                j += 1
            inv_pcalib[i] = j
        return inv_pcalib


def format_intrinsics(intrinsics, target_image_size):
    intrinsics_mat = torch.zeros(4, 4, dtype=torch.float32)
    intrinsics_mat[0, 0] = intrinsics[0] * target_image_size[1]
    intrinsics_mat[1, 1] = intrinsics[1] * target_image_size[0]
    intrinsics_mat[0, 2] = intrinsics[2] * target_image_size[1]
    intrinsics_mat[1, 2] = intrinsics[3] * target_image_size[0]
    intrinsics_mat[2, 2] = 1
    intrinsics_mat[3, 3] = 1
    return intrinsics_mat


class ColorJitterMulti(torchvision.transforms.ColorJitter):
    def fix_transform(self):
        self.transform = self.get_params(self.brightness, self.contrast,
                                         self.saturation, self.hue)

    def __call__(self, x):
        return map_fn(x, self.transform)
