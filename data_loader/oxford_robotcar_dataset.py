from functools import partial
from pathlib import Path

import numpy as np
import numpy.linalg as linalg
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from utils import map_fn

import os
if os.path.exists(os.path.abspath(os.path.join(__file__, os.pardir, 'oxford_robotcar'))):
    from .oxford_robotcar.camera_model import CameraModel
    from .oxford_robotcar.image import load_image
    from .oxford_robotcar.interpolate_poses import interpolate_vo_poses, build_se3_transform

swapaxes = np.asarray([[0, 0, 1, 0],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])

swapaxes_ = np.linalg.inv(swapaxes)


class OxfordRobotCarDataset(Dataset):
    def __init__(self, sequence_folders, pose_files, lidar_folders, model_folder, extrinsics_folder, frame_count=2, dilation=1, scale=0.25, cutout=(1/6, 1/6, 0, 0), lidar_timestamp_range=.5):
        """
        Dataset implementation for Oxford RobotCar
        :param sequence_folders: Folders containing images (Usually [.../sample/stereo/center])
        :param pose_files: Files containing images (Usually [.../sample/vo/vo.csv])
        :param lidar_folders: Files containing images (Usually [.../sample/ldmrs])
        :param model_folder: Folder containing camera models (Usually .../models)
        :param extrinsics_folder: Folder containing extrinsics (Usually .../extrinsics)
        :param frame_count: Number of frames used per sample (excluding keyframe). (Default=2)
        :param dilation: Spacing between the different frames. (Default=1)
        :param scale: Scaling of the input frames. (Default=0.25)
        :param cutout: Crop frames to given box. Specified in percent. (top, bottom, left, right). (Default=(1/6, 1/6, 0, 0)
        :param lidar_timestamp_range: Timestamp range in seconds over which lidar points are accumulated, because a single measurement is too sparse. (Default=.5)
        """
        super().__init__()
        self.sequence_folders = [Path(image_folder) for image_folder in sequence_folders]
        self.pose_files = [Path(pose_file) for pose_file in pose_files]
        self.lidar_folders = [Path(lidar_folder) for lidar_folder in lidar_folders]
        self.model_folder = Path(model_folder)
        self.extrinsics_folder = Path(extrinsics_folder)
        self.frame_count = frame_count
        self.dilation = dilation
        self.scale = scale
        self.cutout = cutout
        self.lidar_timestamp_range = lidar_timestamp_range
        # hack
        self.target_image_size = (320, 640)

        self._extra_frames = self.frame_count
        self._offset = (self._extra_frames // 2) * self.dilation
        self._sequence_files = [sorted(list(sequence_folder.glob("[0-9]*.png"))) for sequence_folder in self.sequence_folders]
        self._sequence_timestamps = [[int(file.stem) for file in sequence_files] for sequence_files in self._sequence_files]
        self._sequence_poses = [interpolate_vo_poses(pose_file, timestamps, min(timestamps)) for pose_file, timestamps in zip(self.pose_files, self._sequence_timestamps)]
        self._sequence_poses = [[pose @ swapaxes for pose in sequence_pose] for sequence_pose in self._sequence_poses]
        self._sequence_lengths = [len(files) - self._extra_frames for files in self._sequence_files]
        self._sequence_camera_models = [CameraModel(self.model_folder, str(sequence_folder)) for sequence_folder in self.sequence_folders]
        self._sequence_intrinsics = self.build_intrinsics_mats()
        self._lidar_files = [sorted(list(lidar_folder.glob("[0-9]*.bin"))) for lidar_folder in self.lidar_folders]
        self._lidar_timestamps = [[int(file.stem) for file in lidar_files] for lidar_files in self._lidar_files]
        self._lidar_poses = [interpolate_vo_poses(pose_file, list(timestamps), seq_timestamps[0]) for pose_file, timestamps, seq_timestamps in zip(self.pose_files, self._lidar_timestamps, self._sequence_timestamps)]
        self._lidar_transform = self.build_lidar_transforms()
        self._camera_transform = self.build_camera_transforms()
        self._length = sum(self._sequence_lengths)

    def __getitem__(self, index):
        sequence_index, index = self.get_dataset_index(index)

        data_dict = {}
        data_dict["keyframe"], data_dict["keyframe_pose"], data_dict["keyframe_intrinsics"] = self.get_timestamp(sequence_index, index + self._offset)
        data_dict["frames"] = []
        data_dict["poses"] = []
        data_dict["intrinsics"] = []

        for i in range(-self.frame_count // 2, (self.frame_count+1) // 2 + 1, 1):
            if i == 0: continue
            frame, pose, intrinsics = self.get_timestamp(sequence_index, index + self._offset + i * self.dilation)
            data_dict["frames"].append(frame)
            data_dict["poses"].append(pose)
            data_dict["intrinsics"].append(intrinsics)

        depth = self.get_depth(sequence_index, index + self._offset, data_dict["keyframe"].shape[1:])

        data_dict = map_fn(data_dict, partial(torch.tensor, dtype=torch.float32))

        data_dict["sequence"] = torch.tensor(sequence_index, dtype=torch.int32)
        data_dict["image_id"] = torch.tensor(index + self._offset, dtype=torch.int32)

        return data_dict, torch.tensor(depth, dtype=torch.float32)

    def __len__(self):
        return self._length

    def get_dataset_index(self, index):
        sequence_index = 0
        for sequence_length in self._sequence_lengths:
            if index < sequence_length:
                break
            else:
                sequence_index += 1
                index -= sequence_length
        return sequence_index, index

    def get_timestamp(self, sequence_index, index):
        frame = load_image(self._sequence_files[sequence_index][index], self._sequence_camera_models[sequence_index]) / 256 - .5
        frame = resize(frame, output_shape=(frame.shape[0] * self.scale, frame.shape[1] * self.scale))
        frame = np.rollaxis(frame, 2, 0)
        shape = tuple(frame.shape)
        frame = frame[:, int(self.cutout[0] * frame.shape[1]):int(frame.shape[1] - self.cutout[1] * frame.shape[1]),  int(self.cutout[2] * frame.shape[2]):int(frame.shape[2] - self.cutout[3] * frame.shape[2])]
        pose = self._sequence_poses[sequence_index][index]
        intrinsics = self._sequence_intrinsics[sequence_index]
        intrinsics[0, 2] -= self.cutout[2] * shape[2]
        intrinsics[1, 2] -= self.cutout[0] * shape[1]
        return frame, pose, intrinsics

    def get_depth(self, sequence_index, index, shape):
        image_timestamp = self._sequence_timestamps[sequence_index][index]
        lidar_indices = [i for i, timestamp in enumerate(self._lidar_timestamps[sequence_index]) if image_timestamp - self.lidar_timestamp_range * 1e6 <= timestamp <= image_timestamp + self.lidar_timestamp_range * 1e6]
        pointcloud = np.array([[0], [0], [0], [0]])
        for i in lidar_indices:
            with open(self.lidar_folders[sequence_index] / f"{self._lidar_timestamps[sequence_index][i]}.bin") as f:
                scan = np.fromfile(f, dtype=np.double)
            scan = scan.reshape((len(scan) // 3, 3)).transpose()
            scan = self._lidar_poses[sequence_index][i] @ self._lidar_transform[sequence_index] @ np.vstack([scan, np.ones((1, scan.shape[1]))])
            pointcloud = np.hstack([pointcloud, scan])
        pointcloud = self._camera_transform[sequence_index] @ (np.linalg.inv(self._sequence_poses[sequence_index][index] @ swapaxes_)) @ pointcloud
        uv, d = self._sequence_camera_models[sequence_index].project(pointcloud, (shape[0] / self.scale / (1 - self.cutout[0] - self.cutout[1]), shape[1] / self.scale / (1 - self.cutout[2] - self.cutout[3])))
        uv *= self.scale
        uv = uv.astype(np.int)
        d = 1 / d

        sort_indices = np.argsort(d)
        uv = uv[:, sort_indices]
        d = d[sort_indices]

        depth = np.zeros((round(shape[0] / (1 - self.cutout[0] - self.cutout[1])), round(shape[1] / (1 - self.cutout[2] - self.cutout[3]))))
        depth[uv[1, :], uv[0, :]] = d
        depth = depth[int(self.cutout[0] * depth.shape[0]):int(depth.shape[0] - self.cutout[1] * depth.shape[0]),  int(self.cutout[2] * depth.shape[1]):int(depth.shape[1] - self.cutout[3] * depth.shape[1])]
        return np.expand_dims(depth, 0)

    def build_intrinsics_mats(self):
        intrinsics = []
        for model in self._sequence_camera_models:
            mat = np.identity(4)
            mat[0, 0] = model.focal_length[0] * self.scale
            mat[1, 1] = model.focal_length[1] * self.scale
            mat[0, 2] = model.principal_point[0] * self.scale
            mat[1, 2] = model.principal_point[1] * self.scale
            intrinsics.append(mat)
        return intrinsics

    def build_lidar_transforms(self):
        lidar_transforms = []
        for model in self._sequence_camera_models:
            extrinsics_path = self.extrinsics_folder / 'ldmrs.txt'
            with open(extrinsics_path) as extrinsics_file:
                extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
            lidar_transforms.append(build_se3_transform(extrinsics))
        return lidar_transforms

    def build_camera_transforms(self):
        camera_transforms = []
        for model in self._sequence_camera_models:
            extrinsics_path = self.extrinsics_folder / (model.camera + '.txt')
            with open(extrinsics_path) as extrinsics_file:
                extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
            camera_transforms.append(build_se3_transform(extrinsics))
        return camera_transforms