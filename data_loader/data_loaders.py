from base import BaseDataLoader

from .kitti_odometry_dataset import *
from .oxford_robotcar_dataset import OxfordRobotCarDataset
from .tum_mono_vo_dataset import *
from .tum_rgbd_dataset import *


class KittiOdometryDataloader(BaseDataLoader):

    def __init__(self, batch_size=1, shuffle=True, validation_split=0.0, num_workers=4, **kwargs):
        self.dataset = KittiOdometryDataset(**kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class OxfordRobotCarDataloader(BaseDataLoader):

    def __init__(self, batch_size=1, shuffle=False, validation_split=0, num_workers=4, **kwargs):

        args = {
            "sequence_folders": ["../data/oxford_robotcar/sample/stereo/centre"],
            "pose_files": ["../data/oxford_robotcar/sample/vo/vo.csv"],
            "lidar_folders": ["../data/oxford_robotcar/sample/ldmrs"],
            "model_folder": "../data/oxford_robotcar/models",
            "extrinsics_folder": "../data/oxford_robotcar/extrinsics",
            "frame_count": 2,
            "cutout": [0, 1 / 3, 0, 0],
            "scale": .5,
            "lidar_timestamp_range": .25
        }

        args.update(kwargs)

        self.dataset = OxfordRobotCarDataset(**args)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class TUMMonoVODataloader(BaseDataLoader):

    def __init__(self, batch_size=1, shuffle=True, validation_split=0.0, num_workers=4, **kwargs):
        self.dataset = TUMMonoVOMultiDataset(**kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class TUMRGBDDataloader(BaseDataLoader):

    def __init__(self, batch_size=1, shuffle=True, validation_split=0.0, num_workers=4, **kwargs):
        self.dataset = TUMRGBDDataset(**kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
