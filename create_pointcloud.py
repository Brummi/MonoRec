import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.model as module_arch
from utils.parse_config import ConfigParser
from utils import to, PLYSaver, DS_Wrapper

import torch.nn.functional as F


def main(config):
    logger = config.get_logger('test')

    output_dir = Path(config.config.get("output_dir", "saved"))
    output_dir.mkdir(exist_ok=True, parents=True)
    file_name = config.config.get("file_name", "pc.ply")
    use_mask = config.config.get("use_mask", True)
    roi = config.config.get("roi", None)

    max_d = config.config.get("max_d", 30)
    min_d = config.config.get("min_d", 3)

    start = config.config.get("start", 0)
    end = config.config.get("end", -1)

    # setup data_loader instances
    data_loader = DataLoader(DS_Wrapper(config.initialize('data_set', module_data), start=start, end=end), batch_size=1, shuffle=False, num_workers=8)

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    mask_fill = 32

    n = data_loader.batch_size

    target_image_size = data_loader.dataset.dataset.target_image_size

    plysaver = PLYSaver(target_image_size[0], target_image_size[1], min_d=min_d, max_d=max_d, batch_size=n, roi=roi, dropout=.75)
    plysaver.to(device)

    pose_buffer = []
    intrinsics_buffer = []
    mask_buffer = []
    keyframe_buffer = []
    depth_buffer = []

    buffer_length = 5
    min_hits = 1
    key_index = buffer_length // 2

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data = to(data, device)
            # if not torch.any(pose_distance_thresh(data, spatial_thresh=1)):
            #     continue
            result = model(data)
            if not isinstance(result, dict):
                result = {"result": result[0]}
            output = result["result"]
            if "cv_mask" not in result:
                result["cv_mask"] = output.new_zeros(output.shape)
            # mask = ((result["cv_mask"] >= .1) & (output >= 1 / max_d)).to(dtype=torch.float32)
            mask = (result["cv_mask"] >= .1).to(dtype=torch.float32)
            mask = (F.conv2d(mask, mask.new_ones((1, 1, mask_fill+1, mask_fill+1)), padding=mask_fill // 2) < 1).to(dtype=torch.float32)

            pose_buffer += data["keyframe_pose"]
            intrinsics_buffer += [data["keyframe_intrinsics"]]
            mask_buffer += [mask]
            keyframe_buffer += [data["keyframe"]]
            depth_buffer += [output]

            if len(pose_buffer) >= buffer_length:
                pose = pose_buffer[key_index]
                intrinsics = intrinsics_buffer[key_index]
                keyframe = keyframe_buffer[key_index]
                depth = depth_buffer[key_index]

                mask = (torch.sum(torch.stack(mask_buffer), dim=0) > buffer_length - min_hits).to(dtype=torch.float32)
                if use_mask:
                    depth *= mask

                plysaver.add_depthmap(depth, keyframe, intrinsics, pose)

                del pose_buffer[0]
                del intrinsics_buffer[0]
                del mask_buffer[0]
                del keyframe_buffer[0]
                del depth_buffer[0]

        with open(output_dir / file_name, "wb") as f:
            plysaver.save(f)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
