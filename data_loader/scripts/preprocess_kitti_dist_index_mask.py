import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data_loader.kitti_odometry_dataset import KittiOdometryDataset
from utils import pose_distance_thresh, unsqueezer, map_fn

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description='''
            This script creates an index mask that filters out images that are too close to the previous image. (Never used in the paper)
            ''')
    parser.add_argument("--dataset", "-d",
                        help="Path of KITTI dataset",
                        default="../../../data/dataset")
    parser.add_argument("--output", "-o",
                        help="Name of directory in the respective sequence folder which images are written to",
                        default=".")
    parser.add_argument("--sequences", "-s",
                        help="Only perform preprocessing for these sequences",
                        nargs="+", default=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"])
    parser.add_argument("--threshold", "-t",
                        help="Use this theshold for filtering",
                        default=0.8)


    args = parser.parse_args()
    base_folder = Path(args.dataset)
    output = Path(args.output)
    sequences = args.sequences
    threshold = args.threshold

    print("Generating moving object index masks")
    for sequence in sequences:
        print(f"Processing sequence {sequence}...")

        dataset = KittiOdometryDataset(base_folder, sequences=[sequence], use_dso_poses=True, lidar_depth=True, depth_folder="image_depth_annotated")

        enough_dist = {}

        for i, (data_dict, _) in tqdm(enumerate(dataset)):
            enough_dist[i + dataset._offset] = pose_distance_thresh(map_fn(data_dict, unsqueezer), spatial_thresh=threshold).item()

        with open(base_folder / "sequences" / sequence / output / "index_mask_dist.json", "w") as f:
            json.dump(enough_dist, f)

main()