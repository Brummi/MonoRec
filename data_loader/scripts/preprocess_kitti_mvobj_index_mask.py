import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description='''
            This script creates index masks filtering out samples that don't contain moving objects.
            ''')
    parser.add_argument("--dataset", "-d",
                        help="Path of KITTI dataset",
                        default="../../../data/dataset")
    parser.add_argument("--mask_folder", "-m",
                        help="Name of directory in the respective sequence folder which images are written to",
                        default="mvobj_mask")
    parser.add_argument("--output", "-o",
                        help="Name of directory in the respective sequence folder which images are written to",
                        default=".")
    parser.add_argument("--sequences", "-s",
                        help="Only perform preprocessing for these sequences",
                        nargs="+", default=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"])


    args = parser.parse_args()
    base_folder = Path(args.dataset)
    mask_folder = Path(args.mask_folder)
    output = Path(args.output)
    sequences = args.sequences

    print("Creating folder structure")
    for sequence in sequences:
        folder = base_folder / "sequences" / sequence / mask_folder
        folder.mkdir(parents=True, exist_ok=True)
        print(folder)

    total_sequence = 0
    total_sequence_pts = 0

    print("Generating moving object index masks")
    for sequence in sequences:
        print(f"Processing sequence {sequence}...")
        folder = base_folder / "sequences" / sequence / mask_folder

        has_object = {}
        total = 0
        total_pts = 0

        for file in tqdm(sorted(folder.glob("*.npy"))):
            mask = np.load(file)
            pts = int(np.sum(mask.astype(dtype=np.int)))
            file_has_object = pts > 0
            has_object[int(file.stem)] = file_has_object
            total += file_has_object
            total_pts += pts

        total_sequence += total
        total_sequence_pts += total_pts

        print(f"{total / len(has_object):0.4f} ({total}/{len(has_object)}) masks contained moving objects")
        print(f"{total_pts} points found in total")

        with open(base_folder / "sequences" / sequence / output / "index_mask.json", "w") as f:
            json.dump(has_object, f)

    print(f"{total} masks contained moving objects (all sequences)")
    print(f"{total_pts} points found in total (all sequences).")

main()