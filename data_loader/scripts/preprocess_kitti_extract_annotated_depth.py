import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

mapping = {
    "2011_10_03_drive_0027": "00",
    "2011_10_03_drive_0042": "01",
    "2011_10_03_drive_0034": "02",
    "2011_09_26_drive_0067": "03",
    "2011_09_30_drive_0016": "04",
    "2011_09_30_drive_0018": "05",
    "2011_09_30_drive_0020": "06",
    "2011_09_30_drive_0027": "07",
    "2011_09_30_drive_0028": "08",
    "2011_09_30_drive_0033": "09",
    "2011_09_30_drive_0034": "10"
}

def main():
    parser = argparse.ArgumentParser(description='''
            This script creates depth images from annotated velodyne data.
            ''')
    parser.add_argument("--output", "-o", help="Path of KITTI odometry dataset", default="../../../data/dataset")
    parser.add_argument("--input", "-i", help="Path to KITTI depth dataset (zipped)", required=True)
    parser.add_argument("--depth_folder", "-d", help="Name of depth map folders for the respective sequences", default="image_depth_annotated")

    args = parser.parse_args()
    input = Path(args.input)
    output = Path(args.output)
    depth_folder = args.depth_folder

    drives = mapping.keys()

    print("Creating folder structure")
    for drive in drives:
        sequence = mapping[drive]
        folder = output/ "sequences" / sequence / depth_folder
        folder.mkdir(parents=True, exist_ok=True)
        print(folder)

    print("Extracting enhanced depth maps")

    with ZipFile(input) as depth_archive:
        for name in depth_archive.namelist():
            if name[0] == "t":
                drive = name[6:27]
            else:
                drive = name[4:25]
            cam = name[-16]
            img = name[-10:]
            if cam == '2' and drive in drives:
                to = output / "sequences" / mapping[drive] / depth_folder / img
                print(name, " -> ", to)
                with depth_archive.open(name) as i, open(to, 'wb') as o:
                    shutil.copyfileobj(i, o)

main()