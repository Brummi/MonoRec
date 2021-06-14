from array import array

import torch

from model.layers import Backprojection


class PLYSaver(torch.nn.Module):
    def __init__(self, height, width, min_d=3, max_d=400, batch_size=1, roi=None, dropout=0):
        super(PLYSaver, self).__init__()
        self.min_d = min_d
        self.max_d = max_d
        self.roi = roi
        self.dropout = dropout
        self.data = array('f')

        self.projector = Backprojection(batch_size, height, width)

    def save(self, file):
        length = len(self.data) // 6
        header = "ply\n" \
                 "format binary_little_endian 1.0\n" \
                 f"element vertex {length}\n" \
                 f"property float x\n" \
                 f"property float y\n" \
                 f"property float z\n" \
                 f"property float red\n" \
                 f"property float green\n" \
                 f"property float blue\n" \
                 f"end_header\n"
        file.write(header.encode(encoding="ascii"))
        self.data.tofile(file)

    def add_depthmap(self, depth: torch.Tensor, image: torch.Tensor, intrinsics: torch.Tensor,
                     extrinsics: torch.Tensor):
        depth = 1 / depth
        image = (image + .5) * 255
        mask = (self.min_d <= depth) & (depth <= self.max_d)
        if self.roi is not None:
            mask[:, :, :self.roi[0], :] = False
            mask[:, :, self.roi[1]:, :] = False
            mask[:, :, :, self.roi[2]] = False
            mask[:, :, :, self.roi[3]:] = False
        if self.dropout > 0:
            mask = mask & (torch.rand_like(depth) > self.dropout)

        coords = self.projector(depth, torch.inverse(intrinsics))
        coords = extrinsics @ coords
        coords = coords[:, :3, :]
        data_batch = torch.cat([coords, image.view_as(coords)], dim=1).permute(0, 2, 1)
        data_batch = data_batch[mask.view(depth.shape[0], 1, -1).permute(0, 2, 1).expand(-1, -1, 6)]

        self.data.extend(data_batch.cpu().tolist())
