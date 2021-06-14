from __future__ import absolute_import, division, print_function

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, Conv2d, LeakyReLU, Upsample, Sigmoid, ConvTranspose2d


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Backprojection(nn.Module):
    def __init__(self, batch_size, height, width):
        super(Backprojection, self).__init__()

        self.N, self.H, self.W = batch_size, height, width

        yy, xx = torch.meshgrid([torch.arange(0., float(self.H)), torch.arange(0., float(self.W))])
        yy = yy.contiguous().view(-1)
        xx = xx.contiguous().view(-1)
        self.ones = nn.Parameter(torch.ones(self.N, 1, self.H * self.W), requires_grad=False)
        self.coord = torch.unsqueeze(torch.stack([xx, yy], 0), 0).repeat(self.N, 1, 1)
        self.coord = nn.Parameter(torch.cat([self.coord, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K) :
        cam_p_norm = torch.matmul(inv_K[:, :3, :3], self.coord[:depth.shape[0], :, :])
        cam_p_euc = depth.view(depth.shape[0], 1, -1) * cam_p_norm
        cam_p_h = torch.cat([cam_p_euc, self.ones[:depth.shape[0], :, :]], 1)

        return cam_p_h

def point_projection(points3D, batch_size, height, width, K, T):
    N, H, W = batch_size, height, width
    cam_coord = torch.matmul(torch.matmul(K, T)[:, :3, :], points3D)
    img_coord = cam_coord[:, :2, :] / (cam_coord[:, 2:3, :] + 1e-7)
    img_coord[:, 0, :] /= W - 1
    img_coord[:, 1, :] /= H - 1
    img_coord = (img_coord - 0.5) * 2
    img_coord = img_coord.view(N, 2, H, W).permute(0, 2, 3, 1)
    return img_coord

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class GaussianAverage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.window = torch.Tensor([
            [0.0947, 0.1183, 0.0947],
            [0.1183, 0.1478, 0.1183],
            [0.0947, 0.1183, 0.0947]])

    def forward(self, x):
        kernel = self.window.to(x.device).to(x.dtype).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=0, groups=x.shape[1])

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, pad_reflection=True, gaussian_average=False, comp_mode=False):
        super(SSIM, self).__init__()
        self.comp_mode = comp_mode

        if not gaussian_average:
            self.mu_x_pool   = nn.AvgPool2d(3, 1)
            self.mu_y_pool   = nn.AvgPool2d(3, 1)
            self.sig_x_pool  = nn.AvgPool2d(3, 1)
            self.sig_y_pool  = nn.AvgPool2d(3, 1)
            self.sig_xy_pool = nn.AvgPool2d(3, 1)
        else:
            self.mu_x_pool = GaussianAverage()
            self.mu_y_pool = GaussianAverage()
            self.sig_x_pool = GaussianAverage()
            self.sig_y_pool = GaussianAverage()
            self.sig_xy_pool = GaussianAverage()

        if pad_reflection:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.pad(x)
        y = self.pad(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_x_y = mu_x * mu_y

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x_sq
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y_sq
        sigma_xy = self.sig_xy_pool(x * y) - mu_x_y

        SSIM_n = (2 * mu_x_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)

        if not self.comp_mode:
            return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        else:
            return torch.clamp((1 - SSIM_n / SSIM_d), 0, 1) / 2


def ssim(x, y, pad_reflection=True, gaussian_average=False, comp_mode=False):
    ssim_ = SSIM(pad_reflection, gaussian_average, comp_mode)
    return ssim_(x, y)


class ResidualImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_image = ResidualImageModule()

    def forward(self, keyframe: torch.Tensor, keyframe_pose: torch.Tensor, keyframe_intrinsics: torch.Tensor,
                depths: torch.Tensor, frames: list, poses: list, intrinsics: list):
        data_dict = {"keyframe": keyframe, "keyframe_pose": keyframe_pose, "keyframe_intrinsics": keyframe_intrinsics,
                     "predicted_inverse_depths": [depths], "frames": frames, "poses": poses, "list": list,
                     "intrinsics": intrinsics, "inv_depth_max": 0, "inv_depth_min": 1}
        data_dict = self.residual_image(data_dict)
        return data_dict["residual_image"]


class ResidualImageModule(nn.Module):
    def __init__(self, use_mono=True, use_stereo=False):
        super().__init__()
        self.use_mono = use_mono
        self.use_stereo = use_stereo
        self.ssim = SSIM()

    def forward(self, data_dict):
        keyframe = data_dict["keyframe"]
        keyframe_intrinsics = data_dict["keyframe_intrinsics"]
        keyframe_pose = data_dict["keyframe_pose"]
        depths = (1-data_dict["predicted_inverse_depths"][0]) * data_dict["inv_depth_max"] + data_dict["predicted_inverse_depths"][0] * data_dict["inv_depth_min"]

        frames = []
        intrinsics = []
        poses = []

        if self.use_mono:
            frames += data_dict["frames"]
            intrinsics += data_dict["intrinsics"]
            poses += data_dict["poses"]
        if self.use_stereo:
            frames += [data_dict["stereoframe"]]
            intrinsics += [data_dict["stereoframe_intrinsics"]]
            poses += [data_dict["stereoframe_pose"]]

        n, c, h, w = keyframe.shape

        backproject_depth = Backprojection(n, h, w)
        backproject_depth.to(keyframe.device)

        inv_k = torch.inverse(keyframe_intrinsics)
        cam_points = (inv_k[:, :3, :3] @ backproject_depth.pix_coords)
        cam_points = cam_points / depths.view(n, 1, -1)
        cam_points = torch.cat([cam_points, backproject_depth.ones], 1)

        masks = []
        residuals = []

        for i, image in enumerate(frames):
            t = torch.inverse(poses[i]) @ keyframe_pose
            pix_coords = point_projection(cam_points, n, h, w, intrinsics[i], t)
            warped_image = F.grid_sample(image + 1, pix_coords)
            mask = torch.any(warped_image == 0, dim=1, keepdim=True)
            warped_image -= .5
            residual = self.ssim(warped_image, keyframe + .5)
            masks.append(mask)
            residuals.append(residual)

        masks = torch.stack(masks, dim=1)
        residuals = torch.stack(residuals, dim=1)
        residuals[masks.expand(-1, -1 , c, -1, -1)] = float("inf")

        residual_image = torch.min(torch.mean(residuals, dim=2, keepdim=True), dim=1)[0]
        residual_image[torch.min(masks, dim=1)[0]] = 0
        data_dict["residual_image"] = residual_image
        return data_dict


class PadSameConv2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        """
        Imitates padding_mode="same" from tensorflow.
        :param kernel_size: Kernelsize of the convolution, int or tuple/list
        :param stride: Stride of the convolution, int or tuple/list
        """
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size_y = kernel_size[0]
            self.kernel_size_x = kernel_size[1]
        else:
            self.kernel_size_y = kernel_size
            self.kernel_size_x = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor):
        _, _, height, width = x.shape

        # For the convolution we want to achieve a output size of (n_h, n_w) = (math.ceil(h / s_y), math.ceil(w / s_y)).
        # Therefore we need to apply n_h convolution kernels with stride s_y. We will have n_h - 1 offsets of size s_y.
        # Additionally, we need to add the size of our kernel. This is the height we require to get n_h. We need to pad
        # the read difference between this and the old height. We will pad math.floor(pad_y / 2) on the left and
        # math-ceil(pad_y / 2) on the right. Same  for pad_x respectively.
        padding_y = (self.stride_y * (math.ceil(height / self.stride_y) - 1) + self.kernel_size_y - height) / 2
        padding_x = (self.stride_x * (math.ceil(width / self.stride_x) - 1) + self.kernel_size_x - width) / 2
        padding = [math.floor(padding_x), math.ceil(padding_x), math.floor(padding_y), math.ceil(padding_y)]
        return F.pad(input=x, pad=padding)


class PadSameConv2dTransposed(torch.nn.Module):
    def __init__(self, stride):
        """
        Imitates padding_mode="same" from tensorflow.
        :param stride: Stride of the convolution_transposed, int or tuple/list
        """
        super().__init__()
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor, orig_shape: torch.Tensor):
        target_shape = x.new_tensor(list(orig_shape))
        target_shape[-2] *= self.stride_y
        target_shape[-1] *= self.stride_x
        oversize = target_shape[-2:] - x.new_tensor(x.shape)[-2:]
        if oversize[0] > 0 and oversize[1] > 0:
            x = F.pad(x, [math.floor(oversize[1] / 2), math.ceil(oversize[1] / 2), math.floor(oversize[0] / 2),
                          math.ceil(oversize[0] / 2)])
        elif oversize[0] > 0 >= oversize[1]:
            x = F.pad(x, [0, 0, math.floor(oversize[0] / 2), math.ceil(oversize[0] / 2)])
            x = x[:, :, :, math.floor(-oversize[1] / 2):-math.ceil(-oversize[1] / 2)]
        elif oversize[0] <= 0 < oversize[1]:
            x = F.pad(x, [math.floor(oversize[1] / 2), math.ceil(oversize[1] / 2)])
            x = x[:, :, math.floor(-oversize[0] / 2):-math.ceil(-oversize[0] / 2), :]
        else:
            x = x[:, :, math.floor(-oversize[0] / 2):-math.ceil(-oversize[0] / 2),
                math.floor(-oversize[1] / 2):-math.ceil(-oversize[1] / 2)]
        return x


class ConvReLU2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky_relu_neg_slope=0.1):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.pad_0 = PadSameConv2d(kernel_size=(kernel_size, 1), stride=(stride, 1))
        self.conv_y = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                             stride=(stride, 1))
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)
        self.pad_1 = PadSameConv2d(kernel_size=(1, kernel_size), stride=(1, stride))
        self.conv_x = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                             stride=(1, stride))

    def forward(self, x: torch.Tensor):
        t = self.pad_0(x)
        t = self.conv_y(t)
        t = self.leaky_relu(t)
        t = self.pad_1(t)
        t = self.conv_x(t)
        return self.leaky_relu(t)


class ConvReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky_relu_neg_slope=0.1):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.leaky_relu(t)


class Upconv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.upsample = Upsample(scale_factor=2)
        self.pad = PadSameConv2d(kernel_size=2)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=1)

    def forward(self, x: torch.Tensor):
        t = self.upsample(x)
        t = self.pad(t)
        return self.conv(t)


class ConvSig(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.sig = Sigmoid()

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.sig(t)


class Refine(torch.nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_neg_slope=0.1):
        """
        Performs a transposed conv2d with padding that imitates tensorflow same behaviour. The transposed conv2d has
        parameters kernel_size=4 and stride=2.
        :param in_channels: Channels that go into the conv2d_transposed
        :param out_channels: Channels that come out of the conv2d_transposed
        """
        super().__init__()
        self.conv2d_t = ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2)
        self.pad = PadSameConv2dTransposed(stride=2)
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor, features_direct=None):
        orig_shape=x.shape
        x = self.conv2d_t(x)
        x = self.leaky_relu(x)
        x = self.pad(x, orig_shape)
        if features_direct is not None:
            x = torch.cat([x, features_direct], dim=1)
        return x
