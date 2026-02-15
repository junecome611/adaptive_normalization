"""
Method 4: EDAIN Min-Max (Pre-Mapped Learnable Normalization)

Stage 1 (fixed): z_pre = clip((x - p2) / (p98 - p2), 0, 1)
Stage 2 (learnable): z = (z_pre - m) / s
Global learnable parameters: m, s
"""

import os
import argparse
import glob
import json
import nibabel as nib
import numpy as np
import re
import csv
import time
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch import amp
from skimage.filters import threshold_otsu
from scipy.ndimage import affine_transform, gaussian_filter, zoom
from typing import Tuple, Optional

from monai.data import CacheDataset, DataLoader, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.utils import set_determinism
from monai.transforms import KeepLargestConnectedComponent, ClipIntensityPercentilesd
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    EnsureTyped, ToTensord, SaveImage, MapTransform, RandomizableTransform,
    SpatialCropd, CenterSpatialCropd
)
from monai.config import KeysCollection

import torch.nn as nn
from monai.data import MetaTensor



# ==============================================================================
# Robust Min-Max Normalization Transform
# ==============================================================================

class RobustMinMaxNormalized(MapTransform):
    """
    Robust Min-Max normalization using percentiles on nonzero voxels.
    Maps intensity to [0, 1] with clipping.
        z = clip((x - p2) / (p98 - p2), 0, 1)
    """

    def __init__(self, keys: KeysCollection, lower_pct: float = 2.0, upper_pct: float = 98.0):
        super().__init__(keys)
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
                was_tensor = True
                has_meta = hasattr(img, 'meta')
                if has_meta:
                    meta = img.meta
            else:
                img_np = np.asarray(img)
                was_tensor = False
                has_meta = False

            result = np.zeros_like(img_np, dtype=np.float32)

            for c in range(img_np.shape[0]):
                channel = img_np[c].astype(np.float32)
                nonzero_mask = channel != 0

                if nonzero_mask.sum() > 0:
                    nonzero_vals = channel[nonzero_mask]
                    p_low = np.percentile(nonzero_vals, self.lower_pct)
                    p_high = np.percentile(nonzero_vals, self.upper_pct)

                    if p_high - p_low > 1e-8:
                        normalized = (channel - p_low) / (p_high - p_low)
                    else:
                        normalized = np.zeros_like(channel)

                    result[c] = np.clip(normalized, 0.0, 1.0)

            if was_tensor:
                if has_meta:
                    from monai.data import MetaTensor
                    d[key] = MetaTensor(torch.from_numpy(result), meta=meta)
                else:
                    d[key] = torch.from_numpy(result)
            else:
                d[key] = result

        return d



# ==============================================================================
# Pre-Mapped Min-Max EDAIN Normalization (Method 4)
# ==============================================================================

class PreMappedEDAIN(nn.Module):
    """
    Learnable affine on pre-mapped [0,1] input: z = (z_pre - m) / s

    Two global learnable parameters:
        m: shift (init=0.0, constrained to [-0.2, 0.2])
        s: scale (init=1.0, constrained to [0.5, 2.0])

    When m=0, s=1: recovers standard robust min-max normalization.
    No fg_mean/fg_std needed since normalization is done in transform.
    """

    def __init__(self):
        super().__init__()
        self._m_raw = nn.Parameter(torch.tensor(0.0))   # tanh(0)=0 -> m=0.0
        self._s_raw = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5 -> s=1.0

    @property
    def m(self):
        return 0.2 * torch.tanh(self._m_raw)  # [-0.2, 0.2]

    @property
    def s(self):
        return 1.5 * torch.sigmoid(self._s_raw) + 0.5  # [0.5, 2.0]

    def forward(self, x, fg_mean=None, fg_std=None):
        """fg_mean, fg_std are ignored (kept for interface compatibility)."""
        return (x - self.m) / self.s

    def get_params_info(self):
        return {'m': self.m.item(), 's': self.s.item()}



class SegmentationModelWithEDAIN(nn.Module):
    """Wrapper: EDAIN normalization + segmentation backbone."""

    def __init__(self, backbone, edain_layer):
        super().__init__()
        self.edain = edain_layer
        self.backbone = backbone

    def forward(self, x, fg_mean=None, fg_std=None):
        if fg_mean is not None and fg_std is not None:
            x = self.edain(x, fg_mean, fg_std)
        else:
            x = self.edain(x)
        return self.backbone(x)




def otsu_select_fn(x):
    """Otsu阈值前景检测"""
    x = x.astype(np.float32)
    thr = threshold_otsu(x)
    return x > thr


class CropForegroundWithLabeld(MapTransform):
    """
    自定义前景裁剪Transform，确保bbox同时包含fg_mask和label的范围。

    改进点：
    1. 计算fg_mask（Otsu前景）的边界框
    2. 计算label的边界框
    3. 取两者的并集作为最终裁剪区域
    4. 应用margin

    这样可以避免肿瘤标签被裁剪掉的问题。

    Args:
        keys: 要处理的键（image和label）
        source_key: 用于计算前景的源键（通常是image）
        label_key: 标签键，用于计算label bbox
        select_fn: 前景选择函数（如otsu_select_fn）
        margin: 裁剪边距
    """

    def __init__(
            self,
            keys: KeysCollection,
            source_key: str = "image",
            label_key: str = "label",
            select_fn=None,
            margin=(10, 10, 3),
    ):
        super().__init__(keys)
        self.source_key = source_key
        self.label_key = label_key
        self.select_fn = select_fn if select_fn is not None else otsu_select_fn
        self.margin = margin

    def __call__(self, data):
        d = dict(data)

        # 获取源图像
        source = d[self.source_key]
        if torch.is_tensor(source):
            source_np = source.detach().cpu().numpy()
        else:
            source_np = np.asarray(source)

        # 获取空间数据（去掉channel维度）
        if source_np.ndim == 4:
            volume_for_fg = source_np[0]
        else:
            volume_for_fg = source_np

        spatial_shape = volume_for_fg.shape
        margin = np.array(self.margin)

        # === 计算fg_mask的边界框 ===
        fg_mask = self.select_fn(volume_for_fg)
        fg_coords = np.argwhere(fg_mask)
        has_fg = len(fg_coords) > 0

        if has_fg:
            fg_mins = fg_coords.min(axis=0)
            fg_maxs = fg_coords.max(axis=0) + 1
        else:
            print("[WARN] No foreground found, will use label bbox only")
            fg_mins = np.array([0, 0, 0])
            fg_maxs = np.array(spatial_shape)

        # === 计算label的边界框 ===
        label_coords = np.array([]).reshape(0, 3)
        if self.label_key in d:
            label = d[self.label_key]
            if torch.is_tensor(label):
                label_np = label.detach().cpu().numpy()
            else:
                label_np = np.asarray(label)

            if label_np.ndim == 4:
                label_spatial = label_np[0]
            else:
                label_spatial = label_np

            label_coords = np.argwhere(label_spatial > 0)

        has_label = len(label_coords) > 0

        if has_label:
            label_mins = label_coords.min(axis=0)
            label_maxs = label_coords.max(axis=0) + 1
        else:
            label_mins = fg_mins.copy()
            label_maxs = fg_maxs.copy()

        # === 检查是否有有效区域 ===
        if not has_fg and not has_label:
            print("[WARN] No foreground and no label found, returning original data")
            return d

        # === 取fg_mask和label边界框的并集 ===
        if has_fg:
            combined_mins = np.minimum(fg_mins, label_mins)
            combined_maxs = np.maximum(fg_maxs, label_maxs)
        else:
            combined_mins = label_mins
            combined_maxs = label_maxs

        # === 应用margin ===
        crop_start = np.maximum(combined_mins - margin, 0)
        crop_end = np.minimum(combined_maxs + margin, spatial_shape)

        # 创建slice对象
        spatial_slices = tuple(slice(int(s), int(e)) for s, e in zip(crop_start, crop_end))

        # === 执行裁剪 ===
        for key in self.keys:
            img = d[key]

            if img.ndim == 4:
                full_slices = (slice(None),) + spatial_slices
            else:
                full_slices = spatial_slices

            if torch.is_tensor(img):
                d[key] = img[full_slices]
            else:
                d[key] = img[full_slices].copy()

        return d




class NnUNetRandRotateScaled(MapTransform, RandomizableTransform):
    """nnU-Net风格的旋转和缩放增强 - 支持各向异性数据"""

    def __init__(
            self,
            keys: KeysCollection,
            rotate_range: Tuple[float, float] = (-30, 30),
            scale_range: Tuple[float, float] = (0.7, 1.4),
            anisotropic: bool = False,
            anisotropy_axis: int = 2,
    ):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=1.0)

        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.anisotropic = anisotropic
        self.anisotropy_axis = anisotropy_axis

        self._do_rotate = False
        self._do_scale = False
        self._rotation_angles = [0.0, 0.0, 0.0]
        self._scale_factor = 1.0

    def randomize(self, data=None):
        r = self.R.random()

        if r < 0.08:
            self._do_rotate = True
            self._do_scale = True
        elif r < 0.24:
            self._do_rotate = False
            self._do_scale = True
        elif r < 0.40:
            self._do_rotate = True
            self._do_scale = False
        else:
            self._do_rotate = False
            self._do_scale = False

        self._rotation_angles = [0.0, 0.0, 0.0]
        if self._do_rotate:
            if self.anisotropic:
                self._rotation_angles[self.anisotropy_axis] = np.deg2rad(
                    self.R.uniform(*self.rotate_range)
                )
            else:
                self._rotation_angles = [
                    np.deg2rad(self.R.uniform(*self.rotate_range)),
                    np.deg2rad(self.R.uniform(*self.rotate_range)),
                    np.deg2rad(self.R.uniform(*self.rotate_range))
                ]

        self._scale_factor = self.R.uniform(*self.scale_range) if self._do_scale else 1.0

    def _build_affine_backward(self, shape):
        center = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
        ax, ay, az = self._rotation_angles

        Rz = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)]
        ])

        R = Rz @ Ry @ Rx
        S = np.diag([self._scale_factor] * 3)
        M = S @ R

        invM = np.linalg.inv(M)
        offset = center - invM @ center

        return invM, offset

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)

        if not self._do_rotate and not self._do_scale:
            return d

        for key in self.keys:
            img = d[key]
            is_label = "label" in key.lower()
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = np.asarray(img)

            result = np.zeros_like(img_np)
            spatial_shape = img_np.shape[1:]

            invM, offset = self._build_affine_backward(spatial_shape)

            for c in range(img_np.shape[0]):
                vol = img_np[c]
                order = 0 if is_label else 3
                mode = 'constant'

                result[c] = affine_transform(
                    vol, matrix=invM, offset=offset,
                    order=order, mode=mode, cval=0.0
                )

            # 保持MetaTensor类型
            if is_metatensor:
                d[key] = MetaTensor(torch.from_numpy(result), meta=img.meta)
            elif torch.is_tensor(img):
                d[key] = torch.from_numpy(result)
            else:
                d[key] = result

        return d


class CenterSpatialCropWithPadd(MapTransform):
    """Center crop到目标大小，如果输入小于目标则先pad"""

    def __init__(self, keys: KeysCollection, roi_size: Tuple[int, ...]):
        MapTransform.__init__(self, keys)
        self.roi_size = roi_size

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            img = d[key]
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img

            spatial_shape = img_np.shape[1:]

            # 计算padding
            pad_need = []
            for i in range(3):
                diff = self.roi_size[i] - spatial_shape[i]
                if diff > 0:
                    pad_before = diff // 2
                    pad_after = diff - pad_before
                    pad_need.append((pad_before, pad_after))
                else:
                    pad_need.append((0, 0))

            # Padding
            if any(p[0] > 0 or p[1] > 0 for p in pad_need):
                channel_pad = [(0, 0)]
                full_pad = channel_pad + pad_need
                img_np = np.pad(img_np, full_pad, mode='constant', constant_values=0)
                spatial_shape = img_np.shape[1:]

            # Center crop
            crop_start = []
            crop_end = []
            for i in range(3):
                diff = spatial_shape[i] - self.roi_size[i]
                start = max(0, diff // 2)
                end = start + self.roi_size[i]
                crop_start.append(start)
                crop_end.append(end)

            slices = (slice(None),) + tuple(slice(s, e) for s, e in zip(crop_start, crop_end))
            result = img_np[slices].copy()

            # 保持MetaTensor类型
            if is_metatensor:
                d[key] = MetaTensor(torch.from_numpy(result), meta=img.meta)
            elif torch.is_tensor(img):
                d[key] = torch.from_numpy(result)
            else:
                d[key] = result

        return d


class NnUNetRandGaussianNoised(MapTransform, RandomizableTransform):
    """高斯噪声增强"""

    def __init__(self, keys: KeysCollection, prob: float = 0.15,
                 variance_range: Tuple[float, float] = (0.0, 0.1)):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.variance_range = variance_range
        self._variance = 0.0

    def randomize(self, data=None):
        super().randomize(None)
        if self._do_transform:
            self._variance = self.R.uniform(*self.variance_range)

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)

        if not self._do_transform:
            return d

        for key in self.keys:
            img = d[key]
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img

            noise = self.R.normal(0, np.sqrt(self._variance), img_np.shape).astype(np.float32)
            result = img_np + noise

            if is_metatensor:
                d[key] = MetaTensor(torch.from_numpy(result), meta=img.meta)
            elif torch.is_tensor(img):
                d[key] = torch.from_numpy(result)
            else:
                d[key] = result

        return d


class NnUNetRandGaussianSmoothd(MapTransform, RandomizableTransform):
    """高斯模糊增强"""

    def __init__(self, keys: KeysCollection, prob: float = 0.2,
                 prob_per_channel: float = 0.5,
                 sigma_range: Tuple[float, float] = (0.5, 1.5)):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.prob_per_channel = prob_per_channel
        self.sigma_range = sigma_range
        self._channel_flags = []
        self._sigmas = []

    def randomize(self, data=None, num_channels=1):
        super().randomize(None)
        self._channel_flags = []
        self._sigmas = []

        if self._do_transform:
            for _ in range(num_channels):
                do_channel = self.R.random() < self.prob_per_channel
                self._channel_flags.append(do_channel)
                self._sigmas.append(self.R.uniform(*self.sigma_range) if do_channel else 0)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            img = d[key]
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img

            num_channels = img_np.shape[0]
            self.randomize(d, num_channels)

            if not self._do_transform:
                continue

            result = img_np.copy()
            for c in range(num_channels):
                if self._channel_flags[c]:
                    result[c] = gaussian_filter(img_np[c], sigma=self._sigmas[c])

            if is_metatensor:
                d[key] = MetaTensor(torch.from_numpy(result), meta=img.meta)
            elif torch.is_tensor(img):
                d[key] = torch.from_numpy(result)
            else:
                d[key] = result

        return d


class NnUNetRandBrightnessd(MapTransform, RandomizableTransform):
    """亮度增强"""

    def __init__(self, keys: KeysCollection, prob: float = 0.15,
                 factor_range: Tuple[float, float] = (0.7, 1.3)):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.factor_range = factor_range
        self._factor = 1.0

    def randomize(self, data=None):
        super().randomize(None)
        if self._do_transform:
            self._factor = self.R.uniform(*self.factor_range)

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)

        if not self._do_transform:
            return d

        for key in self.keys:
            img = d[key]
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img

            result = img_np * self._factor

            if is_metatensor:
                d[key] = MetaTensor(torch.from_numpy(result.astype(np.float32)), meta=img.meta)
            elif torch.is_tensor(img):
                d[key] = torch.from_numpy(result.astype(np.float32))
            else:
                d[key] = result

        return d


class NnUNetRandContrastd(MapTransform, RandomizableTransform):
    """对比度增强"""

    def __init__(self, keys: KeysCollection, prob: float = 0.15,
                 factor_range: Tuple[float, float] = (0.65, 1.5)):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.factor_range = factor_range
        self._factor = 1.0

    def randomize(self, data=None):
        super().randomize(None)
        if self._do_transform:
            self._factor = self.R.uniform(*self.factor_range)

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)

        if not self._do_transform:
            return d

        for key in self.keys:
            img = d[key]
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img

            result = img_np.copy()
            for c in range(img_np.shape[0]):
                channel_mean = img_np[c].mean()
                result[c] = (img_np[c] - channel_mean) * self._factor + channel_mean

            if is_metatensor:
                d[key] = MetaTensor(torch.from_numpy(result.astype(np.float32)), meta=img.meta)
            elif torch.is_tensor(img):
                d[key] = torch.from_numpy(result.astype(np.float32))
            else:
                d[key] = result

        return d


class NnUNetRandSimulateLowResolutiond(MapTransform, RandomizableTransform):
    """低分辨率模拟增强"""

    def __init__(self, keys: KeysCollection, prob: float = 0.25,
                 prob_per_channel: float = 0.5,
                 downsample_factor_range: Tuple[float, float] = (1.0, 2.0),
                 anisotropic: bool = False,
                 anisotropy_axis: int = 2):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.prob_per_channel = prob_per_channel
        self.downsample_factor_range = downsample_factor_range
        self.anisotropic = anisotropic
        self.anisotropy_axis = anisotropy_axis
        self._channel_flags = []
        self._factors = []

    def randomize(self, data=None, num_channels=1):
        super().randomize(None)
        self._channel_flags = []
        self._factors = []

        if self._do_transform:
            for _ in range(num_channels):
                do_channel = self.R.random() < self.prob_per_channel
                self._channel_flags.append(do_channel)
                self._factors.append(self.R.uniform(*self.downsample_factor_range) if do_channel else 1.0)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            img = d[key]
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img

            num_channels = img_np.shape[0]
            self.randomize(d, num_channels)

            if not self._do_transform:
                continue

            result = img_np.copy()
            original_shape = img_np.shape[1:]

            for c in range(num_channels):
                if self._channel_flags[c]:
                    factor = self._factors[c]
                    vol = img_np[c]

                    zoom_down = [1.0 / factor] * 3
                    zoom_up = [factor] * 3

                    if self.anisotropic:
                        zoom_down[self.anisotropy_axis] = 1.0
                        zoom_up[self.anisotropy_axis] = 1.0

                    downsampled = zoom(vol, zoom_down, order=0, mode='nearest')
                    upsampled = zoom(downsampled, zoom_up, order=3, mode='nearest')

                    result[c] = self._match_shape(upsampled, original_shape)

            if is_metatensor:
                d[key] = MetaTensor(torch.from_numpy(result.astype(np.float32)), meta=img.meta)
            elif torch.is_tensor(img):
                d[key] = torch.from_numpy(result.astype(np.float32))
            else:
                d[key] = result

        return d

    def _match_shape(self, arr, target_shape):
        result = np.zeros(target_shape, dtype=arr.dtype)
        slices_src, slices_dst = [], []
        for i in range(len(target_shape)):
            copy_size = min(arr.shape[i], target_shape[i])
            src_start = (arr.shape[i] - copy_size) // 2
            dst_start = (target_shape[i] - copy_size) // 2
            slices_src.append(slice(src_start, src_start + copy_size))
            slices_dst.append(slice(dst_start, dst_start + copy_size))
        result[tuple(slices_dst)] = arr[tuple(slices_src)]
        return result


class NnUNetRandGammaD(MapTransform, RandomizableTransform):
    """Gamma增强"""

    def __init__(self, keys: KeysCollection, prob: float = 0.15,
                 gamma_range: Tuple[float, float] = (0.7, 1.5),
                 invert_image: Optional[bool] = False,
                 invert_prob: float = 0.15):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.invert_prob = invert_prob
        self._gamma = 1.0
        self._invert = False

    def randomize(self, data=None):
        super().randomize(None)
        if self._do_transform:
            self._gamma = self.R.uniform(*self.gamma_range)
            if self.invert_image is None:
                self._invert = (self.R.random() < self.invert_prob)
            else:
                self._invert = bool(self.invert_image)

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)

        if not self._do_transform:
            return d

        for key in self.keys:
            img = d[key]
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img

            result = img_np.copy()

            for c in range(img_np.shape[0]):
                vol = img_np[c]
                orig_min, orig_max = vol.min(), vol.max()

                if orig_max - orig_min > 1e-8:
                    vol_normalized = (vol - orig_min) / (orig_max - orig_min)
                else:
                    vol_normalized = np.zeros_like(vol)

                vol_normalized = np.clip(vol_normalized, 0, 1)

                if self._invert:
                    vol_gamma = 1.0 - np.power(np.clip(1.0 - vol_normalized, 1e-8, 1.0), self._gamma)
                else:
                    vol_gamma = np.power(np.clip(vol_normalized, 1e-8, 1.0), self._gamma)

                result[c] = vol_gamma * (orig_max - orig_min) + orig_min

            if is_metatensor:
                d[key] = MetaTensor(torch.from_numpy(result.astype(np.float32)), meta=img.meta)
            elif torch.is_tensor(img):
                d[key] = torch.from_numpy(result.astype(np.float32))
            else:
                d[key] = result

        return d


class NnUNetRandFlipd(MapTransform, RandomizableTransform):
    """镜像翻转"""

    def __init__(self, keys: KeysCollection, prob: float = 0.5,
                 spatial_axes: Tuple[int, ...] = (0, 1, 2)):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=1.0)
        self.flip_prob = prob
        self.spatial_axes = spatial_axes
        self._axes_to_flip = []

    def randomize(self, data=None):
        self._axes_to_flip = []
        for ax in self.spatial_axes:
            if self.R.random() < self.flip_prob:
                self._axes_to_flip.append(ax)

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)

        if not self._axes_to_flip:
            return d

        flip_axes = [ax + 1 for ax in self._axes_to_flip]

        for key in self.keys:
            img = d[key]
            is_metatensor = isinstance(img, MetaTensor)

            if torch.is_tensor(img):
                flipped = torch.flip(img, dims=flip_axes)
                if is_metatensor:
                    d[key] = MetaTensor(flipped, meta=img.meta)
                else:
                    d[key] = flipped
            else:
                d[key] = np.flip(img, axis=flip_axes).copy()

        return d


# ==============================================================================
# Helper Functions
# ==============================================================================

def _dice_from_labels(pred, target, eps: float = 1e-6):
    """计算Batch的平均Dice Score"""
    if pred.dim() == 5 and pred.size(1) == 1:
        pred = pred[:, 0]
    if target.dim() == 5 and target.size(1) == 1:
        target = target[:, 0]

    pred_bin = (pred > 0.5)
    target_bin = (target > 0.5)

    intersect = (pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    union = pred_bin.sum(dim=(1, 2, 3)).float() + target_bin.sum(dim=(1, 2, 3)).float()
    dice = (2.0 * intersect + eps) / (union + eps)
    return dice.mean()


def extract_subject_id(path: str) -> str:
    """从文件路径提取受试者ID"""
    base = os.path.basename(path)
    m = re.search(r'(Lipo-\d+)', base)
    if m:
        return m.group(1)
    return base.split('_')[0]


def poly_lr(epoch: int, max_epochs: int):
    """nnU-Net style polynomial LR decay"""
    return (1 - epoch / max_epochs) ** 0.9


def get_lr(optim):
    """获取当前学习率"""
    for pg in optim.param_groups:
        return pg.get("lr", None)


def get_deep_supervision_weights(num_stages):
    """生成深监督权重"""
    weights = np.array([1.0 / (2 ** i) for i in range(num_stages)])
    weights = weights / weights.sum()
    return weights

def normalize_logits_output(logits):
    """标准化模型输出为list格式"""
    if isinstance(logits, (list, tuple)):
        return list(logits)
    elif logits.dim() == 6:
        return list(logits.unbind(dim=1))
    else:
        return [logits]



def build_folds_from_split_json(all_files, split_json_path: str):
    """Build folds using a precomputed split.json."""
    with open(split_json_path, "r") as f:
        split = json.load(f)

    folds = []
    for k, v in split.items():
        fold_id = int(k)
        val_subjects = set(v.get("val_subjects_sorted", []))
        train_files = [d for d in all_files if d["subject"] not in val_subjects]
        val_files = [d for d in all_files if d["subject"] in val_subjects]
        folds.append({"fold": fold_id, "train_files": train_files, "val_files": val_files})

    folds = sorted(folds, key=lambda x: x["fold"])
    return folds


def compute_oversized_patch(patch_size, scale_range=(0.7, 1.4), rotate_range=(-30, 30)):
    """计算Oversized Patch大小"""
    oversized_factor = 1.4
    oversized = [int(np.ceil(ps * oversized_factor)) for ps in patch_size]
    return tuple(oversized)



# ==============================================================================
# Optimizer Setup with Separate Learning Rates
# ==============================================================================

def create_optimizer_with_edain_lr(model, base_lr, edain_lr_multipliers=None):
    """Create optimizer with separate learning rates for EDAIN parameters."""
    if edain_lr_multipliers is None:
        edain_lr_multipliers = {'m': 0.1, 's': 0.1}

    param_groups = []

    edain_params = {
        'm': model.edain._m_raw,
        's': model.edain._s_raw,
    }

    for name, param in edain_params.items():
        lr_mult = edain_lr_multipliers.get(name, 1.0)
        param_groups.append({
            'params': [param],
            'lr': base_lr * lr_mult,
            'name': f'edain_{name}'
        })

    backbone_params = list(model.backbone.parameters())
    param_groups.append({
        'params': backbone_params,
        'lr': base_lr,
        'name': 'backbone'
    })

    optimizer = torch.optim.SGD(param_groups, momentum=0.99, nesterov=True, weight_decay=1e-5)
    return optimizer


def update_lr_with_scheduler(optimizer, epoch, max_epochs, base_lr, edain_lr_multipliers):
    """Update learning rates with polynomial decay."""
    decay_factor = poly_lr(epoch, max_epochs)

    for pg in optimizer.param_groups:
        name = pg.get('name', '')
        if name.startswith('edain_'):
            param_name = name.replace('edain_', '')
            mult = edain_lr_multipliers.get(param_name, 1.0)
            pg['lr'] = base_lr * mult * decay_factor
        else:
            pg['lr'] = base_lr * decay_factor



# ==============================================================================
# Main Execution Block
# ==============================================================================


# ==============================================================================
# CLI helpers (for Slurm job arrays / running a single fold)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Medical Image Segmentation (supports Slurm job arrays)."
    )
    parser.add_argument("--base_dir", type=str, default="../dataset/LIPO",
                        help="Dataset base directory.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Global output directory (default: ./outputs/<method_name>).")
    parser.add_argument("--fold", type=int, default=None,
                        help="If set, run only a single fold id in [0..n_splits-1].")
    parser.add_argument("--split_json", type=str, default="./split.json",
                        help="Path to split.json for fixed CV split.")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of CV folds (used when --split_json is not provided).")
    return parser.parse_args()


if __name__ == '__main__':
    set_determinism(seed=2025)
    args = parse_args()

    # ---------------------------------------------------------------------
    # 0. Configuration
    # ---------------------------------------------------------------------
    base_dir = args.base_dir
    split_json = args.split_json
    global_output_dir = args.out_dir if args.out_dir else "./outputs/method4_edain_minmax"
    os.makedirs(global_output_dir, exist_ok=True)


    # ---------------------------------------------------------------------
    # 1. Data Loading
    # ---------------------------------------------------------------------
    image_pattern = os.path.join(base_dir, "Lipo-*_MR_*_image.nii.gz")
    image_paths = sorted(glob.glob(image_pattern))

    pairs = []
    for img_path in image_paths:
        seg_path = img_path.replace("_image.nii.gz", "_segmentation.nii.gz")
        if os.path.exists(seg_path):
            pairs.append((img_path, seg_path))
        else:
            print(f"[WARN] Segmentation not found for: {os.path.basename(img_path)}")

    if not pairs:
        raise FileNotFoundError("No image-label pairs found.")

    all_files = []
    for img_path, seg_path in pairs:
        subj = extract_subject_id(img_path)
        all_files.append({"image": img_path, "label": seg_path, "subject": subj})

    subjects = np.array([d["subject"] for d in all_files])
    indices = np.arange(len(all_files))
    print(f"[INFO] Total pairs loaded: {len(all_files)}")

    # ---------------------------------------------------------------------
    # 2. Dataset Fingerprint Calculation
    # ---------------------------------------------------------------------
    print("[INFO] Calculating dataset fingerprint...")
    all_spacings = []
    all_shapes = []
    for img_path, seg_path in pairs:
        img = nib.load(img_path)
        spacing = img.header.get_zooms()[:3]
        shape = img.header.get_data_shape()[:3]
        all_spacings.append(spacing)
        all_shapes.append(shape)

    all_spacings = np.array(all_spacings)
    median_spacing = np.median(all_spacings, axis=0)

    # Check Anisotropy
    anisotropic = False
    anisotropy_axis = None
    max_spacing = median_spacing.max()
    min_spacing = median_spacing.min()
    target_spacing = median_spacing.copy()

    if max_spacing / min_spacing >= 3.0:
        anisotropic = True
        anisotropy_axis = int(np.argmax(median_spacing))
        percentile_10th_spacing = np.percentile(all_spacings[:, anisotropy_axis], 10)
        target_spacing[anisotropy_axis] = percentile_10th_spacing

    # Calculate Median Shape after Resampling
    resampled_shapes = []
    for shape, spacing in zip(all_shapes, all_spacings):
        new_shape = np.array(shape) * (np.array(spacing) / target_spacing)
        resampled_shapes.append(new_shape)

    median_shape = np.median(np.array(resampled_shapes), axis=0).astype(int)
    target_pixdim = tuple(target_spacing.tolist())

    # Patch Size Estimation
    patch_size = [0, 0, 0]
    if anisotropic:
        axis = anisotropy_axis
        patch_size[axis] = int(min(median_shape[axis], 128))
        for ax in range(3):
            if ax != axis:
                patch_size[ax] = int(min(median_shape[ax], 192))
    else:
        for ax in range(3):
            patch_size[ax] = int(min(median_shape[ax], 128))

    # Network Topology
    channels = [32]
    strides = []
    current_shape = np.array(patch_size, dtype=np.int32)
    max_features = 512

    while True:
        if anisotropic and len(strides) < 2:
            stride = [2, 2, 2]
            stride[anisotropy_axis] = 1
        else:
            stride = [2, 2, 2]

        new_shape = np.ceil(current_shape / stride)
        if (new_shape < 4).any():
            break

        strides.append(tuple(stride))
        current_shape = new_shape
        next_feat = min(channels[-1] * 2, max_features)
        channels.append(next_feat)

    channels = tuple(channels)

    # Final Patch Size Adjustment
    total_stride = np.array([1, 1, 1], dtype=np.int32)
    for stride in strides:
        total_stride *= np.array(stride, dtype=np.int32)

    patch_size_fixed = list(patch_size)
    for ax in range(3):
        current_size = patch_size_fixed[ax]
        stride_val = total_stride[ax]
        if current_size < stride_val:
            patch_size_fixed[ax] = stride_val
        elif current_size % stride_val != 0:
            patch_size_fixed[ax] = int(np.ceil(current_size / stride_val) * stride_val)

    patch_size = tuple(patch_size_fixed)
    oversized_patch = compute_oversized_patch(patch_size)

    print(f"[INFO] Target patch size: {patch_size}")
    print(f"[INFO] Oversized patch size: {oversized_patch}")
    print(f"[INFO] Target spacing: {target_pixdim}")
    print(f"[INFO] Network channels: {channels}")
    print(f"[INFO] Network strides: {strides}")
    print(f"[INFO] Anisotropic: {anisotropic}, Axis: {anisotropy_axis}")

    # Kernel Sizes
    num_stages = len(channels)
    kernel_sizes = []
    for stage in range(num_stages):
        if anisotropic and stage < 2:
            kernel_sizes.append((3, 3, 1))
        else:
            kernel_sizes.append((3, 3, 3))

    up_kernel_sizes = tuple(strides)
    dynunet_strides = tuple([(1, 1, 1)] + list(strides))


    # ---------------------------------------------------------------------
    # 3. Data Transforms
    # ---------------------------------------------------------------------
    num_patches_per_image = 4

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim, mode=(3, "nearest")),

        # === 前景裁剪 ===
        CropForegroundWithLabeld(keys=["image", "label"], source_key="image",
                                 label_key="label", select_fn=otsu_select_fn, margin=(10, 10, 3)),
        # === Robust Min-Max 映射到 [0,1]（固定，不可学） ===
        RobustMinMaxNormalized(keys=["image"], lower_pct=2.0, upper_pct=98.0),

        # === Oversized Patch采样 ===
        SpatialPadd(keys=["image", "label"], spatial_size=oversized_patch),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=oversized_patch,
            pos=2, neg=1,
            num_samples=num_patches_per_image,
        ),

        # === nnU-Net V3 空间增强（在oversized patch上做）===
        NnUNetRandRotateScaled(
            keys=["image", "label"],
            rotate_range=(-180, 180) if anisotropic else (-30, 30),
            scale_range=(0.7, 1.4),
            anisotropic=anisotropic,
            anisotropy_axis=anisotropy_axis if anisotropic else 2,
        ),
        CenterSpatialCropWithPadd(
            keys=["image", "label"],
            roi_size=patch_size,
        ),

        # === 强度增强 ===
        NnUNetRandGaussianNoised(keys=["image"], prob=0.15, variance_range=(0.0, 0.1)),
        NnUNetRandGaussianSmoothd(keys=["image"], prob=0.2, prob_per_channel=0.5, sigma_range=(0.5, 1.5)),
        NnUNetRandBrightnessd(keys=["image"], prob=0.15, factor_range=(0.7, 1.3)),
        NnUNetRandContrastd(keys=["image"], prob=0.15, factor_range=(0.65, 1.5)),
        NnUNetRandSimulateLowResolutiond(
            keys=["image"], prob=0.25, prob_per_channel=0.5,
            downsample_factor_range=(1.0, 2.0),
            anisotropic=anisotropic,
            anisotropy_axis=anisotropy_axis if anisotropic else 2,
        ),
        NnUNetRandGammaD(keys=["image"], prob=0.15, gamma_range=(0.7, 1.5), invert_image=None, invert_prob=0.15),
        NnUNetRandFlipd(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1, 2)),

        # === 类型转换 ===
        EnsureTyped(keys=["image"], dtype=np.float32),
        EnsureTyped(keys=["label"], dtype=np.uint8),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim, mode=(3, "nearest")),

        CropForegroundWithLabeld(keys=["image", "label"], source_key="image",
                                 label_key="label", select_fn=otsu_select_fn, margin=(10, 10, 3)),
        RobustMinMaxNormalized(keys=["image"], lower_pct=2.0, upper_pct=98.0),

        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ToTensord(keys=["image", "label"]),
    ])

    post_process = KeepLargestConnectedComponent(applied_labels=[1])


    # ---------------------------------------------------------------------
    # 4. Cross-Validation
    # ---------------------------------------------------------------------
    if split_json and os.path.exists(split_json):
        print(f"[INFO] Using fixed split from: {split_json}")
        folds = build_folds_from_split_json(all_files, split_json)
    else:
        print("[INFO] split.json not found, using GroupKFold")
        gkf = GroupKFold(n_splits=args.n_splits)
        folds = []
        for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(indices, groups=subjects)):
            train_files = [all_files[i] for i in tr_idx]
            val_files = [all_files[i] for i in va_idx]
            folds.append({"fold": fold_id, "train_files": train_files, "val_files": val_files})

    # If running only one fold (for Slurm job arrays)
    if args.fold is not None:
        if args.fold < 0 or args.fold >= len(folds):
            raise ValueError(f"--fold must be in [0, {len(folds)-1}]")
        folds = [folds[args.fold]]
        print(f"[INFO] Running single fold: {args.fold}")


    # Training Configuration
    pin_memory = torch.cuda.is_available()
    num_workers = 4
    max_epochs = 1000
    val_interval = 2
    early_stop_patience = 75
    base_lr = 0.01

    # EDAIN learning rate multipliers (smaller than backbone for stability)
    edain_lr_multipliers = {'m': 0.1, 's': 0.1}

    for f in folds:
        CURR_FOLD = f["fold"]
        print(f"\n{'=' * 60}")
        print(f"Running Fold {CURR_FOLD} - PreMappedEDAIN")
        print(f"{'=' * 60}")

        # Datasets
        train_ds = CacheDataset(data=f["train_files"], transform=train_transforms,
                                cache_rate=0, num_workers=num_workers)
        val_ds = CacheDataset(data=f["val_files"], transform=val_transforms,
                              cache_rate=1, num_workers=num_workers)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, collate_fn=list_data_collate,
                                  persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory, persistent_workers=(num_workers > 0))

        out_dir = os.path.join(global_output_dir, f"fold_{CURR_FOLD}")
        os.makedirs(out_dir, exist_ok=True)
        fold_dice_csv = os.path.join(out_dir, f"final_dice_fold_{CURR_FOLD}.csv")
        if not os.path.exists(fold_dice_csv):
            with open(fold_dice_csv, "w", newline="") as fcsv:
                csv.writer(fcsv).writerow(["SubjectID", "Fold", "Dice", "PredictionPath"])

        ckpt_path = os.path.join(out_dir, "best_model.pt")
        log_path = os.path.join(out_dir, "training_log.csv")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create backbone
        backbone = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            filters=channels, strides=dynunet_strides,
            kernel_size=kernel_sizes, upsample_kernel_size=up_kernel_sizes,
            act_name=("leakyrelu", {"negative_slope": 0.01}),
            norm_name="instance",
            deep_supervision=True, deep_supr_num=3
        )

        # Create EDAIN layer and wrap model
        edain_layer = PreMappedEDAIN()
        model = SegmentationModelWithEDAIN(backbone, edain_layer).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        edain_params_count = sum(p.numel() for p in model.edain.parameters() if p.requires_grad)
        print(f"[INFO] Total Trainable Parameters: {total_params / 1e6:.2f} M")
        print(f"[INFO] EDAIN Parameters: {edain_params_count}")

        # Loss Function
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True,
                             include_background=False, lambda_dice=1.0, lambda_ce=1.0)

        # Optimizer with separate LRs
        optimizer = create_optimizer_with_edain_lr(model, base_lr, edain_lr_multipliers)
        scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        # Initialize Log File
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as fcsv:
                csv.writer(fcsv).writerow([
                    "epoch", "train_loss", "train_dice", "val_loss", "val_dice",
                    "lr", "time_sec", "m", "s"
                ])

        best_metric = -1.0
        best_epoch = -1
        epochs_no_improve = 0

        # --- Training Loop ---
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss_accum = 0.0
            train_dice_accum = 0.0
            num_batches = 0

            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).long()

                optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    logits = model(images)
                    logits = normalize_logits_output(logits)

                    loss = 0
                    ds_weights = get_deep_supervision_weights(len(logits))

                    for i, head_logit in enumerate(logits):
                        current_spatial_shape = head_logit.shape[-3:]

                        if current_spatial_shape != labels.shape[-3:]:
                            ds_labels = F.interpolate(labels.float(), size=current_spatial_shape,
                                                      mode="nearest").long()
                        else:
                            ds_labels = labels

                        l = loss_fn(head_logit, ds_labels)
                        loss += l * ds_weights[i]

                    train_pred = logits[0].argmax(dim=1, keepdim=True)
                    train_dice_val = _dice_from_labels(train_pred, labels)
                    train_dice_accum += train_dice_val.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_accum += loss.item()
                num_batches += 1

            epoch_loss = train_loss_accum / num_batches
            epoch_train_dice = train_dice_accum / num_batches

            # --- Validation Loop ---
            val_loss_epoch = None
            val_dice_epoch = None

            if epoch % val_interval == 0:
                model.eval()
                dice_scores = []
                val_loss_list = []

                with torch.inference_mode(), amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    for val_data in val_loader:
                        val_images = val_data["image"].to(device)
                        val_labels = val_data["label"].to(device).long()

                        val_logits = sliding_window_inference(
                            inputs=val_images, roi_size=patch_size, sw_batch_size=4,
                            predictor=model, overlap=0.5, mode="gaussian",
                        )

                        if isinstance(val_logits, (list, tuple)):
                            val_logits = val_logits[0]

                        v_loss = loss_fn(val_logits, val_labels)
                        val_loss_list.append(v_loss.item())

                        val_pred = val_logits.argmax(dim=1, keepdim=True)
                        val_pred = post_process(val_pred)
                        dice_val = _dice_from_labels(pred=val_pred, target=val_labels)
                        dice_scores.append(float(dice_val.item()))

                val_dice_epoch = float(np.mean(dice_scores))
                val_loss_epoch = float(np.mean(val_loss_list))

                if val_dice_epoch > best_metric:
                    best_metric = val_dice_epoch
                    best_epoch = epoch
                    epochs_no_improve = 0
                    torch.save({
                        "model": model.state_dict(),
                        "edain_params": model.edain.get_params_info()
                    }, ckpt_path)
                    print(f"   >>> New Best: Val_Dice={best_metric:.4f} | Train_Dice={epoch_train_dice:.4f}")
                else:
                    epochs_no_improve += val_interval

            # Update learning rates
            update_lr_with_scheduler(optimizer, epoch, max_epochs, base_lr, edain_lr_multipliers)

            elapsed = time.time() - t0
            edain_info = model.edain.get_params_info()

            log_str = f"Ep {epoch:03d} | Loss={epoch_loss:.4f} | Train_Dice={epoch_train_dice:.4f}"
            if val_dice_epoch is not None:
                log_str += f" | Val_Loss={val_loss_epoch:.4f} | Val_Dice={val_dice_epoch:.4f} | Best={best_metric:.4f}"
            log_str += f" | LR={get_lr(optimizer):.6f} | Time={elapsed:.1f}s"
            log_str += f" | m={edain_info['m']:.3f} s={edain_info['s']:.3f}"
            print(log_str)

            with open(log_path, "a", newline="") as fcsv:
                csv.writer(fcsv).writerow([
                    epoch,
                    f"{epoch_loss:.6f}", f"{epoch_train_dice:.6f}",
                    ("" if val_loss_epoch is None else f"{val_loss_epoch:.6f}"),
                    ("" if val_dice_epoch is None else f"{val_dice_epoch:.6f}"),
                    f"{get_lr(optimizer):.6f}", f"{elapsed:.2f}",
                    f"{edain_info['m']:.6f}", f"{edain_info['s']:.6f}"
                ])

            if epochs_no_improve >= early_stop_patience:
                print(f"\n[Early Stopping] No improvement for {early_stop_patience} epochs.")
                break

        print(f"\nFold {CURR_FOLD} Complete | Best Epoch: {best_epoch} | Best Val Dice: {best_metric:.4f}")

        final_edain = model.edain.get_params_info()
        print(f"Final EDAIN Parameters: " + ", ".join([f"{k}={v:.4f}" for k, v in final_edain.items()]))


        # ---------------------------------------------------------------------
        # 5. OOF Inference
        # ---------------------------------------------------------------------
        print(f"\n{'=' * 50}")
        print(f"OOF Inference for Fold {CURR_FOLD}")
        print(f"{'=' * 50}")

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        pred_dir = os.path.join(out_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

        with torch.inference_mode():
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device).long()

                val_logits = sliding_window_inference(
                    val_images, patch_size, 4, model, overlap=0.5, mode="gaussian"
                )

                if isinstance(val_logits, (list, tuple)):
                    val_logits = val_logits[0]

                val_pred = val_logits.argmax(dim=1, keepdim=True)
                val_pred = post_process(val_pred)

                current_dice = _dice_from_labels(pred=val_pred, target=val_labels).item()

                try:
                    if "subject" in val_data:
                        subj_id = val_data["subject"][0]
                    else:
                        subj_id = extract_subject_id(val_data["image_meta_dict"]["filename_or_obj"][0])
                except:
                    subj_id = f"Unknown"

                print(f"   {subj_id}: Dice={current_dice:.4f}")

                val_data["pred"] = val_pred
                val_data_unbatched = decollate_batch(val_data)

                for item in val_data_unbatched:
                    pred_to_save = item["pred"]
                    if torch.is_tensor(pred_to_save):
                        pred_to_save = pred_to_save.cpu()

                    if hasattr(item["image"], "meta"):
                        save_meta = item["image"].meta
                    else:
                        save_meta = item.get("image_meta_dict", None)

                    SaveImage(
                        output_dir=pred_dir,
                        output_postfix="pred",
                        separate_folder=False,
                        print_log=False
                    )(pred_to_save, save_meta)

                save_path = os.path.join(pred_dir, f"{subj_id}_pred.nii.gz")
                with open(fold_dice_csv, "a", newline="") as f:
                    csv.writer(f).writerow([subj_id, CURR_FOLD, f"{current_dice:.6f}", save_path])

    print("\n" + "=" * 60)
    print("All folds completed!")
    print("=" * 60)
