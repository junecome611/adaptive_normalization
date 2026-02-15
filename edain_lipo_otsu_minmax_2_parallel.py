"""
EDAIN-style Adaptive Robust Min-Max Normalization for 3D Medical Image Segmentation

This script modifies the original edain_lipo_1 by replacing z-score with robust min-max:
1. After Otsu thresholding, compute 2% and 98% percentiles as robust min and max
2. Apply adaptive normalization formula: x' = (x - m·min) / (s·(max - min))

Parameters:
    - alpha: [0, 1] - ratio of outlier mitigation (0=no winsorization, 1=full)
    - beta: [beta_min, inf) - range parameter for tanh compression
    - m: R - shift multiplier for min
    - s: (0, inf) - scale multiplier for range
"""

import os
import glob
import nibabel as nib
import numpy as np
import re
import csv
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch import amp
from skimage.filters import threshold_otsu

from monai.data import CacheDataset, DataLoader, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.utils import set_determinism
from monai.transforms import KeepLargestConnectedComponent
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandAffined, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, RandAdjustContrastd, RandSimulateLowResolutiond,
    EnsureTyped, ToTensord, SaveImage, MapTransform,
    RandomizableTransform, ScaleIntensityRangePercentilesd
)


# ==============================================================================
# CLI helpers (for Slurm job arrays / running a single fold)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EDAIN-style Adaptive Robust Min-Max Normalization for 3D segmentation (supports Slurm job arrays)."
    )
    parser.add_argument("--base_dir", type=str, default="../dataset/LIPO",
                        help="Dataset base directory (expects Lipo-*_image/seg NIfTI files).")
    parser.add_argument("--out_dir", type=str, default="./outputs/edain_minmax_06version",
                        help="Global output directory (fold subfolders will be created inside).")
    parser.add_argument("--fold", type=int, default=None,
                        help="If set, run only a single fold id in [0..n_splits-1].")
    parser.add_argument("--split_json", type=str, default=None,
                        help="Optional split.json path to enforce a fixed CV split (recommended on cluster).")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of CV folds (used when --split_json is not provided).")
    return parser.parse_args()


def build_folds_from_split_json(all_files, split_json_path: str):
    """Build folds using a precomputed split.json.

    Expected format:
        {
          "0": {"val_subjects_sorted": [...], ...},
          "1": {"val_subjects_sorted": [...], ...},
          ...
        }
    """
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


# ==============================================================================
# EDAIN-style Adaptive Robust Min-Max Normalization Layer
# ==============================================================================

class AdaptiveRobustMinMaxNormalization(nn.Module):
    """
    EDAIN-style adaptive normalization layer using robust min-max instead of z-score.

    Implements three sublayers in sequence:
    1. Adaptive Outlier Mitigation: smoothed winsorization via tanh
    2. Adaptive Shift: learnable shift parameter (based on robust min)
    3. Adaptive Scale: learnable scale parameter (based on robust range)

    Uses local-aware mode: normalization depends on per-image foreground statistics.

    Normalization formula:
        x' = (x - m * robust_min) / (s * (robust_max - robust_min))

    Where:
        - robust_min = 2nd percentile of foreground voxels
        - robust_max = 98th percentile of foreground voxels
        - m: learnable shift multiplier
        - s: learnable scale multiplier

    Parameters:
        alpha: [0, 1] - ratio of winsorization (0=no winsorization, 1=full)
        beta: [beta_min, inf) - range parameter for tanh compression
        m: R - shift multiplier
        s: (0, inf) - scale multiplier

    Forward expects:
        x: [B, C, D, H, W] - input image tensor
        fg_min: [B, C] or [B, 1] - foreground robust min (2nd percentile) per sample
        fg_max: [B, C] or [B, 1] - foreground robust max (98th percentile) per sample
    """

    def __init__(self, beta_min=1.0, init_alpha=0.5, init_beta=5.0, init_m=1.0, init_s=1.0):
        super().__init__()

        self.beta_min = beta_min

        # Learnable parameters (scalars for single-channel images)
        # alpha: use sigmoid to constrain to [0, 1]
        self._alpha_raw = nn.Parameter(torch.tensor(self._inverse_sigmoid(init_alpha)))

        # beta: use softplus to constrain to [beta_min, inf)
        self._beta_raw = nn.Parameter(torch.tensor(self._inverse_softplus(init_beta - beta_min)))

        # m: unconstrained, can be any real number
        self.m = nn.Parameter(torch.tensor(init_m))

        # s: use softplus to constrain to (0, inf)
        self._s_raw = nn.Parameter(torch.tensor(self._inverse_softplus(init_s)))

    @staticmethod
    def _inverse_sigmoid(y):
        """Inverse of sigmoid: logit function"""
        y = np.clip(y, 1e-6, 1 - 1e-6)
        return np.log(y / (1 - y))

    @staticmethod
    def _inverse_softplus(y):
        """Inverse of softplus"""
        y = max(y, 1e-6)
        if y > 20:
            return y
        return np.log(np.exp(y) - 1)

    @property
    def alpha(self):
        """Get constrained alpha in [0, 1]"""
        return torch.sigmoid(self._alpha_raw)

    @property
    def beta(self):
        """Get constrained beta in [beta_min, inf)"""
        return F.softplus(self._beta_raw) + self.beta_min

    @property
    def s(self):
        """Get constrained s in (0, inf)"""
        return F.softplus(self._s_raw) + 1e-6

    def forward(self, x, fg_min, fg_max, return_debug_info=False):
        """
        Apply adaptive robust min-max normalization.

        Args:
            x: [B, C, D, H, W] input tensor
            fg_min: [B_stats, C] or [B_stats, 1] foreground robust min (2nd percentile) per sample
            fg_max: [B_stats, C] or [B_stats, 1] foreground robust max (98th percentile) per sample
            return_debug_info: if True, return dict with intermediate values for debugging

        Returns:
            Normalized tensor [B, C, D, H, W], or (tensor, debug_dict) if return_debug_info=True
        """
        B, C = x.shape[:2]

        # Handle dimension and batch size mismatch
        if fg_min.dim() == 1:
            fg_min = fg_min.unsqueeze(0)
            fg_max = fg_max.unsqueeze(0)

        B_stats = fg_min.shape[0]
        C_stats = fg_min.shape[1]

        # Reshape to [B_stats, C, 1, 1, 1] for broadcasting
        fg_min = fg_min.view(B_stats, C_stats, 1, 1, 1)
        fg_max = fg_max.view(B_stats, C_stats, 1, 1, 1)

        # If batch sizes don't match (sliding window inference scenario),
        # expand statistics to match input batch size
        if B_stats != B:
            fg_min = fg_min.expand(B, C, 1, 1, 1)
            fg_max = fg_max.expand(B, C, 1, 1, 1)

        # Compute robust range, ensure it's not zero
        fg_range = fg_max - fg_min
        fg_range = torch.clamp(fg_range, min=1e-6)

        # Get constrained parameters
        alpha = self.alpha
        beta = self.beta
        m = self.m
        s = self.s

        # ======================
        # 1. Adaptive Outlier Mitigation (smoothed winsorization)
        # ======================
        # For robust min-max, we center around the midpoint of the range
        fg_center = (fg_min + fg_max) / 2.0
        x_centered = x - fg_center
        x_winsorized = beta * torch.tanh(x_centered / beta) + fg_center
        x_om = alpha * x_winsorized + (1 - alpha) * x

        # ======================
        # 2. Adaptive Shift (using robust min)
        # ======================
        # h2(x) = x - (m * fg_min)
        x_shifted = x_om - (m * fg_min)

        # ======================
        # 3. Adaptive Scale (using robust range)
        # ======================
        # h3(x) = x / (s * (fg_max - fg_min))
        x_scaled = x_shifted / (s * fg_range)

        if return_debug_info:
            debug_info = {
                'x_shifted_mean': x_shifted.mean().item(),
                'x_shifted_std': x_shifted.std().item(),
                'x_scaled_mean': x_scaled.mean().item(),
                'x_scaled_std': x_scaled.std().item(),
                'm_times_fg_min': (m * fg_min).mean().item(),
                's_times_fg_range': (s * fg_range).mean().item(),
            }
            return x_scaled, debug_info

        return x_scaled

    def get_params_info(self):
        """Return current parameter values for logging"""
        return {
            'alpha': self.alpha.item(),
            'beta': self.beta.item(),
            'm': self.m.item(),
            's': self.s.item()
        }


class SegmentationModelWithAdaptiveNorm(nn.Module):
    """
    Wrapper that combines adaptive normalization with a segmentation backbone.
    """

    def __init__(self, backbone, adaptive_norm_layer):
        super().__init__()
        self.adaptive_norm = adaptive_norm_layer
        self.backbone = backbone

    def forward(self, x, fg_min=None, fg_max=None):
        """
        Forward pass with optional adaptive normalization.

        If fg_min and fg_max are provided, apply adaptive normalization first.
        Otherwise, pass input directly to backbone (for compatibility).
        """
        if fg_min is not None and fg_max is not None:
            x = self.adaptive_norm(x, fg_min, fg_max)

        return self.backbone(x)


# ==============================================================================
# Custom Transform to Compute Robust Min-Max Statistics
# ==============================================================================

class ComputeRobustMinMaxStatsd(MapTransform):
    """
    Compute robust min-max statistics (2nd and 98th percentiles) using Otsu thresholding
    and bounding box cropping.

    Process:
        1. Use Otsu threshold to identify foreground region
        2. Compute bounding box around foreground
        3. Crop image to bounding box
        4. Compute percentiles on ALL voxels in cropped region (complete intensity info)

    This transform should be applied BEFORE patch extraction so that
    statistics are computed on the whole image foreground.

    Stores results in data dict as:
        - {key}_fg_min: foreground robust min (2nd percentile of cropped region)
        - {key}_fg_max: foreground robust max (98th percentile of cropped region)
    """

    def __init__(self, keys, lower_percentile=2, upper_percentile=98, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            img = d[key]

            # Handle torch tensor or numpy array
            if torch.is_tensor(img):
                img_np = img.cpu().numpy()
            else:
                img_np = np.asarray(img)

            # img_np shape: [C, D, H, W] or [D, H, W]
            if img_np.ndim == 4:
                # Multi-channel: compute per channel
                fg_mins = []
                fg_maxs = []
                for c in range(img_np.shape[0]):
                    channel_data = img_np[c]
                    fg_min, fg_max = self._compute_robust_minmax(channel_data)
                    fg_mins.append(fg_min)
                    fg_maxs.append(fg_max)
                d[f"{key}_fg_min"] = np.array(fg_mins, dtype=np.float32)
                d[f"{key}_fg_max"] = np.array(fg_maxs, dtype=np.float32)
            else:
                # Single channel
                fg_min, fg_max = self._compute_robust_minmax(img_np)
                d[f"{key}_fg_min"] = np.array([fg_min], dtype=np.float32)
                d[f"{key}_fg_max"] = np.array([fg_max], dtype=np.float32)

        return d

    def _compute_robust_minmax(self, volume):
        """
        Compute robust min and max using Otsu threshold for foreground detection,
        then bounding box cropping, and finally percentiles on the cropped region.

        Process:
            1. Use Otsu threshold to identify foreground
            2. Compute bounding box of foreground region
            3. Crop volume to bounding box
            4. Compute percentiles on ALL voxels in the cropped region (not just >threshold)

        Args:
            volume: 3D numpy array [D, H, W]

        Returns:
            fg_min (2nd percentile), fg_max (98th percentile)
        """
        volume = volume.astype(np.float32)

        # Compute Otsu threshold
        try:
            thr = threshold_otsu(volume)
        except:
            thr = volume.mean()

        # Create foreground mask
        fg_mask = volume > thr

        # Check if any foreground exists
        if not fg_mask.any():
            # No foreground found, use whole image stats
            fg_min = np.percentile(volume, self.lower_percentile)
            fg_max = np.percentile(volume, self.upper_percentile)
        else:
            # Compute bounding box of foreground region
            # Find indices where foreground exists along each axis
            nonzero_coords = np.where(fg_mask)

            # Get bounding box coordinates (min and max for each dimension)
            d_min, d_max = nonzero_coords[0].min(), nonzero_coords[0].max() + 1
            h_min, h_max = nonzero_coords[1].min(), nonzero_coords[1].max() + 1
            w_min, w_max = nonzero_coords[2].min(), nonzero_coords[2].max() + 1

            # Crop volume to bounding box
            cropped_volume = volume[d_min:d_max, h_min:h_max, w_min:w_max]

            # Compute percentiles on the ENTIRE cropped region (all voxels, not just >threshold)
            fg_min = np.percentile(cropped_volume, self.lower_percentile)
            fg_max = np.percentile(cropped_volume, self.upper_percentile)

        # Ensure min < max
        if fg_max - fg_min < 1e-6:
            fg_min = volume.min()
            fg_max = volume.max()
            if fg_max - fg_min < 1e-6:
                fg_max = fg_min + 1.0

        return float(fg_min), float(fg_max)


# ==============================================================================
# Helper Functions
# ==============================================================================

def _dice_from_labels(pred, target, eps: float = 1e-6):
    """计算 Batch 的平均 Dice Score (Foreground 1)"""
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
    weights = np.array([1 / (2 ** i) for i in range(num_stages)])
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


def otsu_select_fn(x):
    """用于 CropForegroundd 的 foreground 检测"""
    x = x.astype(np.float32)
    thr = threshold_otsu(x)
    return x > thr


class NnUNetLikeRandFlipd(MapTransform, RandomizableTransform):
    """nnU-Net style random mirroring"""

    def __init__(self, keys, prob=0.5, spatial_axes=(0, 1, 2)):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.spatial_axes = spatial_axes
        self._axes_to_flip = []
        self._do_transform = False

    def randomize(self, data=None):
        self._do_transform = self.R.random() < self.prob
        self._axes_to_flip = []
        if self._do_transform:
            for ax in self.spatial_axes:
                if self.R.random() < 0.5:
                    self._axes_to_flip.append(ax)

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)

        if not self._axes_to_flip:
            return d

        flip_axes_np = [ax + 1 for ax in self._axes_to_flip]

        for key in self.keys:
            arr = d[key]
            if torch.is_tensor(arr):
                d[key] = torch.flip(arr, dims=flip_axes_np)
            else:
                d[key] = np.flip(arr, axis=flip_axes_np).copy()

        return d


# ==============================================================================
# Optimizer Setup with Separate Learning Rates
# ==============================================================================

def create_optimizer_with_adaptive_lr(model, base_lr, adaptive_lr_multipliers=None):
    """
    Create optimizer with separate learning rates for adaptive normalization parameters.
    """
    if adaptive_lr_multipliers is None:
        adaptive_lr_multipliers = {
            'alpha': 10.0,
            'beta': 10.0,
            'm': 1.0,
            's': 1.0,
        }

    param_groups = []

    # Adaptive normalization parameters
    adaptive_params = {
        'alpha': model.adaptive_norm._alpha_raw,
        'beta': model.adaptive_norm._beta_raw,
        'm': model.adaptive_norm.m,
        's': model.adaptive_norm._s_raw,
    }

    for name, param in adaptive_params.items():
        multiplier = adaptive_lr_multipliers.get(name, 1.0)
        param_groups.append({
            'params': [param],
            'lr': base_lr * multiplier,
            'name': f'adaptive_{name}'
        })

    # Backbone parameters
    backbone_params = list(model.backbone.parameters())
    param_groups.append({
        'params': backbone_params,
        'lr': base_lr,
        'name': 'backbone'
    })

    optimizer = torch.optim.SGD(
        param_groups,
        lr=base_lr,
        momentum=0.99,
        nesterov=True,
        weight_decay=1e-5
    )

    return optimizer


def update_lr_with_scheduler(optimizer, epoch, max_epochs, base_lr, adaptive_lr_multipliers):
    """Update learning rates for all parameter groups using polynomial decay."""
    decay_factor = poly_lr(epoch, max_epochs)

    for pg in optimizer.param_groups:
        name = pg.get('name', '')

        if name.startswith('adaptive_'):
            param_name = name.replace('adaptive_', '')
            multiplier = adaptive_lr_multipliers.get(param_name, 1.0)
            pg['lr'] = base_lr * multiplier * decay_factor
        else:
            pg['lr'] = base_lr * decay_factor


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    args = parse_args()
    set_determinism(seed=2025)

    # -------------------------------------------------------------------------
    # 0. Configuration & Global Paths
    # -------------------------------------------------------------------------
    base_dir = args.base_dir
    global_output_dir = args.out_dir
    os.makedirs(global_output_dir, exist_ok=True)

    # Adaptive Learning Rate Multipliers
    # NOTE: m and s increased to 10x to investigate why m doesn't change
    adaptive_lr_multipliers = {
        'alpha': 10.0,  # Outlier mitigation: 10x
        'beta': 10.0,   # Outlier mitigation: 10x
        'm': 10.0,      # Shift: 10x (increased from 1x for debugging)
        's': 10.0,      # Scale: 10x (increased from 1x for debugging)
    }

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. Dataset Fingerprint Calculation
    # -------------------------------------------------------------------------
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

    # Initial Patch Size Estimation
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

    # Determine Network Topology
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

    print(f"[INFO] Target spacing: {target_pixdim}")
    print(f"[INFO] Patch size: {patch_size}")
    print(f"[INFO] Network channels: {channels}")
    print(f"[INFO] Network strides: {strides}")

    # Configure Kernel Sizes
    num_stages = len(channels)
    kernel_sizes = []
    for stage in range(num_stages):
        if anisotropic and stage < 2:
            kernel_sizes.append((3, 3, 1))
        else:
            kernel_sizes.append((3, 3, 3))

    up_kernel_sizes = tuple(strides)
    dynunet_strides = tuple([(1, 1, 1)] + list(strides))

    # -------------------------------------------------------------------------
    # 3. Data Transforms
    # -------------------------------------------------------------------------
    # Key modification: Using ComputeRobustMinMaxStatsd instead of ComputeForegroundStatsd
    # Computes 2nd and 98th percentiles instead of mean and std

    num_patches_per_image = 4

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim, mode=(3, "nearest")),

        # 先进行前景裁剪
        CropForegroundd(keys=["image", "label"], source_key="image",
                        select_fn=otsu_select_fn, margin=(10, 10, 3)),

        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0,
            clip=True,
            channel_wise=True
        ),


        # 计算robust min-max统计量（2%和98%百分位数）
        ComputeRobustMinMaxStatsd(keys=["image"], lower_percentile=2, upper_percentile=98),

        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=patch_size, pos=1, neg=1,
            num_samples=num_patches_per_image,
        ),
        RandAffined(
            keys=["image", "label"], prob=0.2,
            rotate_range=[
                (0, 0),
                (0, 0),
                (-np.deg2rad(180), np.deg2rad(180))
            ] if anisotropic else [
                (-np.deg2rad(30), np.deg2rad(30)),
                (-np.deg2rad(30), np.deg2rad(30)),
                (-np.deg2rad(30), np.deg2rad(30))
            ],
            scale_range=[(-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4)],
            mode=(3, "nearest"), padding_mode="border"
        ),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=["image"], prob=0.2,
                            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
        RandScaleIntensityd(keys=["image"], factors=(-0.35, 0.5), prob=0.15),
        RandSimulateLowResolutiond(keys=["image"], prob=0.25,
                                   zoom_range=(0.5, 1.0), upsample_mode="trilinear"),
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5)),
        NnUNetLikeRandFlipd(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1, 2)),
        EnsureTyped(keys=["image"], dtype=np.float32),
        EnsureTyped(keys=["label"], dtype=np.uint8),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim, mode=(3, "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image",
                        select_fn=otsu_select_fn, margin=(10, 10, 3)),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0,
            clip=True,
            channel_wise=True
        ),


        # 验证集同样计算robust min-max统计量
        ComputeRobustMinMaxStatsd(keys=["image"], lower_percentile=2, upper_percentile=98),

        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ToTensord(keys=["image", "label"]),
    ])

    post_process = KeepLargestConnectedComponent(applied_labels=[1])

    # -------------------------------------------------------------------------
    # 4. Cross-Validation Loop
    # -------------------------------------------------------------------------
    if args.split_json is not None:
        folds = build_folds_from_split_json(all_files, args.split_json)
    else:
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

    # Training Configuration
    pin_memory = torch.cuda.is_available()
    num_workers = 4
    max_epochs = 1000
    val_interval = 2
    early_stop_patience = 50
    base_lr = 0.01

    for f in folds:
        CURR_FOLD = f["fold"]
        print(f"\n{'=' * 60}")
        print(f"Running Fold {CURR_FOLD} with Adaptive Robust Min-Max Normalization")
        print(f"{'=' * 60}")

        # Datasets
        train_ds = CacheDataset(data=f["train_files"], transform=train_transforms,
                                cache_rate=0, num_workers=num_workers)
        val_ds = CacheDataset(data=f["val_files"], transform=val_transforms,
                              cache_rate=10, num_workers=num_workers)

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

        # Create backbone model
        backbone = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            filters=channels, strides=dynunet_strides,
            kernel_size=kernel_sizes, upsample_kernel_size=up_kernel_sizes,
            act_name=("leakyrelu", {"negative_slope": 0.01}),
            norm_name="instance",
            deep_supervision=True, deep_supr_num=3
        )

        # Create Adaptive Robust Min-Max Normalization layer
        adaptive_norm_layer = AdaptiveRobustMinMaxNormalization(
            beta_min=1.0,
            init_alpha=0.5,
            init_beta=5.0,
            init_m=1.0,
            init_s=1.0,
        )

        # Combine into full model
        model = SegmentationModelWithAdaptiveNorm(backbone, adaptive_norm_layer).to(device)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        adaptive_params = sum(p.numel() for p in model.adaptive_norm.parameters() if p.requires_grad)
        print(f"[INFO] Total Trainable Parameters: {total_params / 1e6:.2f} M ({total_params})")
        print(f"[INFO] Adaptive Norm Parameters: {adaptive_params} (alpha, beta, m, s)")
        print(f"[INFO] Adaptive LR Multipliers: {adaptive_lr_multipliers}")

        # === DEBUG: Print potential issues with m parameter ===
        print("\n" + "="*70)
        print("[DEBUG] POTENTIAL REASONS WHY 'm' MIGHT NOT CHANGE:")
        print("="*70)
        print("1. If fg_min (2% percentile of cropped region) ≈ 0:")
        print("   -> The shift term (m * fg_min) ≈ 0 regardless of m value")
        print("   -> Gradient w.r.t. m = ∂L/∂x * (-fg_min) ≈ 0")
        print("")
        print("2. If fg_range = fg_max - fg_min is very large:")
        print("   -> Small changes in shift are negligible after scaling")
        print("")
        print("3. Check the DEBUG output each epoch for:")
        print("   - fg_min_mean, fg_min_min: Should NOT be close to 0")
        print("   - m_grad_mean, m_grad_abs_max: Should NOT be 0")
        print("="*70 + "\n")

        # Loss Function
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True,
                             include_background=False, lambda_dice=0.5, lambda_ce=0.5)

        # Optimizer with separate learning rates
        optimizer = create_optimizer_with_adaptive_lr(model, base_lr, adaptive_lr_multipliers)

        scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        # Initialize Log File (with extended debug columns)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as fcsv:
                csv.writer(fcsv).writerow([
                    "epoch", "train_loss", "train_dice", "val_loss", "val_dice",
                    "lr", "time_sec", "alpha", "beta", "m", "s",
                    # DEBUG columns for fg_min analysis
                    "fg_min_mean", "fg_min_min", "fg_min_max", "fg_max_mean",
                    "fg_range_mean", "fg_range_min",
                    # DEBUG columns for gradient analysis
                    "m_grad_mean", "m_grad_abs_max", "s_grad_mean"
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

            # === DEBUG: Accumulate fg_min/fg_max statistics across batches ===
            epoch_fg_min_values = []
            epoch_fg_max_values = []
            epoch_fg_range_values = []
            epoch_m_grad_values = []
            epoch_s_grad_values = []

            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).long()

                # 获取预计算的robust min-max统计量
                fg_min = batch["image_fg_min"].to(device)
                fg_max = batch["image_fg_max"].to(device)

                # === DEBUG: Collect fg_min/fg_max statistics ===
                epoch_fg_min_values.append(fg_min.detach().cpu().numpy())
                epoch_fg_max_values.append(fg_max.detach().cpu().numpy())
                epoch_fg_range_values.append((fg_max - fg_min).detach().cpu().numpy())

                optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    logits = model(images, fg_min=fg_min, fg_max=fg_max)
                    logits = normalize_logits_output(logits)

                    # Deep Supervision Loss
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

                    # Monitor Train Dice
                    train_pred = logits[0].argmax(dim=1, keepdim=True)
                    train_dice_val = _dice_from_labels(train_pred, labels)
                    train_dice_accum += train_dice_val.item()

                scaler.scale(loss).backward()

                # === DEBUG: Capture gradients before optimizer step ===
                if model.adaptive_norm.m.grad is not None:
                    epoch_m_grad_values.append(model.adaptive_norm.m.grad.item())
                if model.adaptive_norm._s_raw.grad is not None:
                    epoch_s_grad_values.append(model.adaptive_norm._s_raw.grad.item())

                scaler.step(optimizer)
                scaler.update()
                train_loss_accum += loss.item()
                num_batches += 1

            epoch_loss = train_loss_accum / num_batches
            epoch_train_dice = train_dice_accum / num_batches

            # === DEBUG: Compute epoch-level fg_min/fg_max statistics ===
            all_fg_min = np.concatenate(epoch_fg_min_values)
            all_fg_max = np.concatenate(epoch_fg_max_values)
            all_fg_range = np.concatenate(epoch_fg_range_values)

            fg_min_mean = float(all_fg_min.mean())
            fg_min_min = float(all_fg_min.min())
            fg_min_max = float(all_fg_min.max())
            fg_max_mean = float(all_fg_max.mean())
            fg_range_mean = float(all_fg_range.mean())
            fg_range_min = float(all_fg_range.min())

            # Gradient statistics
            m_grad_mean = float(np.mean(epoch_m_grad_values)) if epoch_m_grad_values else 0.0
            m_grad_abs_max = float(np.max(np.abs(epoch_m_grad_values))) if epoch_m_grad_values else 0.0
            s_grad_mean = float(np.mean(epoch_s_grad_values)) if epoch_s_grad_values else 0.0

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
                        val_fg_min = val_data["image_fg_min"].to(device)
                        val_fg_max = val_data["image_fg_max"].to(device)


                        # 自定义predictor函数
                        def predictor(x):
                            return model(x, fg_min=val_fg_min, fg_max=val_fg_max)


                        val_logits = sliding_window_inference(
                            inputs=val_images, roi_size=patch_size, sw_batch_size=4,
                            predictor=predictor, overlap=0.5, mode="gaussian",
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
                        "adaptive_params": model.adaptive_norm.get_params_info()
                    }, ckpt_path)
                    print(f"   >>> New Best: Val_Dice={best_metric:.4f} | Train_Dice={epoch_train_dice:.4f}")
                else:
                    epochs_no_improve += val_interval

            # Update learning rates
            update_lr_with_scheduler(optimizer, epoch, max_epochs, base_lr, adaptive_lr_multipliers)

            elapsed = time.time() - t0

            # Get adaptive parameters for logging
            adaptive_info = model.adaptive_norm.get_params_info()

            # Print Progress (main line)
            log_str = f"Ep {epoch:03d} | Loss={epoch_loss:.4f} | Train_Dice={epoch_train_dice:.4f}"
            if val_dice_epoch is not None:
                log_str += f" | Val_Loss={val_loss_epoch:.4f} | Val_Dice={val_dice_epoch:.4f} | Best={best_metric:.4f}"
            log_str += f" | LR={get_lr(optimizer):.6f} | Time={elapsed:.1f}s"
            log_str += f" | α={adaptive_info['alpha']:.3f} β={adaptive_info['beta']:.2f} m={adaptive_info['m']:.4f} s={adaptive_info['s']:.4f}"
            print(log_str)

            # === DEBUG: Print fg_min statistics and gradient info ===
            debug_str = f"       [DEBUG] fg_min: mean={fg_min_mean:.2f}, min={fg_min_min:.2f}, max={fg_min_max:.2f}"
            debug_str += f" | fg_range: mean={fg_range_mean:.2f}, min={fg_range_min:.2f}"
            debug_str += f" | m_grad: mean={m_grad_mean:.2e}, |max|={m_grad_abs_max:.2e}"
            debug_str += f" | s_grad: mean={s_grad_mean:.2e}"
            print(debug_str)

            # === DEBUG: Additional diagnostic info every 10 epochs ===
            if epoch % 10 == 1:
                print(f"       [DIAG] m*fg_min contribution: {adaptive_info['m'] * fg_min_mean:.4f}")
                print(f"       [DIAG] s*fg_range contribution: {adaptive_info['s'] * fg_range_mean:.4f}")
                print(f"       [DIAG] If fg_min ≈ 0, then m has no effect on shift!")
                print(f"       [DIAG] Current LR for m: {base_lr * adaptive_lr_multipliers['m'] * poly_lr(epoch, max_epochs):.6f}")

            # Log to CSV (with extended debug columns)
            with open(log_path, "a", newline="") as fcsv:
                csv.writer(fcsv).writerow([
                    epoch,
                    f"{epoch_loss:.6f}", f"{epoch_train_dice:.6f}",
                    ("" if val_loss_epoch is None else f"{val_loss_epoch:.6f}"),
                    ("" if val_dice_epoch is None else f"{val_dice_epoch:.6f}"),
                    f"{get_lr(optimizer):.6f}", f"{elapsed:.2f}",
                    f"{adaptive_info['alpha']:.6f}", f"{adaptive_info['beta']:.6f}",
                    f"{adaptive_info['m']:.6f}", f"{adaptive_info['s']:.6f}",
                    # DEBUG columns
                    f"{fg_min_mean:.6f}", f"{fg_min_min:.6f}", f"{fg_min_max:.6f}", f"{fg_max_mean:.6f}",
                    f"{fg_range_mean:.6f}", f"{fg_range_min:.6f}",
                    f"{m_grad_mean:.6e}", f"{m_grad_abs_max:.6e}", f"{s_grad_mean:.6e}"
                ])

            # Early Stopping
            if epochs_no_improve >= early_stop_patience:
                print(f"\n[Early Stopping] No improvement for {early_stop_patience} epochs.")
                break

        print(f"\nFold {CURR_FOLD} Complete | Best Epoch: {best_epoch} | Best Val Dice: {best_metric:.4f}")

        # Print final adaptive parameters
        final_adaptive = model.adaptive_norm.get_params_info()
        print(f"Final Adaptive Parameters: α={final_adaptive['alpha']:.4f}, β={final_adaptive['beta']:.4f}, "
              f"m={final_adaptive['m']:.4f}, s={final_adaptive['s']:.4f}")

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
                val_fg_min = val_data["image_fg_min"].to(device)
                val_fg_max = val_data["image_fg_max"].to(device)


                def predictor(x):
                    return model(x, fg_min=val_fg_min, fg_max=val_fg_max)


                val_logits = sliding_window_inference(
                    val_images, patch_size, 4, predictor, overlap=0.5, mode="gaussian"
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