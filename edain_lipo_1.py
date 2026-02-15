"""
EDAIN-style Adaptive Normalization for 3D Medical Image Segmentation

This script integrates learnable adaptive normalization (inspired by EDAIN paper)
into a 3D segmentation network. The normalization includes:
1. Adaptive Outlier Mitigation (smoothed winsorization via tanh)
2. Adaptive Shift
3. Adaptive Scale

Using local-aware mode where statistics are computed per-image on foreground voxels.
"""

import os
import glob
import nibabel as nib
import numpy as np
import re
import csv
import time
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
    RandomizableTransform
)


# ==============================================================================
# EDAIN-style Adaptive Normalization Layer
# ==============================================================================

class EDAINNormalization(nn.Module):
    """
    EDAIN-style adaptive normalization layer for 3D medical images.

    Implements three sublayers in sequence:
    1. Adaptive Outlier Mitigation: smoothed winsorization via tanh
    2. Adaptive Shift: learnable shift parameter
    3. Adaptive Scale: learnable scale parameter

    Uses local-aware mode: normalization depends on per-image foreground statistics.

    Parameters:
        alpha: [0, 1] - ratio of winsorization (0=no winsorization, 1=full)
        beta: [beta_min, inf) - range parameter for tanh compression
        m: R - shift multiplier
        s: (0, inf) - scale multiplier

    Forward expects:
        x: [B, C, D, H, W] - input image tensor
        fg_mean: [B, C] or [B, 1] - foreground mean per sample
        fg_std: [B, C] or [B, 1] - foreground std per sample
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

    def forward(self, x, fg_mean, fg_std):
        """
        Apply EDAIN normalization.

        Args:
            x: [B, C, D, H, W] input tensor
            fg_mean: [B_stats, C] or [B_stats, 1] foreground mean per sample
            fg_std: [B_stats, C] or [B_stats, 1] foreground std per sample

        Note:
            During sliding_window_inference, x may have larger batch size than
            fg_mean/fg_std (multiple windows from same image). In this case,
            we broadcast the statistics to match x's batch size.

        Returns:
            Normalized tensor [B, C, D, H, W]
        """
        B, C = x.shape[:2]

        # Handle dimension and batch size mismatch
        # fg_mean/fg_std might have smaller batch size during sliding window inference
        if fg_mean.dim() == 1:
            # Shape [C] -> [1, C]
            fg_mean = fg_mean.unsqueeze(0)
            fg_std = fg_std.unsqueeze(0)

        B_stats = fg_mean.shape[0]
        C_stats = fg_mean.shape[1]

        # Reshape to [B_stats, C, 1, 1, 1] for broadcasting
        fg_mean = fg_mean.view(B_stats, C_stats, 1, 1, 1)
        fg_std = fg_std.view(B_stats, C_stats, 1, 1, 1)

        # If batch sizes don't match (sliding window inference scenario),
        # expand statistics to match input batch size
        # This assumes all windows in the batch come from the same image
        if B_stats != B:
            fg_mean = fg_mean.expand(B, C, 1, 1, 1)
            fg_std = fg_std.expand(B, C, 1, 1, 1)

        # Ensure fg_std is not zero
        fg_std = torch.clamp(fg_std, min=1e-6)

        # Get constrained parameters
        alpha = self.alpha
        beta = self.beta
        m = self.m
        s = self.s

        # ======================
        # 1. Adaptive Outlier Mitigation (smoothed winsorization)
        # ======================
        # h1(x) = alpha * [beta * tanh((x - mu) / beta) + mu] + (1 - alpha) * x
        # Using local-aware: mu = fg_mean (per-image foreground mean)

        x_centered = x - fg_mean
        x_winsorized = beta * torch.tanh(x_centered / beta) + fg_mean
        x_om = alpha * x_winsorized + (1 - alpha) * x

        # ======================
        # 2. Adaptive Shift
        # ======================
        # h2(x) = x - (m * fg_mean)
        # Local-aware: shift amount depends on per-image mean

        x_shifted = x_om - (m * fg_mean)

        # ======================
        # 3. Adaptive Scale
        # ======================
        # h3(x) = x / (s * fg_std)
        # Local-aware: scale amount depends on per-image std

        x_scaled = x_shifted / (s * fg_std)

        return x_scaled

    def get_params_info(self):
        """Return current parameter values for logging"""
        return {
            'alpha': self.alpha.item(),
            'beta': self.beta.item(),
            'm': self.m.item(),
            's': self.s.item()
        }


class SegmentationModelWithEDAIN(nn.Module):
    """
    Wrapper that combines EDAIN normalization with a segmentation backbone.
    """

    def __init__(self, backbone, edain_layer):
        super().__init__()
        self.edain = edain_layer
        self.backbone = backbone

    def forward(self, x, fg_mean=None, fg_std=None):
        """
        Forward pass with optional EDAIN normalization.

        If fg_mean and fg_std are provided, apply EDAIN normalization first.
        Otherwise, pass input directly to backbone (for compatibility).
        """
        if fg_mean is not None and fg_std is not None:
            x = self.edain(x, fg_mean, fg_std)

        return self.backbone(x)


# ==============================================================================
# Custom Transform to Compute Foreground Statistics
# ==============================================================================

class ComputeForegroundStatsd(MapTransform):
    """
    Compute foreground statistics (mean, std) using Otsu thresholding.

    This transform should be applied BEFORE patch extraction so that
    statistics are computed on the whole image foreground.

    Stores results in data dict as:
        - {key}_fg_mean: foreground mean
        - {key}_fg_std: foreground standard deviation
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

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
                fg_means = []
                fg_stds = []
                for c in range(img_np.shape[0]):
                    channel_data = img_np[c]
                    fg_mean, fg_std = self._compute_fg_stats(channel_data)
                    fg_means.append(fg_mean)
                    fg_stds.append(fg_std)
                d[f"{key}_fg_mean"] = np.array(fg_means, dtype=np.float32)
                d[f"{key}_fg_std"] = np.array(fg_stds, dtype=np.float32)
            else:
                # Single channel
                fg_mean, fg_std = self._compute_fg_stats(img_np)
                d[f"{key}_fg_mean"] = np.array([fg_mean], dtype=np.float32)
                d[f"{key}_fg_std"] = np.array([fg_std], dtype=np.float32)

        return d

    def _compute_fg_stats(self, volume):
        """
        Compute foreground mean and std using Otsu threshold.

        Args:
            volume: 3D numpy array [D, H, W]

        Returns:
            fg_mean, fg_std
        """
        volume = volume.astype(np.float32)

        # Compute Otsu threshold
        try:
            thr = threshold_otsu(volume)
        except:
            # If Otsu fails (e.g., uniform image), use simple thresholding
            thr = volume.mean()

        # Create foreground mask
        fg_mask = volume > thr

        # Get foreground voxels
        fg_voxels = volume[fg_mask]

        if len(fg_voxels) == 0:
            # No foreground found, use whole image stats
            fg_mean = volume.mean()
            fg_std = volume.std()
        else:
            fg_mean = fg_voxels.mean()
            fg_std = fg_voxels.std()

        # Ensure std is not zero
        if fg_std < 1e-6:
            fg_std = 1.0

        return float(fg_mean), float(fg_std)


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

def create_optimizer_with_edain_lr(model, base_lr, edain_lr_multipliers=None):
    """
    Create optimizer with separate learning rates for EDAIN parameters.

    Args:
        model: SegmentationModelWithEDAIN instance
        base_lr: base learning rate for backbone
        edain_lr_multipliers: dict with keys 'alpha', 'beta', 'm', 's'
                              and values as multipliers relative to base_lr
                              Default: {'alpha': 10, 'beta': 10, 'm': 1, 's': 1}

    Returns:
        optimizer with parameter groups
    """
    if edain_lr_multipliers is None:
        edain_lr_multipliers = {
            'alpha': 10.0,  # Outlier mitigation: 10x base LR
            'beta': 10.0,   # Outlier mitigation: 10x base LR
            'm': 1.0,       # Shift: same as base LR
            's': 1.0,       # Scale: same as base LR
        }

    # Separate parameters into groups
    param_groups = []

    # EDAIN parameters
    edain_params = {
        'alpha': model.edain._alpha_raw,
        'beta': model.edain._beta_raw,
        'm': model.edain.m,
        's': model.edain._s_raw,
    }

    for name, param in edain_params.items():
        multiplier = edain_lr_multipliers.get(name, 1.0)
        param_groups.append({
            'params': [param],
            'lr': base_lr * multiplier,
            'name': f'edain_{name}'
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


def update_lr_with_scheduler(optimizer, epoch, max_epochs, base_lr, edain_lr_multipliers):
    """
    Update learning rates for all parameter groups using polynomial decay.
    """
    decay_factor = poly_lr(epoch, max_epochs)

    for pg in optimizer.param_groups:
        name = pg.get('name', '')

        if name.startswith('edain_'):
            param_name = name.replace('edain_', '')
            multiplier = edain_lr_multipliers.get(param_name, 1.0)
            pg['lr'] = base_lr * multiplier * decay_factor
        else:
            pg['lr'] = base_lr * decay_factor


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    set_determinism(seed=2025)

    # -------------------------------------------------------------------------
    # 0. Configuration & Global Paths
    # -------------------------------------------------------------------------
    base_dir = "../dataset/LIPO"
    global_output_dir = "../outputs/edain_lipo_experiment"
    os.makedirs(global_output_dir, exist_ok=True)

    global_dice_csv = os.path.join(global_output_dir, "final_dice.csv")
    if not os.path.exists(global_dice_csv):
        with open(global_dice_csv, "w", newline="") as f:
            csv.writer(f).writerow(["SubjectID", "Fold", "Dice", "PredictionPath"])

    # EDAIN Learning Rate Multipliers
    # 根据EDAIN论文，异常值缓解层需要更大的学习率
    edain_lr_multipliers = {
        'alpha': 10.0,  # 异常值缓解: 10x
        'beta': 10.0,   # 异常值缓解: 10x
        'm': 1.0,       # 平移: 1x (与主网络相同)
        's': 1.0,       # 缩放: 1x (与主网络相同)
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
    # 关键修改：移除了 ScaleIntensityRangePercentilesd 和 NormalizeIntensityd
    # 改为使用 ComputeForegroundStatsd 计算前景统计量
    # 归一化将在模型内部通过 EDAIN 层进行

    num_patches_per_image = 4

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim, mode=(3, "nearest")),

        # 先进行前景裁剪
        CropForegroundd(keys=["image", "label"], source_key="image",
                       select_fn=otsu_select_fn, margin=(10, 10, 3)),

        # 在patch提取之前，计算整个前景的统计量
        # 这些统计量将被传递给EDAIN层使用
        ComputeForegroundStatsd(keys=["image"]),

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

        # 验证集同样计算前景统计量
        ComputeForegroundStatsd(keys=["image"]),

        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ToTensord(keys=["image", "label"]),
    ])

    post_process = KeepLargestConnectedComponent(applied_labels=[1])

    # -------------------------------------------------------------------------
    # 4. Cross-Validation Loop
    # -------------------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    folds = []
    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(indices, groups=subjects)):
        train_files = [all_files[i] for i in tr_idx]
        val_files = [all_files[i] for i in va_idx]
        folds.append({"fold": fold_id, "train_files": train_files, "val_files": val_files})

    # Training Configuration
    pin_memory = torch.cuda.is_available()
    num_workers = 4
    max_epochs = 1000
    val_interval = 2
    early_stop_patience = 50
    base_lr = 0.01

    for CURR_FOLD, f in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"Running Fold {CURR_FOLD} with EDAIN Normalization")
        print(f"{'=' * 60}")

        # Datasets
        train_ds = CacheDataset(data=f["train_files"], transform=train_transforms,
                                cache_rate=0.1, num_workers=num_workers)
        val_ds = CacheDataset(data=f["val_files"], transform=val_transforms,
                              cache_rate=0.1, num_workers=num_workers)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, collate_fn=list_data_collate,
                                  persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory, persistent_workers=(num_workers > 0))

        out_dir = os.path.join(global_output_dir, f"fold_{CURR_FOLD}")
        os.makedirs(out_dir, exist_ok=True)
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

        # Create EDAIN layer
        edain_layer = EDAINNormalization(
            beta_min=1.0,
            init_alpha=0.5,  # 初始化：50% winsorization
            init_beta=5.0,   # 初始化：较大的压缩范围
            init_m=1.0,      # 初始化：标准平移
            init_s=1.0,      # 初始化：标准缩放
        )

        # Combine into full model
        model = SegmentationModelWithEDAIN(backbone, edain_layer).to(device)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        edain_params = sum(p.numel() for p in model.edain.parameters() if p.requires_grad)
        print(f"[INFO] Total Trainable Parameters: {total_params / 1e6:.2f} M ({total_params})")
        print(f"[INFO] EDAIN Parameters: {edain_params} (alpha, beta, m, s)")
        print(f"[INFO] EDAIN LR Multipliers: {edain_lr_multipliers}")

        # Loss Function (only backbone loss, not modified)
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True,
                             include_background=False, lambda_dice=0.5, lambda_ce=0.5)

        # Optimizer with separate learning rates for EDAIN parameters
        optimizer = create_optimizer_with_edain_lr(
            model, base_lr, edain_lr_multipliers
        )

        scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        # Initialize Log File
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as fcsv:
                csv.writer(fcsv).writerow([
                    "epoch", "train_loss", "train_dice", "val_loss", "val_dice",
                    "lr", "time_sec", "edain_alpha", "edain_beta", "edain_m", "edain_s"
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

                # 获取预计算的前景统计量
                fg_mean = batch["image_fg_mean"].to(device)
                fg_std = batch["image_fg_std"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    # 将统计量传递给模型进行EDAIN归一化
                    logits = model(images, fg_mean=fg_mean, fg_std=fg_std)
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
                        val_fg_mean = val_data["image_fg_mean"].to(device)
                        val_fg_std = val_data["image_fg_std"].to(device)

                        # 自定义predictor函数，传入统计量
                        def predictor(x):
                            return model(x, fg_mean=val_fg_mean, fg_std=val_fg_std)

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
                    # 保存时也保存EDAIN参数
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

            # Get EDAIN parameters for logging
            edain_info = model.edain.get_params_info()

            # Print Progress
            log_str = f"Ep {epoch:03d} | Loss={epoch_loss:.4f} | Train_Dice={epoch_train_dice:.4f}"
            if val_dice_epoch is not None:
                log_str += f" | Val_Loss={val_loss_epoch:.4f} | Val_Dice={val_dice_epoch:.4f} | Best={best_metric:.4f}"
            log_str += f" | LR={get_lr(optimizer):.6f} | Time={elapsed:.1f}s"
            log_str += f" | α={edain_info['alpha']:.3f} β={edain_info['beta']:.2f} m={edain_info['m']:.3f} s={edain_info['s']:.3f}"
            print(log_str)

            # Log to CSV
            with open(log_path, "a", newline="") as fcsv:
                csv.writer(fcsv).writerow([
                    epoch,
                    f"{epoch_loss:.6f}", f"{epoch_train_dice:.6f}",
                    ("" if val_loss_epoch is None else f"{val_loss_epoch:.6f}"),
                    ("" if val_dice_epoch is None else f"{val_dice_epoch:.6f}"),
                    f"{get_lr(optimizer):.6f}", f"{elapsed:.2f}",
                    f"{edain_info['alpha']:.6f}", f"{edain_info['beta']:.6f}",
                    f"{edain_info['m']:.6f}", f"{edain_info['s']:.6f}"
                ])

            # Early Stopping
            if epochs_no_improve >= early_stop_patience:
                print(f"\n[Early Stopping] No improvement for {early_stop_patience} epochs.")
                break

        print(f"\nFold {CURR_FOLD} Complete | Best Epoch: {best_epoch} | Best Val Dice: {best_metric:.4f}")

        # Print final EDAIN parameters
        final_edain = model.edain.get_params_info()
        print(f"Final EDAIN Parameters: α={final_edain['alpha']:.4f}, β={final_edain['beta']:.4f}, "
              f"m={final_edain['m']:.4f}, s={final_edain['s']:.4f}")

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
                val_fg_mean = val_data["image_fg_mean"].to(device)
                val_fg_std = val_data["image_fg_std"].to(device)

                def predictor(x):
                    return model(x, fg_mean=val_fg_mean, fg_std=val_fg_std)

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
                with open(global_dice_csv, "a", newline="") as f:
                    csv.writer(f).writerow([subj_id, CURR_FOLD, f"{current_dice:.6f}", save_path])

    print("\n" + "=" * 60)
    print("All folds completed!")
    print("=" * 60)