"""
lipo_obelisk_baseline_v10.py
────────────────────────────
Baseline v10 — raw-axis pipeline, designed as direct comparison for v10 (adaptive).

Design principles
─────────────────
1. Raw NIfTI orientation (no Orientationd(RAS)). NIfTI storage convention puts
   the slice acquisition axis at array axis 2 for all cases regardless of
   axial / coronal / sagittal acquisition orientation.
2. Dataset fingerprint computed from raw NIfTI headers, using the standard
   nnU-Net heuristic: median spacing, anisotropy detection by max/min >= 3,
   10th-percentile slice target on the anisotropic axis.
3. CropForegroundd with **Otsu** threshold (NOT ">0", which is a no-op for MRI:
   dataset analysis showed >0 keeps 81% of voxels on average vs Otsu 20%).
4. ZScore normalization on the **raw image** (Plan A): both baseline and v10
   apply ZScore to raw values, so resampling acts on already-normalized data
   in both pipelines.
5. CPU MONAI Spacingd with cubic (mode=3) for image, nearest for label.
   This is THE single variable that distinguishes baseline from v10
   (v10 replaces this with in-model GPU bilinear grid_sample with learnable α).
6. GPU patch sampler (shared with v10): pos-sampling guarantees the patch
   contains tumor; neg-sampling is random within the resampled volume.
7. GPU augmentation strictly aligned with nnU-Net (joint rotate+scale, gamma 0.3,
   no additive brightness, etc.).
8. OOF: warp predictions back to raw original space via inverse Spacingd
   (using torch.grid_sample with raw spacing), Dice computed against raw label.
"""

import os
import argparse
import glob
import json
import csv
import multiprocessing
import math
import time
import re
from typing import Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast as torch_autocast
from torch import amp
from sklearn.model_selection import GroupKFold
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt, binary_erosion

from monai.data import CacheDataset, DataLoader, list_data_collate, MetaTensor
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, CropForegroundd, Spacingd,
    EnsureTyped, ToTensord, MapTransform, KeepLargestConnectedComponent,
)


# ============================================================================
# Helpers
# ============================================================================

def otsu_select_fn(img):
    """CropForegroundd selection function using Otsu threshold (MRI-appropriate).

    For MRI: '>0' keeps ~81% of voxels (median 91%); Otsu keeps ~20% — the
    actual body region. Verified on the lipo dataset (see test_cropfg_strategies).
    """
    arr = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    vol = arr[0] if arr.ndim == 4 else arr
    try:
        thr = threshold_otsu(vol.astype(np.float32))
    except Exception:
        thr = vol.mean()
    mask = vol > thr
    return mask[np.newaxis] if arr.ndim == 4 else mask


class ZScoreNormalizeForegroundd(MapTransform):
    """Per-image z-score normalization on RAW image, before any resampling.

    fg_mode:
        gt_zero — voxels > 0 (after CropForegroundd this is body region)
        otsu    — voxels > Otsu threshold (use on uncropped full volume)
    """

    def __init__(self, keys, fg_mode='gt_zero', allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.fg_mode = fg_mode

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # Compute normalized values on CPU numpy
            if torch.is_tensor(img):
                img_np = img.detach().cpu().numpy().copy()
            else:
                img_np = np.asarray(img).copy()

            if img_np.ndim == 4:
                for c in range(img_np.shape[0]):
                    m, s = self._stats(img_np[c])
                    img_np[c] = (img_np[c] - m) / s
            else:
                m, s = self._stats(img_np)
                img_np = (img_np - m) / s

            # Write back IN-PLACE to preserve MetaTensor wrapper (and its affine)
            # Critical: MetaTensor is a subclass of Tensor, so checking
            # isinstance(img, MetaTensor) must come BEFORE torch.is_tensor(img),
            # and we use copy_ to avoid replacing the tensor object.
            if torch.is_tensor(img):
                img.copy_(torch.from_numpy(img_np.astype(np.float32)).to(img.device))
                # d[key] already points to img; MetaTensor class + affine preserved
            else:
                d[key] = img_np
        return d

    def _stats(self, vol):
        vol = vol.astype(np.float32)
        if self.fg_mode == 'otsu':
            try:
                thr = threshold_otsu(vol)
            except Exception:
                thr = 0.0
            fg = vol[vol > thr]
        else:
            fg = vol[vol > 0]
        if len(fg) == 0:
            m, s = vol.mean(), vol.std()
        else:
            m, s = float(fg.mean()), float(fg.std())
        if s < 1e-6:
            s = 1.0
        return m, s


class StoreAffined(MapTransform):
    """Store the MetaTensor 4x4 affine as a plain numpy array in data dict.

    Place AFTER all geometric transforms (CropFG, Spacingd) so the captured
    affine maps src_voxel → world and already includes all crop/resample
    translations+scales. Needed for correct warp-back to raw NIfTI space.
    """

    def __init__(self, keys, key_out='src_affine', allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.key_out = key_out

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]
        if isinstance(img, MetaTensor) and hasattr(img, 'affine'):
            aff = img.affine
            if torch.is_tensor(aff):
                aff = aff.detach().cpu().numpy()
            d[self.key_out] = np.asarray(aff, dtype=np.float32).copy()
        else:
            d[self.key_out] = np.eye(4, dtype=np.float32)
        return d


class StoreSpacingd(MapTransform):
    """Read voxel spacing from MetaTensor affine after Spacingd has been applied.

    This stores the RESAMPLED spacing (which equals target_pixdim if Spacingd
    was applied successfully). For OOF, we also need the ORIGINAL raw spacing,
    so we read it before Spacingd and store separately.
    """

    def __init__(self, keys, key_out='spacing', allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.key_out = key_out

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]
        if isinstance(img, MetaTensor) and hasattr(img, 'affine'):
            aff = img.affine
            if torch.is_tensor(aff):
                aff = aff.cpu().numpy()
            sp = np.sqrt((aff[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
        else:
            sp = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        d[self.key_out] = sp
        return d


def _dice_from_labels(pred, target, eps=1e-6):
    if pred.dim() == 5 and pred.size(1) == 1:
        pred = pred[:, 0]
    if target.dim() == 5 and target.size(1) == 1:
        target = target[:, 0]
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    inter = (pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    union = pred_bin.sum(dim=(1, 2, 3)).float() + target_bin.sum(dim=(1, 2, 3)).float()
    return ((2.0 * inter + eps) / (union + eps)).mean()


def extract_subject_id(path):
    base = os.path.basename(path)
    m = re.search(r'(Lipo-\d+)', base)
    return m.group(1) if m else base.split('_')[0]


def poly_lr(epoch, max_epochs):
    return (1 - epoch / max(max_epochs, 1)) ** 0.9


def get_lr(optim):
    for pg in optim.param_groups:
        return pg.get('lr', None)


def get_deep_supervision_weights(n):
    w = np.array([1.0 / (2 ** i) for i in range(n)])
    return w / w.sum()


def normalize_logits_output(logits):
    if isinstance(logits, (list, tuple)):
        return list(logits)
    elif logits.dim() == 6:
        return list(logits.unbind(dim=1))
    return [logits]


# ============================================================================
# nnU-Net-strict GPU Augmentation
# ============================================================================
# Aligned with batchgenerators / nnU-Net default training augmentation:
# - SpatialTransform: prob=0.2 JOINT rotate+scale
#   - rotation: ±π for anisotropic axis only (axis 2 in raw NIfTI for our data)
#   - scale: (0.7, 1.4) per-axis
# - GaussianNoise: prob=0.1, var (0, 0.1)
# - GaussianBlur: prob=0.2, sigma (0.5, 1.5)
# - BrightnessMultiplicative: prob=0.15, (0.75, 1.25)  -- NO additive brightness
# - ContrastAugmentation: prob=0.15, (0.75, 1.25)
# - SimulateLowResolution: prob=0.25, zoom (0.5, 1.0)
# - GammaTransform: prob=0.3, gamma (0.7, 1.5)
# - Mirror: per-axis 0.5 (no outer gate, treat axes independently)

class NNUNetGPUAugmentation(nn.Module):
    def __init__(self,
                 anisotropic=True, anisotropy_axis=2,
                 p_spatial=0.2, rotate_deg=180.0, scale_range=(0.7, 1.4),
                 p_noise=0.1, noise_var=(0.0, 0.1),
                 p_blur=0.2, blur_sigma=(0.5, 1.5),
                 p_bright_mul=0.15, bright_mul=(0.75, 1.25),
                 p_contrast=0.15, contrast=(0.75, 1.25),
                 p_lowres=0.25, lowres_zoom=(0.5, 1.0),
                 p_gamma=0.3, gamma=(0.7, 1.5), p_gamma_invert=0.5,
                 p_flip_per_axis=0.5):
        super().__init__()
        self.anisotropic = anisotropic
        self.anisotropy_axis = anisotropy_axis
        self.p_spatial = p_spatial
        self.rotate_deg = rotate_deg
        self.scale_range = scale_range
        self.p_noise = p_noise
        self.noise_var = noise_var
        self.p_blur = p_blur
        self.blur_sigma = blur_sigma
        self.p_bright_mul = p_bright_mul
        self.bright_mul = bright_mul
        self.p_contrast = p_contrast
        self.contrast = contrast
        self.p_lowres = p_lowres
        self.lowres_zoom = lowres_zoom
        self.p_gamma = p_gamma
        self.gamma = gamma
        self.p_gamma_invert = p_gamma_invert
        self.p_flip_per_axis = p_flip_per_axis

    @torch.no_grad()
    def _rand(self):
        return torch.rand(1).item()

    @torch.no_grad()
    def _uniform(self, lo, hi):
        return lo + (hi - lo) * torch.rand(1).item()

    # ─── Spatial: joint rotate + scale ────────────────────────────────────────
    def spatial_only(self, img, lbl):
        """Joint rotate+scale, prob = self.p_spatial. Both happen together or not."""
        if self._rand() >= self.p_spatial:
            return img, lbl

        B = img.shape[0]
        device = img.device

        # Rotation
        angle = math.radians(self._uniform(-self.rotate_deg, self.rotate_deg))
        c, s = math.cos(angle), math.sin(angle)
        if self.anisotropic:
            # Rotate in the in-plane plane (the two non-anisotropic axes).
            # For raw NIfTI with anisotropy_axis=2: rotate in axes 0,1 (D, H plane).
            # In affine_grid convention, axes are (D, H, W) for 5D (B, C, D, H, W).
            ax = self.anisotropy_axis
            if ax == 0:
                # rotate in H-W plane; D unchanged
                R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]],
                                 device=device, dtype=torch.float32)
            elif ax == 1:
                # rotate in D-W plane; H unchanged
                R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]],
                                 device=device, dtype=torch.float32)
            else:  # ax == 2
                # rotate in D-H plane; W unchanged
                R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]],
                                 device=device, dtype=torch.float32)
        else:
            # Isotropic: rotate around random axis with smaller range
            angle_iso = math.radians(self._uniform(-30, 30))
            c, s = math.cos(angle_iso), math.sin(angle_iso)
            plane = torch.randint(0, 3, (1,)).item()
            if plane == 0:
                R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=device, dtype=torch.float32)
            elif plane == 1:
                R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=torch.float32)
            else:
                R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=device, dtype=torch.float32)

        # Scale (per-axis independent in nnU-Net)
        s0 = self._uniform(self.scale_range[0], self.scale_range[1])
        s1 = self._uniform(self.scale_range[0], self.scale_range[1])
        s2 = self._uniform(self.scale_range[0], self.scale_range[1])
        S = torch.diag(torch.tensor([s0, s1, s2], device=device, dtype=torch.float32))

        theta = (S @ R)
        affine = torch.zeros(B, 3, 4, device=device, dtype=torch.float32)
        for b in range(B):
            affine[b, :3, :3] = theta

        grid = F.affine_grid(affine, img.shape, align_corners=True)
        img_out = F.grid_sample(img, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        lbl_out = F.grid_sample(lbl.float(), grid, mode='nearest',
                                padding_mode='border', align_corners=True).long()
        return img_out, lbl_out

    # ─── Intensity transforms ──────────────────────────────────────────────────
    def _gaussian_noise(self, x):
        if self._rand() >= self.p_noise:
            return x
        var = self._uniform(*self.noise_var)
        return x + torch.randn_like(x) * (var ** 0.5)

    def _gaussian_blur(self, x):
        if self._rand() >= self.p_blur:
            return x
        sigma = self._uniform(*self.blur_sigma)
        ks = int(2 * round(3 * sigma) + 1)
        if ks % 2 == 0:
            ks += 1
        ax = torch.arange(ks, device=x.device, dtype=torch.float32) - ks // 2
        kernel_1d = torch.exp(-0.5 * ax ** 2 / sigma ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        B, C, D, H, W = x.shape
        out = x
        for dim, pad_spec in [(2, (0, 0, 0, 0, ks // 2, ks // 2)),
                               (3, (0, 0, ks // 2, ks // 2, 0, 0)),
                               (4, (ks // 2, ks // 2, 0, 0, 0, 0))]:
            k_shape = [ks if i == dim - 2 else 1 for i in range(3)]
            k_3d = kernel_1d.view(1, 1, *k_shape).expand(C, -1, -1, -1, -1)
            out = F.conv3d(F.pad(out, pad_spec, mode='replicate'), k_3d, groups=C)
        return out

    def _bright_mul(self, x):
        if self._rand() >= self.p_bright_mul:
            return x
        return x * self._uniform(*self.bright_mul)

    def _contrast_aug(self, x):
        if self._rand() >= self.p_contrast:
            return x
        factor = self._uniform(*self.contrast)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        return (x - mean) * factor + mean

    def _lowres(self, x):
        if self._rand() >= self.p_lowres:
            return x
        zoom = self._uniform(*self.lowres_zoom)
        orig_shape = x.shape[2:]
        new_shape = [max(1, int(round(s * zoom))) for s in orig_shape]
        down = F.interpolate(x, size=new_shape, mode='nearest')
        up = F.interpolate(down, size=list(orig_shape), mode='trilinear',
                           align_corners=False)
        return up

    def _gamma(self, x):
        if self._rand() >= self.p_gamma:
            return x
        gamma = self._uniform(*self.gamma)
        invert = self._rand() < self.p_gamma_invert
        x_min, x_max = x.min(), x.max()
        rng = x_max - x_min
        if rng < 1e-8:
            return x
        x_norm = ((x - x_min) / rng).clamp(1e-8, 1.0)
        if invert:
            x_g = 1.0 - (1.0 - x_norm).clamp(1e-8, 1.0).pow(gamma)
        else:
            x_g = x_norm.pow(gamma)
        return x_g * rng + x_min

    def _flip(self, img, lbl):
        for ax in [2, 3, 4]:
            if self._rand() < self.p_flip_per_axis:
                img = torch.flip(img, [ax])
                lbl = torch.flip(lbl, [ax])
        return img, lbl

    def intensity_only(self, img, lbl):
        img = self._gaussian_noise(img)
        img = self._gaussian_blur(img)
        img = self._bright_mul(img)
        img = self._contrast_aug(img)
        img = self._lowres(img)
        img = self._gamma(img)
        img, lbl = self._flip(img, lbl)
        return img, lbl


# ============================================================================
# Shared GPU patch sampler
# ============================================================================
# pos_ratio fraction of patches are GUARANTEED to contain tumor (sampled by
# picking a tumor voxel then choosing a center within reach so the patch
# centered there will contain that voxel). neg patches are random.
# Both pos and neg patches are sampled with respect to the OVERSIZED patch size,
# so subsequent spatial augmentation has full content (no border-padding artifacts).

def compute_oversized_patch(patch_size, factor=1.4):
    return tuple(int(math.ceil(ps * factor)) for ps in patch_size)


def center_crop_tensor(x, target_size):
    starts = [max(0, (x.shape[a + 2] - target_size[a]) // 2) for a in range(3)]
    return x[:, :, starts[0]:starts[0] + target_size[0],
                   starts[1]:starts[1] + target_size[1],
                   starts[2]:starts[2] + target_size[2]]


def pad_to_at_least(img, lbl, min_shape):
    """Pad img and lbl on the right of each spatial axis to at least min_shape."""
    pad = [0] * 6
    need = False
    for ax in range(3):
        if img.shape[ax + 2] < min_shape[ax]:
            deficit = min_shape[ax] - img.shape[ax + 2]
            pad[(2 - ax) * 2 + 1] = deficit
            need = True
    if need:
        img = F.pad(img, pad, mode='constant', value=0)
        lbl = F.pad(lbl.float(), pad, mode='constant', value=0).long()
    return img, lbl


def sample_patch_centers(lbl_res, num_patches, oversized_patch, pos_ratio=0.5):
    """
    Sample N patch centers. pos_ratio fraction guarantee patch contains tumor.

    Algorithm:
    - Pos centers: pick a fg voxel uniformly, then pick a center such that
      the patch [center - patch/2, center + patch/2] contains that voxel.
      Constrains center to be within (oversized_patch/2 - 1) of fg voxel
      AND within valid range so patch fits in volume.
    - Neg centers: uniform random within valid range (patch fits in volume).
    """
    device = lbl_res.device
    vol_shape = torch.tensor(lbl_res.shape[2:], device=device, dtype=torch.float32)
    half = torch.tensor(oversized_patch, device=device, dtype=torch.float32) / 2.0
    valid_low = half
    valid_high = vol_shape - half
    valid_high = torch.maximum(valid_high, valid_low)  # ensure low <= high

    num_pos = int(num_patches * pos_ratio)
    num_neg = num_patches - num_pos

    fg_coords = (lbl_res[0, 0] > 0).nonzero(as_tuple=False).float()
    centers = []

    # Pos
    if len(fg_coords) > 0 and num_pos > 0:
        for _ in range(num_pos):
            idx = torch.randint(len(fg_coords), (1,), device=device).item()
            fg = fg_coords[idx]
            # center must be within oversized_patch/2 of fg so patch contains it
            ctr_low = torch.maximum(fg - (half - 1), valid_low)
            ctr_high = torch.minimum(fg + (half - 1), valid_high)
            ctr_high = torch.maximum(ctr_high, ctr_low)
            ctr = ctr_low + torch.rand(3, device=device) * (ctr_high - ctr_low)
            centers.append(ctr)
    elif num_pos > 0:
        for _ in range(num_pos):
            centers.append(valid_low + torch.rand(3, device=device) * (valid_high - valid_low))

    # Neg
    for _ in range(num_neg):
        centers.append(valid_low + torch.rand(3, device=device) * (valid_high - valid_low))

    return centers


def crop_patches_at_centers(volume, centers, patch_size):
    """Crop patches of patch_size centered at each center. Pure indexing."""
    pD, pH, pW = patch_size
    D, H, W = volume.shape[2:]
    patches = []
    crop_idx = []
    for ctr in centers:
        d1 = int(round(ctr[0].item())) - pD // 2
        h1 = int(round(ctr[1].item())) - pH // 2
        w1 = int(round(ctr[2].item())) - pW // 2
        d1 = max(0, min(d1, D - pD))
        h1 = max(0, min(h1, H - pH))
        w1 = max(0, min(w1, W - pW))
        patches.append(volume[:, :, d1:d1 + pD, h1:h1 + pH, w1:w1 + pW])
        crop_idx.append((d1, h1, w1))
    return torch.cat(patches, dim=0), crop_idx


# ============================================================================
# Baseline model: shared patcher (no resampler — Spacingd is in CPU transforms)
# ============================================================================

class BaselinePatchModel(nn.Module):
    """
    Already-resampled image+label come in. Sample patches → augment → forward.
    """

    def __init__(self, backbone, patch_size, augmenter=None, oversized_factor=1.4):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.oversized_patch = compute_oversized_patch(patch_size, oversized_factor)
        self.augmenter = augmenter

    def forward(self, image_res, label_res, num_patches=4, pos_ratio=0.5):
        with torch_autocast('cuda', enabled=False):
            image_res = image_res.float()
            label_res = label_res.long()

            image_res, label_res = pad_to_at_least(
                image_res, label_res, self.oversized_patch)

            centers = sample_patch_centers(
                label_res, num_patches, self.oversized_patch, pos_ratio)

            patches, crop_idx = crop_patches_at_centers(
                image_res, centers, self.oversized_patch)
            lbl_patches, _ = crop_patches_at_centers(
                label_res, centers, self.oversized_patch)

            if self.training and self.augmenter is not None:
                patches, lbl_patches = self.augmenter.spatial_only(patches, lbl_patches)

            patches = center_crop_tensor(patches, self.patch_size)
            lbl_patches = center_crop_tensor(lbl_patches, self.patch_size)

            if self.training and self.augmenter is not None:
                patches, lbl_patches = self.augmenter.intensity_only(patches, lbl_patches)

        logits = self.backbone(patches)

        first_logit = logits[0] if isinstance(logits, (list, tuple)) else logits
        if torch.isnan(first_logit).any():
            print(f"[NaN-DIAG] NaN after backbone! input range=[{patches.min():.2f}, {patches.max():.2f}]")
            return None, None, None

        return logits, lbl_patches, crop_idx

    def inference_forward(self, x):
        return self.backbone(x)


# ============================================================================
# OOF utilities
# ============================================================================

def warp_back_via_grid_sample(probs, src_spacing, src_shape,
                              dst_spacing, dst_shape, mode='bilinear'):
    """
    Warp `probs` (a tensor in resampled space) back to original raw space.

    probs       : [1, C, D', H', W']  in resampled space (spacing = src_spacing)
    src_shape   : (D', H', W')  shape of probs
    dst_shape   : (D, H, W)  target raw shape
    src_spacing : [3]  the spacing OF THE PROBS volume (resampled spacing)
    dst_spacing : [3]  the original raw spacing

    Returns probs in shape [1, C, D, H, W].
    """
    device = probs.device
    s_src = torch.tensor(src_spacing, device=device, dtype=torch.float32)
    s_dst = torch.tensor(dst_spacing, device=device, dtype=torch.float32)
    grids = []
    for d in range(3):
        idx = torch.arange(dst_shape[d], device=device, dtype=torch.float32)
        # Map dst voxel i -> src voxel coord. Both anchored at voxel 0,
        # spacing-correct: i_dst * s_dst = i_src * s_src
        vox = idx * (s_dst[d] / s_src[d])
        norm = 2.0 * vox / max(src_shape[d] - 1, 1) - 1.0
        grids.append(norm)
    gd, gh, gw = torch.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
    grid = torch.stack([gw, gh, gd], dim=-1).unsqueeze(0)
    return F.grid_sample(probs, grid, mode=mode,
                         padding_mode='border', align_corners=True)


def warp_back_via_affine(probs, src_affine, dst_affine, dst_shape,
                         mode='bilinear'):
    """Warp `probs` in SRC voxel grid back to DST voxel grid using affines.

    src_affine : (4,4) maps src voxel (i,j,k,1) → world (x,y,z,1)
    dst_affine : (4,4) maps dst voxel (i,j,k,1) → world (x,y,z,1)
                 (dst = raw NIfTI, usually orig_nii.affine)
    dst_shape  : (D, H, W) of the target volume in dst voxels.

    Correctly handles: CropFG translation, Spacingd scale, orientation
    (anything baked into either affine). Replaces the old
    `warp_back_via_grid_sample` that assumed src voxel 0 == dst voxel 0.
    """
    device = probs.device
    D, H, W = dst_shape

    if not torch.is_tensor(src_affine):
        src_affine = torch.tensor(src_affine, device=device, dtype=torch.float32)
    else:
        src_affine = src_affine.to(device=device, dtype=torch.float32)
    if not torch.is_tensor(dst_affine):
        dst_affine = torch.tensor(dst_affine, device=device, dtype=torch.float32)
    else:
        dst_affine = dst_affine.to(device=device, dtype=torch.float32)

    ii = torch.arange(D, device=device, dtype=torch.float32)
    jj = torch.arange(H, device=device, dtype=torch.float32)
    kk = torch.arange(W, device=device, dtype=torch.float32)
    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing='ij')
    ones = torch.ones_like(gi)
    # dst voxel coords (i, j, k, 1) as [D,H,W,4]
    coords = torch.stack([gi, gj, gk, ones], dim=-1)

    # world = dst_affine @ coords^T
    world = torch.einsum('ab,dhwb->dhwa', dst_affine, coords)
    # src_vox = inv(src_affine) @ world
    src_affine_inv = torch.linalg.inv(src_affine)
    src_vox = torch.einsum('ab,dhwb->dhwa', src_affine_inv, world)[..., :3]

    D_s, H_s, W_s = probs.shape[2], probs.shape[3], probs.shape[4]
    n_i = 2.0 * src_vox[..., 0] / max(D_s - 1, 1) - 1.0
    n_j = 2.0 * src_vox[..., 1] / max(H_s - 1, 1) - 1.0
    n_k = 2.0 * src_vox[..., 2] / max(W_s - 1, 1) - 1.0
    # grid_sample expects (x, y, z) = (W_axis, H_axis, D_axis)
    grid = torch.stack([n_k, n_j, n_i], dim=-1).unsqueeze(0)
    return F.grid_sample(probs, grid, mode=mode,
                         padding_mode='border', align_corners=True)


def compute_hd95(pred_bin, label_bin, spacing):
    if pred_bin.sum() == 0 or label_bin.sum() == 0:
        return float('inf')
    pe = pred_bin ^ binary_erosion(pred_bin)
    le = label_bin ^ binary_erosion(label_bin)
    if pe.sum() == 0 or le.sum() == 0:
        return float('inf')
    dt_l = distance_transform_edt(~le, sampling=spacing)
    dt_p = distance_transform_edt(~pe, sampling=spacing)
    d_pl = dt_l[pe]
    d_lp = dt_p[le]
    return float(max(np.percentile(d_pl, 95), np.percentile(d_lp, 95)))


def compute_assd(pred_bin, label_bin, spacing):
    if pred_bin.sum() == 0 or label_bin.sum() == 0:
        return float('inf')
    pe = pred_bin ^ binary_erosion(pred_bin)
    le = label_bin ^ binary_erosion(label_bin)
    if pe.sum() == 0 or le.sum() == 0:
        return float('inf')
    dt_l = distance_transform_edt(~le, sampling=spacing)
    dt_p = distance_transform_edt(~pe, sampling=spacing)
    return float((dt_l[pe].mean() + dt_p[le].mean()) / 2)


def compute_volume_mm3(mask, spacing):
    return float(mask.sum() * np.prod(spacing))


# ============================================================================
# Dataset fingerprint (raw NIfTI, baseline v3 / nnU-Net heuristic)
# ============================================================================

def compute_fingerprint(pairs):
    """Compute target spacing + patch size + DynUNet config from raw NIfTI headers."""
    print("[INFO] Computing dataset fingerprint (raw NIfTI, no RAS)…")
    all_spacings, all_shapes = [], []
    for ip, _ in pairs:
        img = nib.load(ip)
        sp = np.array(img.header.get_zooms()[:3], dtype=np.float64)
        sh = np.array(img.header.get_data_shape()[:3])
        all_spacings.append(sp)
        all_shapes.append(sh)
    all_spacings = np.array(all_spacings)
    all_shapes = np.array(all_shapes)

    median_sp = np.median(all_spacings, axis=0)
    target_sp = median_sp.copy()
    anisotropic = False
    aniso_axis = int(np.argmax(median_sp))
    if median_sp.max() / median_sp.min() >= 3.0:
        anisotropic = True
        target_sp[aniso_axis] = float(np.percentile(all_spacings[:, aniso_axis], 10))

    print(f"[INFO] Median spacing (raw): {median_sp.round(4)}")
    print(f"[INFO] Target spacing:       {target_sp.round(4)}")
    print(f"[INFO] Anisotropic: {anisotropic}, axis: {aniso_axis}")

    resampled_shapes = all_shapes * (all_spacings / target_sp)
    median_shape = np.median(resampled_shapes, axis=0).astype(int)
    print(f"[INFO] Median resampled shape: {median_shape}")

    # Patch size — same logic as baseline v3
    patch_size = [0, 0, 0]
    if anisotropic:
        ax = aniso_axis
        patch_size[ax] = int(min(median_shape[ax], 128))
        for a in range(3):
            if a != ax:
                patch_size[a] = int(min(median_shape[a], 192))
    else:
        for a in range(3):
            patch_size[a] = int(min(median_shape[a], 128))

    # Network strides + channels
    channels = [32]
    strides = []
    cur = np.array(patch_size, dtype=np.int32)
    while True:
        st = [2, 2, 2]
        if anisotropic and len(strides) < 2:
            st[aniso_axis] = 1
        ns = np.ceil(cur / st)
        if (ns < 4).any():
            break
        strides.append(tuple(st))
        cur = ns
        channels.append(min(channels[-1] * 2, 512))
    channels = tuple(channels)

    total_stride = np.array([1, 1, 1], dtype=np.int32)
    for s in strides:
        total_stride *= np.array(s, dtype=np.int32)
    for ax in range(3):
        if patch_size[ax] < total_stride[ax]:
            patch_size[ax] = int(total_stride[ax])
        elif patch_size[ax] % total_stride[ax] != 0:
            patch_size[ax] = int(np.ceil(patch_size[ax] / total_stride[ax]) * total_stride[ax])
    patch_size = tuple(patch_size)

    num_stages = len(channels)
    kernel_sizes = []
    for st in range(num_stages):
        if anisotropic and st < 2:
            kernel_sizes.append((3, 3, 1))
        else:
            kernel_sizes.append((3, 3, 3))
    up_kernel_sizes = tuple(strides)
    dynunet_strides = tuple([(1, 1, 1)] + list(strides))

    print(f"[INFO] Patch size: {patch_size}")
    print(f"[INFO] Channels: {channels}")
    print(f"[INFO] Strides: {strides}, total_stride: {total_stride}")

    return {
        'target_spacing': tuple(float(x) for x in target_sp),
        'anisotropic': anisotropic, 'anisotropy_axis': aniso_axis,
        'patch_size': patch_size, 'channels': channels,
        'strides': strides, 'kernel_sizes': kernel_sizes,
        'up_kernel_sizes': up_kernel_sizes, 'dynunet_strides': dynunet_strides,
        'total_stride': total_stride,
    }


# ============================================================================
# Folds
# ============================================================================

def build_folds_from_split_json(all_files, split_json):
    with open(split_json) as f:
        splits = json.load(f)
    folds = []
    by_subj = {d['subject']: d for d in all_files}
    for fid, fold_dict in enumerate(splits if isinstance(splits, list) else [splits[k] for k in sorted(splits.keys())]):
        train_subj = fold_dict.get('train', [])
        val_subj = fold_dict.get('val', [])
        train_files = [by_subj[s] for s in train_subj if s in by_subj]
        val_files = [by_subj[s] for s in val_subj if s in by_subj]
        folds.append({'fold': fid, 'train_files': train_files, 'val_files': val_files})
    return folds


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Baseline v10 — raw-axis MONAI Spacingd baseline (matched with v10 adaptive).')
    p.add_argument('--base_dir', type=str, default='../dataset/lipo')
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--split_json', type=str, default='./lipo_split.json')
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--max_epochs', type=int, default=1000)
    p.add_argument('--val_interval', type=int, default=2)
    p.add_argument('--early_stop_patience', type=int, default=100)
    p.add_argument('--num_patches', type=int, default=4)
    p.add_argument('--pos_ratio', type=float, default=0.5)
    p.add_argument('--variant', type=str, default='v10_match',
                   choices=['v3_match', 'v10_match'],
                   help='v3_match: Spacingd→CropFG→ZScore(otsu). '
                        'v10_match: CropFG→ZScore(gt_zero)→Spacingd.')
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    set_determinism(seed=2025)

    base_dir = args.base_dir
    global_output_dir = args.out_dir or './outputs/lipo_obelisk_baseline_v10'
    split_json = args.split_json
    os.makedirs(global_output_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(base_dir, '*_image.nii.gz')))
    pairs = []
    for ip in img_paths:
        stem = os.path.basename(ip).replace('_image.nii.gz', '')
        lp = os.path.join(base_dir, f'{stem}_segmentation.nii.gz')
        if os.path.exists(lp):
            pairs.append((ip, lp))
    print(f"[INFO] Total pairs: {len(pairs)}")

    all_files = [{'image': ip, 'label': lp, 'label_path': lp,
                  'subject': extract_subject_id(ip)} for ip, lp in pairs]

    fp = compute_fingerprint(pairs)
    target_pixdim = fp['target_spacing']
    patch_size = fp['patch_size']
    channels = fp['channels']
    kernel_sizes = fp['kernel_sizes']
    up_kernel_sizes = fp['up_kernel_sizes']
    dynunet_strides = fp['dynunet_strides']
    total_stride = fp['total_stride']
    anisotropic = fp['anisotropic']
    anisotropy_axis = fp['anisotropy_axis']

    # ── CPU transforms ────────────────────────────────────────────────────────
    # variant v3_match  : Spacingd → CropFG(Otsu) → ZScore(fg=otsu, on resampled)
    # variant v10_match : CropFG(Otsu) → ZScore(fg=gt_zero, on raw) → Spacingd
    # StoreAffined captures final affine (post-everything) for warp-back.
    if args.variant == 'v3_match':
        print(f"[INFO] Pipeline variant: v3_match "
              f"(Spacingd → CropFG → ZScore(fg=otsu))")
        core_xforms = [
            Spacingd(keys=['image', 'label'], pixdim=target_pixdim,
                     mode=(3, 'nearest')),
            CropForegroundd(keys=['image', 'label'], source_key='image',
                            select_fn=otsu_select_fn, margin=(10, 10, 3)),
            ZScoreNormalizeForegroundd(keys=['image'], fg_mode='otsu'),
        ]
    else:
        print(f"[INFO] Pipeline variant: v10_match "
              f"(CropFG → ZScore(fg=gt_zero) → Spacingd)")
        core_xforms = [
            CropForegroundd(keys=['image', 'label'], source_key='image',
                            select_fn=otsu_select_fn, margin=(10, 10, 3)),
            ZScoreNormalizeForegroundd(keys=['image'], fg_mode='gt_zero'),
            Spacingd(keys=['image', 'label'], pixdim=target_pixdim,
                     mode=(3, 'nearest')),
        ]

    train_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        *core_xforms,
        StoreAffined(keys=['image'], key_out='src_affine'),
        EnsureTyped(keys=['image'], dtype=np.float32),
        EnsureTyped(keys=['label'], dtype=np.uint8),
        ToTensord(keys=['image', 'label']),
    ])
    val_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        *core_xforms,
        StoreAffined(keys=['image'], key_out='src_affine'),
        EnsureTyped(keys=['image'], dtype=np.float32),
        EnsureTyped(keys=['label'], dtype=np.uint8),
        ToTensord(keys=['image', 'label']),
    ])

    post_process = KeepLargestConnectedComponent(applied_labels=[1])

    # ── Folds ────────────────────────────────────────────────────────────────
    if split_json and os.path.exists(split_json):
        folds = build_folds_from_split_json(all_files, split_json)
    else:
        gkf = GroupKFold(n_splits=args.n_splits)
        folds = []
        subjects = np.array([d['subject'] for d in all_files])
        indices = np.arange(len(all_files))
        for fid, (tr_idx, va_idx) in enumerate(gkf.split(indices, groups=subjects)):
            folds.append({'fold': fid,
                          'train_files': [all_files[i] for i in tr_idx],
                          'val_files': [all_files[i] for i in va_idx]})

    for fo in folds:
        print(f"  Fold {fo['fold']}: train={len(fo['train_files'])}, val={len(fo['val_files'])}")

    # ── Training loop ─────────────────────────────────────────────────────────
    max_epochs = args.max_epochs
    val_interval = args.val_interval
    early_stop_patience = args.early_stop_patience
    base_lr = 0.01
    pin_memory = torch.cuda.is_available()
    avail_cpus = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                  else multiprocessing.cpu_count())
    num_workers = max(0, min(avail_cpus - 1, 4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folds_to_run = folds
    if args.fold is not None:
        folds_to_run = [f for f in folds if f['fold'] == args.fold]
        if not folds_to_run:
            raise ValueError(f"Fold {args.fold} not found.")

    global_dice_csv = os.path.join(global_output_dir, 'oof_all_folds_dice.csv')
    if not os.path.exists(global_dice_csv):
        with open(global_dice_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['SubjectID', 'Fold', 'Dice', 'HD95', 'ASSD',
                                    'PredVol_mm3', 'GTVol_mm3'])

    for fold_info in folds_to_run:
        CURR_FOLD = fold_info['fold']
        train_files = fold_info['train_files']
        val_files = fold_info['val_files']

        print(f"\n{'=' * 60}")
        print(f"Fold {CURR_FOLD} — Baseline v10 (raw-axis Spacingd)")
        print(f"{'=' * 60}")

        out_dir = os.path.join(global_output_dir, f'fold_{CURR_FOLD}')
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, 'best_model.pth')
        log_path = os.path.join(out_dir, 'training_log.csv')
        fold_dice_csv = os.path.join(out_dir, 'fold_dice.csv')

        if not os.path.exists(fold_dice_csv):
            with open(fold_dice_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['SubjectID', 'Fold', 'Dice'])

        # Datasets & loaders
        train_ds = CacheDataset(train_files, train_transforms,
                                cache_rate=0, num_workers=num_workers)
        val_ds = CacheDataset(val_files, val_transforms,
                              cache_rate=1.0, num_workers=num_workers)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                                  num_workers=0, pin_memory=pin_memory,
                                  collate_fn=list_data_collate)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=0, pin_memory=pin_memory)

        # Model
        backbone = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            filters=channels, strides=dynunet_strides,
            kernel_size=kernel_sizes, upsample_kernel_size=up_kernel_sizes,
            act_name=('leakyrelu', {'negative_slope': 0.01}),
            norm_name='instance',
            deep_supervision=True, deep_supr_num=3,
        )
        augmenter = NNUNetGPUAugmentation(
            anisotropic=anisotropic, anisotropy_axis=anisotropy_axis)
        model = BaselinePatchModel(backbone, patch_size, augmenter).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Params: {n_params / 1e6:.2f} M")

        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True,
                             squared_pred=True, include_background=False,
                             lambda_dice=0.5, lambda_ce=0.5)
        optimizer = torch.optim.SGD(
            model.backbone.parameters(),
            lr=base_lr, momentum=0.99, nesterov=True, weight_decay=1e-5)
        scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as fc:
                csv.writer(fc).writerow([
                    'epoch', 'train_loss', 'train_dice', 'val_loss',
                    'val_dice_resampled', 'val_dice_origspace',
                    'lr', 'time_sec'])

        best_val_loss = float('inf')
        best_dice_orig = -1.0
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            t0 = time.time()
            train_loss_acc, train_dice_acc, n_steps, n_nan = 0.0, 0.0, 0, 0

            for batch in train_loader:
                images = batch['image'].to(device)      # already resampled by Spacingd
                labels = batch['label'].to(device).long()

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    logits, lbl_patches, _ = model(
                        images, labels, num_patches=args.num_patches,
                        pos_ratio=args.pos_ratio)
                    if logits is None:
                        n_nan += 1
                        continue

                    logits_list = normalize_logits_output(logits)
                    ds_w = get_deep_supervision_weights(len(logits_list))
                    loss = torch.tensor(0.0, device=device)
                    for i, head in enumerate(logits_list):
                        if head.shape[-3:] != lbl_patches.shape[-3:]:
                            ds_lbl = F.interpolate(
                                lbl_patches.float(), size=head.shape[-3:],
                                mode='nearest').long()
                        else:
                            ds_lbl = lbl_patches
                        loss = loss + loss_fn(head, ds_lbl) * ds_w[i]

                    with torch.no_grad():
                        pred = logits_list[0].argmax(dim=1, keepdim=True)
                        train_dice_acc += _dice_from_labels(pred, lbl_patches).item()

                if torch.isnan(loss) or torch.isinf(loss):
                    n_nan += 1
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                train_loss_acc += loss.item()
                n_steps += 1

            # LR schedule
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * poly_lr(epoch + 1, max_epochs)

            # Validation
            if (epoch + 1) % val_interval == 0:
                epoch_loss = train_loss_acc / max(n_steps, 1)
                epoch_dice = train_dice_acc / max(n_steps + n_nan, 1)

                model.eval()
                dice_res_list, dice_orig_list, vloss_list = [], [], []
                with torch.inference_mode(), \
                     amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    for vd in val_loader:
                        vi = vd['image'].to(device)
                        vl = vd['label'].to(device).long()

                        # Pad val volume to at least patch_size for sliding window
                        vi_p, vl_p = pad_to_at_least(vi, vl, patch_size)
                        # Pad to total_stride divisibility
                        rs = list(vi_p.shape[2:])
                        ps_div, p2 = [], False
                        for ax in range(3):
                            ts = int(total_stride[ax])
                            if rs[ax] % ts != 0:
                                ps_div.append(int(np.ceil(rs[ax] / ts) * ts))
                                p2 = True
                            else:
                                ps_div.append(rs[ax])
                        if p2:
                            vi_p = F.pad(vi_p,
                                         (0, ps_div[2] - rs[2],
                                          0, ps_div[1] - rs[1],
                                          0, ps_div[0] - rs[0]),
                                         mode='constant', value=0)

                        def predictor(x): return model.inference_forward(x)
                        val_logits = sliding_window_inference(
                            inputs=vi_p, roi_size=patch_size,
                            sw_batch_size=8, predictor=predictor,
                            overlap=0.5, mode='gaussian')
                        if isinstance(val_logits, (list, tuple)):
                            val_logits = val_logits[0]
                        if p2:
                            val_logits = val_logits[:, :, :rs[0], :rs[1], :rs[2]]

                        # Loss + dice in resampled space
                        v_loss = loss_fn(val_logits, vl_p).item()
                        vloss_list.append(v_loss)
                        vp_res = val_logits.argmax(dim=1, keepdim=True)
                        d_res = _dice_from_labels(post_process(vp_res), vl_p).item()
                        dice_res_list.append(d_res)

                        # Dice in original raw space — load raw label, warp predictions back
                        label_path = vd['label_path'][0]
                        orig_nii = nib.load(label_path)
                        orig_data = orig_nii.get_fdata().astype(np.float32)
                        orig_label = torch.from_numpy(orig_data).unsqueeze(0).unsqueeze(0).to(device)
                        orig_spacing = np.array(orig_nii.header.get_zooms()[:3])
                        orig_shape = tuple(orig_label.shape[2:])

                        probs = torch.softmax(val_logits, dim=1)
                        src_affine = vd['src_affine'][0].cpu().numpy() \
                            if torch.is_tensor(vd['src_affine']) \
                            else np.asarray(vd['src_affine'][0])
                        probs_orig = warp_back_via_affine(
                            probs,
                            src_affine=src_affine,
                            dst_affine=orig_nii.affine,
                            dst_shape=orig_shape,
                            mode='bilinear')
                        pred_orig = probs_orig.argmax(dim=1, keepdim=True)
                        pred_orig = post_process(pred_orig)
                        d_orig = _dice_from_labels(pred_orig, orig_label.long()).item()
                        dice_orig_list.append(d_orig)

                val_dice_res = float(np.mean(dice_res_list))
                val_dice_orig = float(np.mean(dice_orig_list))
                val_loss_ep = float(np.mean(vloss_list))

                # Early stop on val_loss
                if val_loss_ep < best_val_loss:
                    best_val_loss = val_loss_ep
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += val_interval

                # Checkpoint on best dice_orig
                if val_dice_orig > best_dice_orig:
                    best_dice_orig = val_dice_orig
                    best_epoch = epoch + 1
                    torch.save({'model': model.state_dict()}, ckpt_path)
                    print(f"   >>> Ckpt: D_orig={val_dice_orig:.4f} | VLoss={val_loss_ep:.4f}")

                elapsed = time.time() - t0
                msg = (f"Ep {epoch + 1:03d}/{max_epochs} | "
                       f"Loss={epoch_loss:.4f} | TrDice={epoch_dice:.4f} | "
                       f"VL={val_loss_ep:.4f} | D_res={val_dice_res:.4f} | "
                       f"D_orig={val_dice_orig:.4f} | LR={get_lr(optimizer):.5f} | "
                       f"{elapsed:.0f}s")
                if n_nan > 0:
                    msg += f" | nan={n_nan}"
                print(msg)

                with open(log_path, 'a', newline='') as fc:
                    csv.writer(fc).writerow([
                        epoch + 1, f'{epoch_loss:.6f}', f'{epoch_dice:.6f}',
                        f'{val_loss_ep:.6f}', f'{val_dice_res:.6f}',
                        f'{val_dice_orig:.6f}', f'{get_lr(optimizer):.6f}',
                        f'{elapsed:.2f}'])

                if epochs_no_improve >= early_stop_patience:
                    print(f"\n[Early Stop] No improvement for {early_stop_patience} epochs.")
                    break

        print(f"\nFold {CURR_FOLD} done | Best epoch={best_epoch} | "
              f"BestVLoss={best_val_loss:.4f} | BestDice_orig={best_dice_orig:.4f}")

        # ── OOF inference ────────────────────────────────────────────────────
        print(f"\n{'=' * 50}")
        print(f"OOF Inference Fold {CURR_FOLD}")
        print(f"{'=' * 50}")

        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()

        pred_dir = os.path.join(out_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        oof_results = []

        with torch.inference_mode():
            for vd in val_loader:
                vi = vd['image'].to(device)
                vl = vd['label'].to(device).long()

                vi_p, vl_p = pad_to_at_least(vi, vl, patch_size)
                rs = list(vi_p.shape[2:])
                ps_div, p2 = [], False
                for ax in range(3):
                    ts = int(total_stride[ax])
                    if rs[ax] % ts != 0:
                        ps_div.append(int(np.ceil(rs[ax] / ts) * ts))
                        p2 = True
                    else:
                        ps_div.append(rs[ax])
                if p2:
                    vi_p = F.pad(vi_p,
                                 (0, ps_div[2] - rs[2],
                                  0, ps_div[1] - rs[1],
                                  0, ps_div[0] - rs[0]),
                                 mode='constant', value=0)

                def predictor(x): return model.inference_forward(x)
                val_logits = sliding_window_inference(
                    inputs=vi_p, roi_size=patch_size, sw_batch_size=8,
                    predictor=predictor, overlap=0.5, mode='gaussian')
                if isinstance(val_logits, (list, tuple)):
                    val_logits = val_logits[0]
                if p2:
                    val_logits = val_logits[:, :, :rs[0], :rs[1], :rs[2]]

                # Warp back to raw original space
                label_path = vd['label_path'][0]
                orig_nii = nib.load(label_path)
                orig_data = orig_nii.get_fdata().astype(np.float32)
                orig_label = torch.from_numpy(orig_data).unsqueeze(0).unsqueeze(0).to(device)
                orig_spacing = np.array(orig_nii.header.get_zooms()[:3])
                orig_shape = tuple(orig_label.shape[2:])

                probs = torch.softmax(val_logits, dim=1)
                src_affine = vd['src_affine'][0].cpu().numpy() \
                    if torch.is_tensor(vd['src_affine']) \
                    else np.asarray(vd['src_affine'][0])
                probs_orig = warp_back_via_affine(
                    probs, src_affine=src_affine,
                    dst_affine=orig_nii.affine, dst_shape=orig_shape,
                    mode='bilinear')
                pred_orig = probs_orig.argmax(dim=1, keepdim=True)
                pred_orig = post_process(pred_orig)

                cur_dice = _dice_from_labels(pred_orig, orig_label.long()).item()

                pred_np = pred_orig[0, 0].cpu().numpy().astype(bool)
                label_np = orig_label[0, 0].cpu().numpy().astype(bool)
                hd95 = compute_hd95(pred_np, label_np, orig_spacing)
                assd = compute_assd(pred_np, label_np, orig_spacing)
                pred_vol = compute_volume_mm3(pred_np, orig_spacing)
                gt_vol = compute_volume_mm3(label_np, orig_spacing)

                try:
                    subj_id = (vd['subject'][0] if 'subject' in vd
                               else extract_subject_id(vd['image_meta_dict']['filename_or_obj'][0]))
                except Exception:
                    subj_id = 'Unknown'

                print(f"   {subj_id}: Dice={cur_dice:.4f} | HD95={hd95:.2f}mm | "
                      f"ASSD={assd:.2f}mm | Vol={pred_vol:.0f}/{gt_vol:.0f}mm3")

                oof_results.append({
                    'subject': subj_id, 'dice': cur_dice, 'hd95': hd95,
                    'assd': assd, 'pred_vol': pred_vol, 'gt_vol': gt_vol})

                save_path = os.path.join(pred_dir, f'{subj_id}_pred.nii.gz')
                pred_nii = nib.Nifti1Image(pred_np.astype(np.uint8),
                                           orig_nii.affine, orig_nii.header)
                nib.save(pred_nii, save_path)

                with open(fold_dice_csv, 'a', newline='') as fout:
                    csv.writer(fout).writerow([subj_id, CURR_FOLD, f'{cur_dice:.6f}'])
                with open(global_dice_csv, 'a', newline='') as fout:
                    csv.writer(fout).writerow([
                        subj_id, CURR_FOLD, f'{cur_dice:.6f}',
                        f'{hd95:.4f}', f'{assd:.4f}',
                        f'{pred_vol:.0f}', f'{gt_vol:.0f}'])

        dices = [r['dice'] for r in oof_results]
        hd95s = [r['hd95'] for r in oof_results if r['hd95'] != float('inf')]
        assds = [r['assd'] for r in oof_results if r['assd'] != float('inf')]
        print(f"\nFold {CURR_FOLD} OOF Summary (n={len(oof_results)})")
        print(f"  Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
        if hd95s:
            print(f"  HD95: {np.mean(hd95s):.2f} +/- {np.std(hd95s):.2f} mm")
        if assds:
            print(f"  ASSD: {np.mean(assds):.2f} +/- {np.std(assds):.2f} mm")

    print(f"\n{'=' * 60}")
    print("All folds completed!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
