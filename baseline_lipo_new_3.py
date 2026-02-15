import os
import glob
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

from monai.data import CacheDataset, DataLoader, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.utils import set_determinism
from monai.transforms import KeepLargestConnectedComponent, ScaleIntensityRangePercentilesd
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandAffined, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, RandAdjustContrastd, RandSimulateLowResolutiond,
    RandFlipd, EnsureTyped, ToTensord, SaveImage, MapTransform,
    RandomizableTransform
)

# --- Helper Functions ---

def _dice_from_labels(pred, target, eps: float = 1e-6):
    """
    计算 Batch 的平均 Dice Score (Foreground 1)
    pred:   [B, 1, H, W, D] or [B, H, W, D]
    target: [B, 1, H, W, D] or [B, H, W, D]
    """
    # Squeeze channel dimension if present
    if pred.dim() == 5 and pred.size(1) == 1:
        pred = pred[:, 0]  # 选择了第二维度为0，就相当于删掉了第二维的信息
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
    """生成深监督权重: 1, 0.5, 0.25... 并归一化"""
    weights = np.array([1 / (2 ** i) for i in range(num_stages)])
    weights = weights / weights.sum()
    return weights


def normalize_logits_output(logits):
    """
    标准化模型输出为list格式  # 把模型输出 logits 统一整理成一个 Python list，无论模型有没有 deep supervision，都返回同样格式
    处理三种可能的输出格式：
    1. 6D tensor [B, Heads, C, D, H, W] (深监督，相同尺寸)
    2. List/Tuple of 5D tensors (深监督，不同尺寸)
    3. 5D tensor [B, C, D, H, W] (无深监督)
    """
    if isinstance(logits, (list, tuple)):
        return list(logits)
    elif logits.dim() == 6:
        return list(logits.unbind(dim=1)) # 会把第二个维度拆开为多个张量，再变成list
    else:
        return [logits] # 统一成list

def otsu_select_fn(x):
    """
    x: numpy array representing a volume
    返回 True/False mask，用于 CropForegroundd 的 foreground 检测
    """
    x = x.astype(np.float32)
    # 计算 Otsu 阈值
    thr = threshold_otsu(x)
    # 返回前景 mask
    return x > thr

class NnUNetLikeRandFlipd(MapTransform, RandomizableTransform):
    """
    nnU-Net style random mirroring:
    - with prob `prob`, enter mirroring mode
    - inside mirroring mode, each axis in `spatial_axes` is flipped independently with prob 0.5
    """

    def __init__(self, keys, prob=0.5, spatial_axes=(0, 1, 2)):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.spatial_axes = spatial_axes
        self._axes_to_flip = []
        self._do_transform = False

    def randomize(self, data=None):
        """
        决定是否做 mirroring，以及要翻转哪些轴
        """
        # overall: 是否启用 mirroring
        self._do_transform = self.R.random() < self.prob
        self._axes_to_flip = []

        if self._do_transform:
            # mirroring 模式下，每个轴再以 0.5 概率翻转
            for ax in self.spatial_axes:
                if self.R.random() < 0.5:
                    self._axes_to_flip.append(ax)

    def __call__(self, data):
        """
        MapTransform 要求子类实现 __call__，这里自己写逻辑：
        - 调用 randomize()
        - 根据 _axes_to_flip 翻转 image / label
        """
        d = dict(data)
        self.randomize(d)

        if not self._axes_to_flip:
            # 不翻任何轴，直接返回
            return d

        # MONAI 默认 3D: [C, H, W, D]
        # spatial_axes=(0,1,2) 对应 H,W,D → 实际 flip 的轴需要 +1，跳过 channel 维
        flip_axes_np = [ax + 1 for ax in self._axes_to_flip]

        for key in self.keys:
            arr = d[key]

            # torch tensor：用 torch.flip
            if torch.is_tensor(arr):
                # torch.flip 用 dims，和 numpy 的 axis 一样语义
                d[key] = torch.flip(arr, dims=flip_axes_np)
            else:
                # numpy array：用 np.flip
                d[key] = np.flip(arr, axis=flip_axes_np).copy()

        return d

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    set_determinism(seed=2025)

    # ---------------------------------------------------------------------
    # 0. Configuration & Global Paths
    # ---------------------------------------------------------------------
    base_dir = ".dataset/LIPO"
    global_output_dir = "./outputs/baseline_lipo_new_3"
    os.makedirs(global_output_dir, exist_ok=True)

    global_dice_csv = os.path.join(global_output_dir, "final_dice.csv")
    if not os.path.exists(global_dice_csv):
        with open(global_dice_csv, "w", newline="") as f:
            csv.writer(f).writerow(["SubjectID", "Fold", "Dice", "PredictionPath"])

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
        raise FileNotFoundError("No image-label pairs found. Please check 'base_dir' path.")

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

    # Initial Patch Size Estimation (without divisibility constraints)
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
        # Use anisotropic stride for first 2 stages if data is anisotropic
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

    # Final Patch Size Adjustment: ensure divisibility by total stride
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

    # ---------------------------------------------------------------------
    # 3. Data Transforms
    # ---------------------------------------------------------------------
    num_patches_per_image = 4

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim, mode=(3, "nearest")), # third-order spline interpolation
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", select_fn=otsu_select_fn, margin=(10, 10, 3)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=patch_size, pos=1, neg=1,
            num_samples=num_patches_per_image,
        ),
        RandAffined(
            keys=["image", "label"], prob=0.2,
            rotate_range=[
                (0,0),
                (0,0),
                (-np.deg2rad(180), np.deg2rad(180)) # fixed z axis, only rotate at x/y axis
            ] if anisotropic else [
                (-np.deg2rad(30), np.deg2rad(30)),
                (-np.deg2rad(30), np.deg2rad(30)),
                (-np.deg2rad(30), np.deg2rad(30))
            ],
            scale_range=[(-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4)],
            mode=(3, "nearest"), padding_mode="border"
        ),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),  # prob=0.15，方差 ~ U(0,0.1) 在0到0.1之间随机 所以还是可以修改
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)), #prob=0.2, 然后每个模态再0.5，方差0.5-15


        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15), # Brightness：乘 U(0.7,1.3)，prob=0.15。
        RandScaleIntensityd(keys=["image"], factors=(-0.35, 0.5), prob=0.15),
        RandSimulateLowResolutiond(keys=["image"], prob=0.25, zoom_range=(0.5, 1.0), upsample_mode="trilinear"), # 这个改的有问题 Simulation of low resolution
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5)), # Gamma augmentation
        # RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        # RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        # RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),

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
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", select_fn=otsu_select_fn, margin=(10, 10, 3)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ToTensord(keys=["image", "label"]),
    ])

    post_process = KeepLargestConnectedComponent(applied_labels=[1])

    # ---------------------------------------------------------------------
    # 4. Cross-Validation Loop
    # ---------------------------------------------------------------------
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
    early_stop_patience = 50  # 早停: 50个epoch无提升则停止

    for CURR_FOLD, f in enumerate(folds):
        print(f"\n{'=' * 50}")
        print(f"Running Fold {CURR_FOLD}")
        print(f"{'=' * 50}")

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

        # Model
        model = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            filters=channels, strides=dynunet_strides,
            kernel_size=kernel_sizes, upsample_kernel_size=up_kernel_sizes,
            act_name=("leakyrelu", {"negative_slope": 0.01}),
            norm_name="instance",
            deep_supervision=True, deep_supr_num = 3
        ).to(device)

        # [新增] 打印模型参数量
        # -----------------------------------------------------------
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Model Total Trainable Parameters: {total_params / 1e6:.2f} M ({total_params})")
        # -----------------------------------------------------------

        # Loss Function
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True,
                             include_background=False, lambda_dice=0.5, lambda_ce=0.5)

        # Optimizer & Scheduler
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99,
                                    nesterov=True, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: poly_lr(epoch, max_epochs))
        scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        # Initialize Log File
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as fcsv:
                csv.writer(fcsv).writerow(
                    ["epoch", "train_loss", "train_dice", "val_loss", "val_dice", "lr", "time_sec"])

        best_metric = -1.0
        best_epoch = -1
        epochs_no_improve = 0  # 早停计数器

        # --- Training Loop ---
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss_accum = 0.0
            train_dice_accum = 0.0
            num_batches = 0

            # 正常的epoch循环：遍历整个训练集
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).long()

                optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    logits = model(images)
                    logits = normalize_logits_output(logits)

                    # Deep Supervision with Label Downsampling
                    loss = 0
                    ds_weights = get_deep_supervision_weights(len(logits))

                    for i, head_logit in enumerate(logits):
                        current_spatial_shape = head_logit.shape[-3:]

                        # Downsample labels if needed
                        if current_spatial_shape != labels.shape[-3:]:
                            ds_labels = F.interpolate(labels.float(), size=current_spatial_shape,
                                                      mode="nearest").long()
                        else:
                            ds_labels = labels

                        l = loss_fn(head_logit, ds_labels)
                        loss += l * ds_weights[i]

                    # Monitor Train Dice (use highest resolution output)
                    train_pred = logits[0].argmax(dim=1, keepdim=True)
                    train_dice_val = _dice_from_labels(train_pred, labels)
                    train_dice_accum += train_dice_val.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_accum += loss.item()
                num_batches += 1

            # 计算epoch平均值
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

                        # Handle potential list output
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

                # Save Best Model & Early Stopping Check
                if val_dice_epoch > best_metric:
                    best_metric = val_dice_epoch
                    best_epoch = epoch
                    epochs_no_improve = 0  # 重置早停计数器
                    torch.save({"model": model.state_dict()}, ckpt_path)
                    print(f"   >>> New Best: Val_Dice={best_metric:.4f} | Train_Dice={epoch_train_dice:.4f}")
                else:
                    epochs_no_improve += val_interval  # 增加早停计数（按验证间隔累加）

            scheduler.step()
            elapsed = time.time() - t0

            # Print Progress
            log_str = f"Ep {epoch:03d} | Loss={epoch_loss:.4f} | Train_Dice={epoch_train_dice:.4f}"
            if val_dice_epoch is not None:
                log_str += f" | Val_Loss={val_loss_epoch:.4f} | Val_Dice={val_dice_epoch:.4f} | Best={best_metric:.4f}"
            log_str += f" | LR={get_lr(optimizer):.6f} | Time={elapsed:.1f}s"
            print(log_str)

            # Log to CSV
            with open(log_path, "a", newline="") as fcsv:
                csv.writer(fcsv).writerow([
                    epoch,
                    f"{epoch_loss:.6f}", f"{epoch_train_dice:.6f}",
                    ("" if val_loss_epoch is None else f"{val_loss_epoch:.6f}"),
                    ("" if val_dice_epoch is None else f"{val_dice_epoch:.6f}"),
                    f"{get_lr(optimizer):.6f}", f"{elapsed:.2f}"
                ])

            # Early Stopping Check
            if epochs_no_improve >= early_stop_patience:
                print(f"\n[Early Stopping] No improvement for {early_stop_patience} epochs. Stopping training.")
                break

        print(f"\nFold {CURR_FOLD} Complete | Best Epoch: {best_epoch} | Best Val Dice: {best_metric:.4f}")

        # ---------------------------------------------------------------------
        # 5. OOF Inference & Case-level Dice Recording
        # ---------------------------------------------------------------------
        print(f"\n{'=' * 50}")
        print(f"OOF Inference for Fold {CURR_FOLD}")
        print(f"{'=' * 50}")

        model.load_state_dict(torch.load(ckpt_path)["model"])
        model.eval()

        pred_dir = os.path.join(out_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)



        with torch.inference_mode():
            for val_data in val_loader:
                # 1. 准备数据
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device).long()

                # 2. 推理
                val_logits = sliding_window_inference(
                    val_images, patch_size, 4, model, overlap=0.5, mode="gaussian"
                )

                # 兼容列表输出
                if isinstance(val_logits, (list, tuple)):
                    val_logits = val_logits[0]

                # 3. 生成预测 mask (Argmax)
                val_pred = val_logits.argmax(dim=1, keepdim=True)
                val_pred = post_process(val_pred)

                # 4. 计算 Dice (用于记录)
                current_dice = _dice_from_labels(pred=val_pred, target=val_labels).item()

                # 提取 Subject ID
                try:
                    # 尝试从 batch 中直接提取，或从解包后的数据提取
                    if "subject" in val_data:
                        subj_id = val_data["subject"][0]
                    else:
                        subj_id = extract_subject_id(val_data["image_meta_dict"]["filename_or_obj"][0])
                except:
                    subj_id = f"Unknown_{count}"

                print(f"   {subj_id}: Dice={current_dice:.4f}")

                # 5. [关键修改] 保存图像
                # val_pred 是生成的纯 Tensor，没有元数据信息。
                # val_images 是 MetaTensor，包含 affine 等空间信息。
                # 我们将预测结果挂载到 MetaTensor 上，利用 MONAI 的 decollate_batch 自动处理元数据

                # 手动把预测结果塞进 batch 字典里，方便 decollate
                val_data["pred"] = val_pred

                # decollate_batch 会把 Batch(B=1) 拆成 List[Item]，并自动处理 MetaTensor 的 affine
                val_data_unbatched = decollate_batch(val_data)

                for item in val_data_unbatched:
                    # item["pred"] 现在是一个单样本的 MetaTensor (如果 MONAI 版本支持)
                    # 或者我们需要显式把 input 的 affine 赋给它

                    pred_to_save = item["pred"]

                    # 确保是 CPU 上的数据
                    if torch.is_tensor(pred_to_save):
                        pred_to_save = pred_to_save.cpu()

                    # 这里的关键：如果 pred_to_save 丢失了 meta 信息，
                    # 我们从同个样本的 input image (item["image"]) 获取
                    # SaveImage 允许直接传入 Tensor，只要它有 Meta 信息，或者显式传入 meta_dict

                    # 构建一个正确的 meta_dict (只包含 affine 即可满足 NIfTI 保存)
                    # item["image"] 应该是 MetaTensor，带有 affine
                    if hasattr(item["image"], "meta"):
                        save_meta = item["image"].meta
                    else:
                        # 降级方案：如果没有 meta 属性，尝试找 affine
                        save_meta = item.get("image_meta_dict", None)

                    # 执行保存
                    SaveImage(
                        output_dir=pred_dir,
                        output_postfix="pred",
                        separate_folder=False,
                        print_log=False
                    )(pred_to_save, save_meta)  # 传入 image 的 meta

                # 记录 CSV
                save_path = os.path.join(pred_dir, f"{subj_id}_pred.nii.gz")
                with open(global_dice_csv, "a", newline="") as f:
                    csv.writer(f).writerow([subj_id, CURR_FOLD, f"{current_dice:.6f}", save_path])