"""3D CT loading — logic aligned with HEADLINE-patched qwen_vl_utils/vision_process.py.

Parameters come directly from YAML (CTPreprocessConfig), not environment variables,
so behaviour is consistent even when the installed qwen_vl_utils build hardcodes
T=16 / H=W=512 (e.g. cls conda env).
"""

from __future__ import annotations

import logging
import random
import time

import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from ct_cls.config import CTPreprocessConfig

logger = logging.getLogger(__name__)

# 1mm volumes are large; NFS reads under DataLoader workers can fail transiently.
_NII_LOAD_RETRIES = 3
_NII_LOAD_RETRY_DELAY_SEC = 2.0


def normalize_ct_window(volume: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    hu_min = float(window_center) - float(window_width) / 2.0
    hu_max = float(window_center) + float(window_width) / 2.0
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / max(hu_max - hu_min, 1e-6)
    return (volume * 255.0).astype(np.float32)


def build_ct_video_tensor(image3d: np.ndarray, windows: list[tuple[float, float]]) -> torch.Tensor:
    if windows:
        channels = [normalize_ct_window(image3d, center, width) for center, width in windows]
        ct_tensor = torch.from_numpy(np.stack(channels, axis=0))
        return ct_tensor.permute(3, 0, 2, 1).contiguous()

    base_tensor = torch.from_numpy(image3d).permute(2, 1, 0).unsqueeze(1).float()
    return base_tensor.repeat(1, 3, 1, 1).contiguous()


def center_weighted_frame_indices(
    total_frames: int,
    target_frames: int,
    sigma_ratio: float = 0.22,
) -> np.ndarray:
    if total_frames <= target_frames:
        return np.arange(total_frames, dtype=np.int64)

    positions = np.arange(total_frames, dtype=np.float64)
    center = (total_frames - 1) / 2.0
    sigma = max(total_frames * sigma_ratio, 1.0)
    weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
    weights /= weights.sum()
    sampled = np.random.choice(total_frames, size=target_frames, replace=False, p=weights)
    sampled.sort()
    return sampled.astype(np.int64)


def temporal_resize(video: torch.Tensor, target_t: int, method: str = "linear") -> torch.Tensor:
    t, c, h, w = video.shape
    if t == target_t:
        return video
    if t > target_t and method == "sample":
        idx = torch.linspace(0, t - 1, target_t).round().long()
        return video.index_select(0, idx)
    if method == "linear":
        video = video.permute(1, 2, 3, 0).contiguous()
        video = video.view(-1, 1, t)
        video = torch.nn.functional.interpolate(video, size=target_t, mode="linear", align_corners=False)
        return video.view(c, h, w, target_t).permute(3, 0, 1, 2).contiguous()
    raise ValueError(f"Unsupported temporal resize method: {method}")


def random_crop_resize_hw(
    video: torch.Tensor,
    target_h: int,
    target_w: int,
    scale_min: float,
    scale_max: float = 1.0,
) -> torch.Tensor:
    """Random crop on H/W (all T,C slices share the same box), then resize back."""
    if scale_min <= 0 or scale_max <= 0 or scale_min > scale_max:
        raise ValueError(f"invalid crop scale range: [{scale_min}, {scale_max}]")

    _, _, height, width = video.shape
    if height != target_h or width != target_w:
        raise ValueError(
            f"expected spatial size ({target_h}, {target_w}), got ({height}, {width})"
        )

    scale = random.uniform(scale_min, scale_max)
    crop_h = min(height, max(1, int(round(height * scale))))
    crop_w = min(width, max(1, int(round(width * scale))))
    if crop_h == height and crop_w == width:
        return video

    top = random.randint(0, height - crop_h)
    left = random.randint(0, width - crop_w)
    cropped = video[:, :, top : top + crop_h, left : left + crop_w]
    return F.resize(
        cropped,
        size=[target_h, target_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).contiguous()


def maybe_apply_ct_train_augment(video: torch.Tensor, cfg: CTPreprocessConfig) -> torch.Tensor:
    if cfg.random_crop_scale_min is None:
        return video

    target_h, target_w, _ = cfg.input_size
    return random_crop_resize_hw(
        video,
        target_h=target_h,
        target_w=target_w,
        scale_min=float(cfg.random_crop_scale_min),
        scale_max=float(cfg.random_crop_scale_max),
    )


def _load_nii_volume(nii_path: str) -> np.ndarray:
    last_exc: Exception | None = None
    for attempt in range(1, _NII_LOAD_RETRIES + 1):
        try:
            return nib.load(nii_path).get_fdata(dtype=np.float32)
        except (EOFError, OSError) as exc:
            last_exc = exc
            if attempt < _NII_LOAD_RETRIES:
                logger.warning(
                    "NIfTI load failed (attempt %d/%d): %s | %s",
                    attempt,
                    _NII_LOAD_RETRIES,
                    nii_path,
                    exc,
                )
                time.sleep(_NII_LOAD_RETRY_DELAY_SEC * attempt)
    raise EOFError(
        f"Failed to read NIfTI after {_NII_LOAD_RETRIES} attempts: {nii_path}"
    ) from last_exc


def load_ct_video(nii_path: str, cfg: CTPreprocessConfig) -> torch.Tensor:
    """Load .nii.gz and return video tensor shaped (T, C, H, W)."""
    image3d = _load_nii_volume(nii_path)
    video = build_ct_video_tensor(image3d, cfg.windows)

    image_size_h, image_size_w, video_size_t = cfg.input_size
    total_num_frames = video.shape[0]

    if total_num_frames > video_size_t:
        if cfg.sampling_method == "center_weighted":
            sampled_indices = center_weighted_frame_indices(
                total_frames=total_num_frames,
                target_frames=video_size_t,
                sigma_ratio=cfg.sampling_sigma_ratio,
            )
            video = video.index_select(0, torch.from_numpy(sampled_indices).long())
        elif cfg.sampling_method == "uniform":
            sampled_indices = torch.linspace(0, total_num_frames - 1, video_size_t).round().long()
            video = video.index_select(0, sampled_indices)
        else:
            raise ValueError(f"Unsupported sampling_method: {cfg.sampling_method}")
    elif video.shape[0] < video_size_t:
        video = temporal_resize(video, target_t=video_size_t, method="linear")

    video = F.resize(
        img=video,
        size=[image_size_h, image_size_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return video.contiguous()


def trace_ct_shapes(nii_path: str, cfg: CTPreprocessConfig) -> list[tuple[str, tuple[int, ...]]]:
    """Return ordered (step, shape) pairs for debugging preprocessing."""
    steps: list[tuple[str, tuple[int, ...]]] = []
    image3d = _load_nii_volume(nii_path)
    steps.append(("nii.get_fdata (X,Y,Z)", tuple(image3d.shape)))

    video = build_ct_video_tensor(image3d, cfg.windows)
    steps.append(("after multi-window + permute (T,C,H,W)", tuple(video.shape)))

    image_size_h, image_size_w, video_size_t = cfg.input_size
    total_num_frames = video.shape[0]
    steps.append(("config input_size (H,W,T)", (image_size_h, image_size_w, video_size_t)))

    if total_num_frames > video_size_t:
        if cfg.sampling_method == "center_weighted":
            sampled_indices = center_weighted_frame_indices(
                total_frames=total_num_frames,
                target_frames=video_size_t,
                sigma_ratio=cfg.sampling_sigma_ratio,
            )
            video = video.index_select(0, torch.from_numpy(sampled_indices).long())
        else:
            sampled_indices = torch.linspace(0, total_num_frames - 1, video_size_t).round().long()
            video = video.index_select(0, sampled_indices)
        steps.append((f"after temporal sampling -> T={video_size_t}", tuple(video.shape)))
    elif video.shape[0] < video_size_t:
        video = temporal_resize(video, target_t=video_size_t, method="linear")
        steps.append((f"after temporal upsample -> T={video_size_t}", tuple(video.shape)))
    else:
        steps.append(("temporal dim already equals target T", tuple(video.shape)))

    video = F.resize(video, size=[image_size_h, image_size_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    steps.append(("after spatial resize (final fetch_video output)", tuple(video.shape)))
    return steps
