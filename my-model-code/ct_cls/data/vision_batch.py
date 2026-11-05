from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ct_cls.config import ExperimentConfig, ModelConfig


def _slice_to_pil(slice_tensor: torch.Tensor) -> Image.Image:
    """(C, H, W) float CT window -> PIL RGB image for Qwen image_processor."""
    arr = slice_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 255).numpy()
    return Image.fromarray(arr.astype(np.uint8))


def prepare_video_batch(
    videos: list[torch.Tensor],
    processor,
    *,
    do_sample_frames: bool = True,
    vision_mode: str = "video",
) -> dict[str, torch.Tensor | str | None]:
    """3D/video path: Qwen3-VL video_processor (temporal patch + spatial merge)."""
    processed = processor.video_processor(
        videos=videos,
        return_tensors="pt",
        do_sample_frames=do_sample_frames,
    )
    return {
        "vision_mode": vision_mode,
        "pixel_values": processed["pixel_values_videos"],
        "grid_thw": processed["video_grid_thw"],
        "slice_counts": None,
    }


def prepare_video_temporal_pool_batch(videos: list[torch.Tensor], processor) -> dict[str, torch.Tensor | str | None]:
    """3D video with do_sample_frames=False, then spatial-mean per temporal group in encoder."""
    return prepare_video_batch(
        videos,
        processor,
        do_sample_frames=False,
        vision_mode="video_temporal_pool",
    )


def prepare_slice_2d_batch(videos: list[torch.Tensor], processor) -> dict[str, torch.Tensor | list[int]]:
    """2D/slice path: each (C,H,W) slice through Qwen3-VL image_processor."""
    images: list[Image.Image] = []
    slice_counts: list[int] = []
    for video in videos:
        num_slices = int(video.shape[0])
        slice_counts.append(num_slices)
        for slice_idx in range(num_slices):
            images.append(_slice_to_pil(video[slice_idx]))

    processed = processor.image_processor(images=images, return_tensors="pt")
    return {
        "vision_mode": "slice_2d",
        "pixel_values": processed["pixel_values"],
        "grid_thw": processed["image_grid_thw"],
        "slice_counts": slice_counts,
    }


def prepare_vision_batch(
    videos: list[torch.Tensor],
    processor,
    cfg: ModelConfig,
) -> dict[str, torch.Tensor | list[int] | str]:
    mode = cfg.vision_mode
    if mode == "video":
        return prepare_video_batch(videos, processor, do_sample_frames=True, vision_mode="video")
    if mode == "video_temporal_pool":
        return prepare_video_temporal_pool_batch(videos, processor)
    if mode == "slice_2d":
        return prepare_slice_2d_batch(videos, processor)
    raise ValueError(f"Unsupported vision_mode: {mode}")


def run_model_forward(
    model: torch.nn.Module,
    videos: list[torch.Tensor] | None,
    processor,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    batch: dict | None = None,
) -> torch.Tensor:
    """Unified forward for Qwen3-VL / Qwen3.5 classification backends."""
    if batch is not None and batch.get("preprocessed"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        grid_thw = batch["grid_thw"].to(device, non_blocking=True)
        slice_counts = batch.get("slice_counts")
        return model(pixel_values, grid_thw, slice_counts)

    vision_inputs = prepare_vision_batch(videos, processor, cfg.model)
    pixel_values = vision_inputs["pixel_values"].to(device)
    grid_thw = vision_inputs["grid_thw"].to(device)
    slice_counts = vision_inputs.get("slice_counts")
    return model(pixel_values, grid_thw, slice_counts)
