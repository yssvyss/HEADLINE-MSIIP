from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

from ct_cls.config import ModelConfig


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def vision_token_dim(vision_encoder: nn.Module) -> int:
    """Token dim after vision merger (= out_hidden_size, NOT internal hidden_size)."""
    vision_cfg = vision_encoder.config
    token_dim = getattr(vision_cfg, "out_hidden_size", None)
    if token_dim is None:
        token_dim = vision_cfg.hidden_size
    return int(token_dim)


def _load_vision_state_dict(model_path: Path) -> dict[str, torch.Tensor] | None:
    vision_weights = model_path / "vision_encoder.safetensors"
    if not vision_weights.is_file():
        return None
    return load_safetensors(str(vision_weights), device="cpu")


def _freeze_if_needed(vision_encoder: nn.Module, cfg: ModelConfig) -> None:
    if cfg.freeze_vision:
        for param in vision_encoder.parameters():
            param.requires_grad = False


def _load_qwen3_vl_vision_only(
    cfg: ModelConfig,
) -> tuple[nn.Module, Any, int, torch.dtype] | None:
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    model_path = Path(cfg.model_path)
    state_dict = _load_vision_state_dict(model_path)
    if state_dict is None:
        return None

    vision_config_path = model_path / "vision_config.json"
    if vision_config_path.is_file():
        with open(vision_config_path, encoding="utf-8") as f:
            vision_config = Qwen3VLVisionConfig(**json.load(f))
    else:
        full_config = Qwen3VLConfig.from_pretrained(cfg.model_path, local_files_only=True)
        vision_config = full_config.vision_config
    vision_config._attn_implementation = cfg.attn_impl

    dtype = resolve_dtype(cfg.torch_dtype)
    vision_encoder = Qwen3VLVisionModel(vision_config)
    vision_encoder.load_state_dict(state_dict, strict=True)
    vision_encoder.to(dtype=dtype)
    token_dim = vision_token_dim(vision_encoder)
    processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True)
    _freeze_if_needed(vision_encoder, cfg)
    return vision_encoder, processor, token_dim, dtype


def _load_qwen3_5_vision_only(
    cfg: ModelConfig,
) -> tuple[nn.Module, Any, int, torch.dtype] | None:
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    model_path = Path(cfg.model_path)
    state_dict = _load_vision_state_dict(model_path)
    if state_dict is None:
        return None

    vision_config_path = model_path / "vision_config.json"
    if vision_config_path.is_file():
        with open(vision_config_path, encoding="utf-8") as f:
            vision_config = Qwen3_5VisionConfig(**json.load(f))
    else:
        full_config = Qwen3_5Config.from_pretrained(cfg.model_path, local_files_only=True)
        vision_config = full_config.vision_config
    vision_config._attn_implementation = cfg.attn_impl

    dtype = resolve_dtype(cfg.torch_dtype)
    vision_encoder = Qwen3_5VisionModel(vision_config)
    vision_encoder.load_state_dict(state_dict, strict=True)
    vision_encoder.to(dtype=dtype)
    token_dim = vision_token_dim(vision_encoder)
    processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True)
    _freeze_if_needed(vision_encoder, cfg)
    return vision_encoder, processor, token_dim, dtype


def load_qwen3_vl_vision(cfg: ModelConfig) -> tuple[nn.Module, Any, int, torch.dtype]:
    """Load only Qwen3-VL vision encoder + processor (no LLM)."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    vision_only = _load_qwen3_vl_vision_only(cfg)
    if vision_only is not None:
        return vision_only

    dtype = resolve_dtype(cfg.torch_dtype)
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        attn_implementation=cfg.attn_impl,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    vision_encoder = full_model.model.visual
    token_dim = vision_token_dim(vision_encoder)
    processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True)
    _freeze_if_needed(vision_encoder, cfg)

    del full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vision_encoder, processor, token_dim, dtype


def load_qwen3_5_vision(cfg: ModelConfig) -> tuple[nn.Module, Any, int, torch.dtype]:
    """Load only Qwen3.5 vision encoder + processor (no LLM)."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    vision_only = _load_qwen3_5_vision_only(cfg)
    if vision_only is not None:
        return vision_only

    dtype = resolve_dtype(cfg.torch_dtype)
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        attn_implementation=cfg.attn_impl,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    vision_encoder = full_model.model.visual
    token_dim = vision_token_dim(vision_encoder)
    processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True)
    _freeze_if_needed(vision_encoder, cfg)

    del full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vision_encoder, processor, token_dim, dtype


def load_qwen_vision(cfg: ModelConfig) -> tuple[nn.Module, Any, int, torch.dtype]:
    """Dispatch Qwen vision loading by ``vision_backend``."""
    if cfg.vision_backend == "qwen3_vl":
        return load_qwen3_vl_vision(cfg)
    if cfg.vision_backend == "qwen3_5":
        return load_qwen3_5_vision(cfg)
    raise ValueError(
        f"Unsupported Qwen vision_backend: {cfg.vision_backend!r}. "
        "Use qwen3_vl or qwen3_5."
    )


def encode_video_tokens(
    vision_encoder: nn.Module,
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Qwen3-VL vision tower and pack per-sample tokens as (B, N, D)."""
    pixel_values = pixel_values_videos.type(vision_encoder.dtype)
    vision_output = vision_encoder(
        pixel_values,
        grid_thw=video_grid_thw,
        return_dict=True,
    )
    token_tensor = vision_output.pooler_output
    split_sizes = (video_grid_thw.prod(-1) // vision_encoder.spatial_merge_size**2).tolist()
    token_groups = torch.split(token_tensor, split_sizes)

    batch_size = len(token_groups)
    max_len = max(group.shape[0] for group in token_groups)
    token_dim = token_groups[0].shape[-1]
    device = token_tensor.device
    dtype = token_tensor.dtype

    padded = torch.zeros(batch_size, max_len, token_dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.bool)
    for idx, group in enumerate(token_groups):
        num_tokens = group.shape[0]
        padded[idx, :num_tokens] = group
        mask[idx, :num_tokens] = True
    return padded, mask


def encode_video_temporal_pool_tokens(
    vision_encoder: nn.Module,
    pixel_values: torch.Tensor,
    video_grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D video encode (with spatial merge), then spatial-mean per temporal group -> (B, T', D)."""
    pixel_values = pixel_values.type(vision_encoder.dtype)
    vision_output = vision_encoder(
        pixel_values,
        grid_thw=video_grid_thw,
        return_dict=True,
    )
    token_tensor = vision_output.pooler_output
    merge = vision_encoder.spatial_merge_size
    merge_factor = merge**2

    split_sizes = (video_grid_thw.prod(-1) // merge_factor).tolist()
    token_groups = torch.split(token_tensor, split_sizes)

    batch_size = len(token_groups)
    token_dim = token_groups[0].shape[-1]
    device = token_tensor.device
    dtype = token_tensor.dtype

    pooled_groups: list[torch.Tensor] = []
    temporal_lengths: list[int] = []
    for batch_idx, group in enumerate(token_groups):
        grid_t, grid_h, grid_w = video_grid_thw[batch_idx].tolist()
        h_merged = grid_h // merge
        w_merged = grid_w // merge
        expected = grid_t * h_merged * w_merged
        if group.shape[0] != expected:
            raise ValueError(
                f"token count mismatch for sample {batch_idx}: "
                f"got {group.shape[0]}, expected {expected} "
                f"(grid_thw={[grid_t, grid_h, grid_w]}, merge={merge})"
            )
        grid = group.view(grid_t, h_merged, w_merged, token_dim)
        temporal_tokens = grid.mean(dim=(1, 2))
        pooled_groups.append(temporal_tokens)
        temporal_lengths.append(grid_t)

    max_temporal = max(temporal_lengths)
    padded = torch.zeros(batch_size, max_temporal, token_dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_temporal, device=device, dtype=torch.bool)
    for batch_idx, temporal_tokens in enumerate(pooled_groups):
        t_len = temporal_tokens.shape[0]
        padded[batch_idx, :t_len] = temporal_tokens
        mask[batch_idx, :t_len] = True
    return padded, mask


def _pool_image_tokens(token_group: torch.Tensor, slice_token_pool: str) -> torch.Tensor:
    if slice_token_pool in {"cls", "first"}:
        return token_group[0]
    if slice_token_pool == "mean":
        return token_group.mean(dim=0)
    raise ValueError(f"Unsupported slice_token_pool: {slice_token_pool}")


def encode_slice_2d_tokens(
    vision_encoder: nn.Module,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    slice_counts: list[int],
    slice_token_pool: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode each 2D slice -> 1 token; pack as (B, T, D) where T = slice_counts[b]."""
    pixel_values = pixel_values.type(vision_encoder.dtype)
    vision_output = vision_encoder(
        pixel_values,
        grid_thw=image_grid_thw,
        return_dict=True,
    )
    token_tensor = vision_output.pooler_output
    merge_factor = vision_encoder.spatial_merge_size**2
    split_sizes = (image_grid_thw.prod(-1) // merge_factor).tolist()
    token_groups = torch.split(token_tensor, split_sizes)

    if len(token_groups) != sum(slice_counts):
        raise ValueError(
            f"slice count mismatch: got {len(token_groups)} image groups, "
            f"expected {sum(slice_counts)} from slice_counts={slice_counts}"
        )

    per_slice_tokens = [_pool_image_tokens(group, slice_token_pool) for group in token_groups]

    batch_size = len(slice_counts)
    max_slices = max(slice_counts)
    token_dim = per_slice_tokens[0].shape[-1]
    device = token_tensor.device
    dtype = token_tensor.dtype

    padded = torch.zeros(batch_size, max_slices, token_dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_slices, device=device, dtype=torch.bool)

    offset = 0
    for batch_idx, num_slices in enumerate(slice_counts):
        sample_tokens = torch.stack(per_slice_tokens[offset : offset + num_slices], dim=0)
        padded[batch_idx, :num_slices] = sample_tokens
        mask[batch_idx, :num_slices] = True
        offset += num_slices

    return padded, mask
