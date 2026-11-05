from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ct_cls.utils.device import normalize_cuda_visible_devices


@dataclass
class CTPreprocessConfig:
    windows: list[tuple[float, float]] = field(
        default_factory=lambda: [(40.0, 80.0), (40.0, 120.0), (40.0, 160.0)]
    )
    input_size: tuple[int, int, int] = (512, 512, 64)
    sampling_method: str = "center_weighted"
    sampling_sigma_ratio: float = 0.22
    # Train-only: random crop H/W to [scale_min, scale_max] * input_size, then resize back.
    random_crop_scale_min: float | None = None
    random_crop_scale_max: float = 1.0


@dataclass
class DataConfig:
    manifest_path: str = ""
    id_column: str = "id"
    num_workers: int = 4
    # Run Qwen image/video processor inside DataLoader workers (opt-in speedup).
    preprocess_in_workers: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False


@dataclass
class ModelConfig:
    # qwen3_vl | qwen3_5
    vision_backend: str = "qwen3_vl"
    model_path: str = "/opt/ml/model/multilabel/backbones/qwen3-vl-4b"
    freeze_vision: bool = False
    torch_dtype: str = "float16"
    attn_impl: str = "flash_attention_2"
    # mean_pool: masked mean -> Linear
    # mlp_pool: masked mean -> MLP -> logits
    # query_attention: 可学习 query token 对 vision token 做 cross-attn，每个 query 单独分类头
    head_type: str = "mean_pool"
    head_mlp_hidden_size: int | None = None
    head_dropout: float = 0.1
    # video: 3D video_processor，merge 后全部 token（约 512 @ T=32）
    # slice_2d: 每切片 image_processor，每片 1 token（共 T 个）
    # video_temporal_pool: 3D video + merge 后按 temporal 格点对 spatial mean（T/2 个 token @ T=32）
    vision_mode: str = "video"
    # slice_2d 时如何把单张切片的 N 个 vision token 压成 1 个：mean | cls（无 CLS 时用 index 0）
    slice_token_pool: str = "mean"
    num_queries: int | None = None
    query_num_layers: int = 2
    query_num_heads: int = 8
    query_dropout: float = 0.1


@dataclass
class TrainConfig:
    output_dir: str = "./outputs/experiment"
    num_epochs: float = 3  # supports fractions, e.g. 0.01 = 1% of one epoch then save/val
    batch_size: int = 2
    grad_accum_steps: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42
    log_steps: int = 10
    eval_steps: int | None = 200  # null = only validate at end of each epoch
    save_steps: int | None = 500  # null = only save at end of each epoch
    pos_weight: list[float] | None = None
    loss_type: str = "bce"  # bce | focal
    focal_gamma: float = 2.0
    focal_alpha: float | str | None = 0.25  # scalar, null, or "auto" from train label counts
    cuda_visible_devices: str | None = None  # e.g. "0" or "1,2,3"
    deepspeed_config: str | None = None  # path to DeepSpeed JSON; enables ZeRO multi-GPU via accelerate


@dataclass
class ExperimentConfig:
    experiment_name: str = "finding_top10"
    data: DataConfig = field(default_factory=DataConfig)
    ct: CTPreprocessConfig = field(default_factory=CTPreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _parse_windows(raw: list[str] | list[list[float]]) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, str):
            center, width = item.split(",")
            windows.append((float(center), float(width)))
        else:
            windows.append((float(item[0]), float(item[1])))
    return windows


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    ct_raw = raw.get("ct", {})
    ct = CTPreprocessConfig(
        windows=_parse_windows(ct_raw.get("windows", ["40,80", "40,120", "40,160"])),
        input_size=tuple(ct_raw.get("input_size", [512, 512, 64])),
        sampling_method=ct_raw.get("sampling_method", "center_weighted"),
        sampling_sigma_ratio=float(ct_raw.get("sampling_sigma_ratio", 0.22)),
        random_crop_scale_min=ct_raw.get("random_crop_scale_min"),
        random_crop_scale_max=float(ct_raw.get("random_crop_scale_max", 1.0)),
    )

    data_raw = raw.get("data", {})
    data = DataConfig(
        manifest_path=data_raw["manifest_path"],
        id_column=data_raw.get("id_column", "id"),
        num_workers=int(data_raw.get("num_workers", 4)),
        preprocess_in_workers=bool(data_raw.get("preprocess_in_workers", False)),
        prefetch_factor=data_raw.get("prefetch_factor"),
        persistent_workers=bool(data_raw.get("persistent_workers", False)),
    )

    model_raw = raw.get("model", {})
    model = ModelConfig(
        vision_backend=model_raw.get("vision_backend", "qwen3_vl"),
        model_path=model_raw["model_path"],
        freeze_vision=bool(model_raw.get("freeze_vision", False)),
        torch_dtype=model_raw.get("torch_dtype", "float16"),
        attn_impl=model_raw.get("attn_impl", "flash_attention_2"),
        head_type=model_raw.get("head_type", "mean_pool"),
        head_mlp_hidden_size=model_raw.get("head_mlp_hidden_size"),
        head_dropout=float(model_raw.get("head_dropout", 0.1)),
        vision_mode=model_raw.get("vision_mode", "video"),
        slice_token_pool=model_raw.get("slice_token_pool", "mean"),
        num_queries=model_raw.get("num_queries"),
        query_num_layers=int(model_raw.get("query_num_layers", 2)),
        query_num_heads=int(model_raw.get("query_num_heads", 8)),
        query_dropout=float(model_raw.get("query_dropout", 0.1)),
    )

    train_raw = raw.get("train", {})
    train = TrainConfig(
        output_dir=train_raw.get("output_dir", "./outputs/experiment"),
        num_epochs=float(train_raw.get("num_epochs", 3)),
        batch_size=int(train_raw.get("batch_size", 2)),
        grad_accum_steps=int(train_raw.get("grad_accum_steps", 3)),
        learning_rate=float(train_raw.get("learning_rate", 2e-5)),
        weight_decay=float(train_raw.get("weight_decay", 0.01)),
        warmup_ratio=float(train_raw.get("warmup_ratio", 0.05)),
        max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
        seed=int(train_raw.get("seed", 42)),
        log_steps=int(train_raw.get("log_steps", 10)),
        eval_steps=_optional_int(train_raw.get("eval_steps", 200), 200),
        save_steps=_optional_int(train_raw.get("save_steps", 500), 500),
        pos_weight=train_raw.get("pos_weight"),
        loss_type=train_raw.get("loss_type", "bce"),
        focal_gamma=float(train_raw.get("focal_gamma", 2.0)),
        focal_alpha=train_raw.get("focal_alpha", 0.25),
        cuda_visible_devices=normalize_cuda_visible_devices(train_raw.get("cuda_visible_devices")),
        deepspeed_config=train_raw.get("deepspeed_config"),
    )

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "experiment"),
        data=data,
        ct=ct,
        model=model,
        train=train,
    )
