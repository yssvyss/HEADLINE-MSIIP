from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ct_cls.config import CTPreprocessConfig, ExperimentConfig, ModelConfig
from ct_cls.data.ct_loader import load_ct_video, maybe_apply_ct_train_augment
from ct_cls.data.vision_batch import prepare_vision_batch
from ct_cls.data.worker_processor import get_worker_processor

logger = logging.getLogger(__name__)

# Transient NFS / truncated gzip reads on large 1mm volumes.
_LOAD_SKIP_EXCEPTIONS = (EOFError, OSError)


class CTClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        preprocess_cfg: CTPreprocessConfig,
        split: str | None = None,
        load_fn: Callable[[str, CTPreprocessConfig], torch.Tensor] | None = None,
        skip_on_load_error: bool = True,
        skip_log_path: str | Path | None = None,
        model_cfg: ModelConfig | None = None,
        preprocess_in_workers: bool = False,
    ) -> None:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self.label_names: list[str] = manifest["label_names"]
        samples = manifest["samples"]
        if split is not None:
            samples = [s for s in samples if s.get("split") == split]
        self.samples = samples
        self.preprocess_cfg = preprocess_cfg
        self.load_fn = load_fn or load_ct_video
        self.split = split
        self.skip_on_load_error = skip_on_load_error
        self.skip_log_path = Path(skip_log_path) if skip_log_path is not None else None
        self._bad_indices: set[int] = set()
        self._logged_skip_paths: set[str] = set()
        self.model_cfg = model_cfg
        self.preprocess_in_workers = preprocess_in_workers
        if preprocess_in_workers and model_cfg is None:
            raise ValueError("preprocess_in_workers requires model_cfg")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, sample: dict) -> dict:
        video = self.load_fn(sample["nii_path"], self.preprocess_cfg)
        if self.split == "train" and isinstance(self.preprocess_cfg, CTPreprocessConfig):
            video = maybe_apply_ct_train_augment(video, self.preprocess_cfg)
        labels = torch.tensor(sample["labels"], dtype=torch.float32)
        item = {
            "id": sample["id"],
            "nii_path": sample["nii_path"],
            "labels": labels,
        }
        if self.preprocess_in_workers:
            assert self.model_cfg is not None
            processor = get_worker_processor(self.model_cfg)
            vision_inputs = prepare_vision_batch([video], processor, self.model_cfg)
            item["pixel_values"] = vision_inputs["pixel_values"]
            item["grid_thw"] = vision_inputs["grid_thw"]
            item["slice_counts"] = vision_inputs["slice_counts"]
            item["vision_mode"] = vision_inputs["vision_mode"]
        else:
            item["video"] = video
        return item

    def _log_skip(self, sample: dict, exc: Exception) -> None:
        nii_path = sample["nii_path"]
        if nii_path in self._logged_skip_paths:
            return
        self._logged_skip_paths.add(nii_path)

        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "id": sample.get("id"),
            "nii_path": nii_path,
            "split": self.split,
            "error": repr(exc),
        }
        logger.warning(
            "Skipping unreadable NIfTI: id=%s split=%s path=%s | %s",
            sample.get("id"),
            self.split,
            nii_path,
            exc,
        )
        if self.skip_log_path is not None:
            self.skip_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.skip_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __getitem__(self, idx: int) -> dict:
        if not self.skip_on_load_error:
            return self._load_sample(self.samples[idx])

        n = len(self.samples)
        if n == 0:
            raise RuntimeError("Dataset has no samples")

        last_exc: Exception | None = None
        for offset in range(n):
            sample_idx = (idx + offset) % n
            if sample_idx in self._bad_indices:
                continue
            sample = self.samples[sample_idx]
            try:
                return self._load_sample(sample)
            except _LOAD_SKIP_EXCEPTIONS as exc:
                last_exc = exc
                self._bad_indices.add(sample_idx)
                self._log_skip(sample, exc)

        raise RuntimeError(
            f"All {n} samples in split={self.split!r} failed to load; "
            f"see skipped_nii.jsonl for details"
        ) from last_exc


def collate_fn(batch: list[dict]) -> dict:
    return {
        "ids": [item["id"] for item in batch],
        "nii_paths": [item["nii_path"] for item in batch],
        "videos": [item["video"] for item in batch],
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "preprocessed": False,
    }


def collate_preprocessed_fn(batch: list[dict]) -> dict:
    slice_counts: list[int] = []
    for item in batch:
        counts = item["slice_counts"]
        if counts is None:
            continue
        slice_counts.extend(counts)
    return {
        "ids": [item["id"] for item in batch],
        "nii_paths": [item["nii_path"] for item in batch],
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "grid_thw": torch.cat([item["grid_thw"] for item in batch], dim=0),
        "slice_counts": slice_counts or None,
        "vision_mode": batch[0]["vision_mode"],
        "preprocessed": True,
    }


def use_worker_preprocess(cfg: ExperimentConfig) -> bool:
    return bool(cfg.data.preprocess_in_workers)


def build_collate_fn(cfg: ExperimentConfig):
    if use_worker_preprocess(cfg):
        return collate_preprocessed_fn
    return collate_fn


def build_dataloader_kwargs(cfg: ExperimentConfig, *, shuffle: bool) -> dict:
    kwargs: dict = {
        "batch_size": cfg.train.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.data.num_workers,
        "collate_fn": build_collate_fn(cfg),
        "pin_memory": True,
    }
    if cfg.data.num_workers > 0:
        if cfg.data.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
        if cfg.data.persistent_workers:
            kwargs["persistent_workers"] = True
    return kwargs


def _dataset_loader(cfg: ExperimentConfig) -> tuple[CTPreprocessConfig, Callable]:
    if cfg.model.vision_backend not in {"qwen3_vl", "qwen3_5"}:
        raise ValueError(f"Unsupported vision_backend for inference: {cfg.model.vision_backend}")
    return cfg.ct, load_ct_video


def build_datasets(
    cfg: ExperimentConfig,
    val_split: str = "val",
    skip_log_path: str | Path | None = None,
) -> tuple[CTClassificationDataset, CTClassificationDataset | None]:
    preprocess_cfg, load_fn = _dataset_loader(cfg)
    worker_preprocess = use_worker_preprocess(cfg)
    model_cfg = cfg.model if worker_preprocess else None
    ds_common = {
        "preprocess_cfg": preprocess_cfg,
        "load_fn": load_fn,
        "skip_log_path": skip_log_path,
        "model_cfg": model_cfg,
        "preprocess_in_workers": worker_preprocess,
    }
    train_ds = CTClassificationDataset(
        cfg.data.manifest_path,
        split="train",
        **ds_common,
    )
    val_ds = CTClassificationDataset(
        cfg.data.manifest_path,
        split=val_split,
        **ds_common,
    )
    if len(val_ds) == 0:
        val_ds = None
    return train_ds, val_ds
