from __future__ import annotations

from typing import Any

import torch

from ct_cls.config import ModelConfig

_PROCESSOR_CACHE: dict[int, Any] = {}


def get_worker_processor(model_cfg: ModelConfig):
    """Lazy-load HuggingFace processor once per DataLoader worker process."""
    worker_info = torch.utils.data.get_worker_info()
    cache_key = worker_info.id if worker_info is not None else -1
    if cache_key not in _PROCESSOR_CACHE:
        from transformers import AutoProcessor

        _PROCESSOR_CACHE[cache_key] = AutoProcessor.from_pretrained(
            model_cfg.model_path,
            local_files_only=True,
        )
    return _PROCESSOR_CACHE[cache_key]
