from __future__ import annotations

import os


def normalize_cuda_visible_devices(value: str | int | list[int] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    text = str(value).strip()
    return text if text else None


def setup_cuda_visible_devices(value: str | int | list[int] | None) -> str | None:
    """Set CUDA_VISIBLE_DEVICES before any CUDA initialization."""
    devices = normalize_cuda_visible_devices(value)
    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    return devices
