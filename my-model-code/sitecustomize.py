from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path


def _load_ct_vision_process():
    patch_path = Path(__file__).resolve().parent / "qwen_vl_vision_process_ct.py"
    spec = importlib.util.spec_from_file_location(
        "_headline_qwen_vl_vision_process_ct",
        patch_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load qwen-vl CT vision patch from {patch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_qwen_vl_utils() -> None:
    if os.environ.get("HEADLINE_PATCH_QWEN_VL", "1") == "0":
        return

    try:
        package = importlib.import_module("qwen_vl_utils")
        original = importlib.import_module("qwen_vl_utils.vision_process")
    except Exception as exc:
        print(f"[headline] qwen_vl_utils patch skipped: {exc}", file=sys.stderr)
        return

    patched = _load_ct_vision_process()

    # Replace the submodule object so later imports receive the CT-aware version.
    sys.modules["qwen_vl_utils.vision_process"] = patched
    setattr(package, "vision_process", patched)

    # qwen_vl_utils commonly re-exports these names from the package root.
    for name in (
        "fetch_video",
        "process_vision_info",
        "extract_vision_info",
        "fetch_image",
    ):
        if hasattr(patched, name):
            setattr(original, name, getattr(patched, name))
            setattr(package, name, getattr(patched, name))

    print("[headline] patched qwen_vl_utils.vision_process for NIfTI video input", file=sys.stderr)


_patch_qwen_vl_utils()
