from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline_io import read_json, write_json


MODEL_CODE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_DIR = MODEL_CODE_DIR / "classification"
CONFIG_DIR = CLASSIFICATION_DIR / "configs"
MODELS_JSON = CLASSIFICATION_DIR / "models.json"

if str(MODEL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_CODE_DIR))
if str(CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_DIR))

from classification.prediction_records import build_prediction_records  # noqa: E402
from ct_cls.config import load_config  # noqa: E402
from ct_cls.data.dataset import (  # noqa: E402
    CTClassificationDataset,
    _dataset_loader,
    build_dataloader_kwargs,
    use_worker_preprocess,
)
from ct_cls.data.vision_batch import run_model_forward  # noqa: E402
from ct_cls.models.classifier import build_model  # noqa: E402


def build_inference_manifest(base_dataset_path: Path, output_path: Path, label_names: list[str]) -> Path:
    samples = []
    dummy_labels = [0] * len(label_names)
    for sample in read_json(base_dataset_path):
        videos = sample.get("videos") or []
        if not videos:
            raise ValueError(f"base dataset sample has no videos: {sample.get('id')}")
        nii_path = videos[0]
        samples.append(
            {
                "id": sample["id"],
                "nii_path": nii_path,
                "labels": dummy_labels,
                "split": "infer",
            }
        )

    manifest = {
        "label_names": label_names,
        "meta": {
            "source_base_dataset": str(base_dataset_path),
            "split": "infer",
        },
        "samples": samples,
    }
    write_json(output_path, manifest)
    return output_path


def load_models() -> list[dict[str, Any]]:
    data = read_json(MODELS_JSON)
    models = data["models"]
    requested = os.environ.get("HEADLINE_MULTILABEL_MODEL_IDS", "").strip()
    if not requested:
        return models

    keep = {item.strip() for item in requested.split(",") if item.strip()}
    selected = [model for model in models if model["id"] in keep]
    missing = keep - {model["id"] for model in selected}
    if missing:
        raise ValueError(f"Unknown HEADLINE_MULTILABEL_MODEL_IDS: {sorted(missing)}")
    return selected


def checkpoint_dir_for(model_id: str, model_path: Path) -> Path:
    root = Path(
        os.environ.get(
            "HEADLINE_MULTILABEL_CHECKPOINT_ROOT",
            model_path / "multilabel" / "checkpoints",
        )
    )
    return root / model_id


def checkpoint_meta_path(model_id: str, checkpoint_dir: Path) -> Path:
    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        return config_path
    raise FileNotFoundError(
        f"Missing config.json for multilabel model {model_id}: "
        f"expected {config_path}. Put it beside model.pt in the Model tarball."
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    checkpoint_path = checkpoint_dir / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing multilabel checkpoint: {checkpoint_path}. "
            "Expected weights under /opt/ml/model/multilabel/checkpoints/<model_id>/model.pt"
        )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.vision_encoder.load_state_dict(state["vision_encoder"], strict=True)
    model.head.load_state_dict(state["head"], strict=True)


def resolve_backbone_path(original_model_path: str, model_path: Path) -> str:
    is_qwen35 = "Qwen3.5" in original_model_path
    env_key = "HEADLINE_CLS_QWEN35_PATH" if is_qwen35 else "HEADLINE_CLS_QWEN3VL_PATH"
    if env_key in os.environ:
        return os.environ[env_key]

    if is_qwen35:
        return str(model_path / "vlm")

    backbone_root = Path(
        os.environ.get(
            "HEADLINE_MULTILABEL_BACKBONE_ROOT",
            model_path / "multilabel" / "backbones",
        )
    )
    return str(backbone_root / Path(original_model_path).name)


def cuda_total_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return float(props.total_memory) / 1024**3


def default_multilabel_batch_size() -> str:
    total_gb = cuda_total_memory_gb()
    if total_gb is not None and total_gb >= 14:
        return "2"
    return "1"


def patch_config_for_runtime(cfg: Any, model_record: dict[str, Any], manifest_path: Path, model_path: Path) -> Any:
    cfg.data.manifest_path = str(manifest_path)
    cfg.data.num_workers = int(os.environ.get("HEADLINE_MULTILABEL_NUM_WORKERS", "0"))
    cfg.data.prefetch_factor = None if cfg.data.num_workers == 0 else cfg.data.prefetch_factor
    cfg.data.persistent_workers = False if cfg.data.num_workers == 0 else cfg.data.persistent_workers
    cfg.data.preprocess_in_workers = os.environ.get("HEADLINE_MULTILABEL_PREPROCESS_IN_WORKERS", "0") == "1"
    cfg.train.batch_size = int(os.environ.get("HEADLINE_MULTILABEL_BATCH_SIZE", default_multilabel_batch_size()))
    cfg.train.cuda_visible_devices = None
    cfg.model.model_path = resolve_backbone_path(cfg.model.model_path, model_path)
    if os.environ.get("HEADLINE_MULTILABEL_ATTN_IMPL"):
        cfg.model.attn_impl = os.environ["HEADLINE_MULTILABEL_ATTN_IMPL"]
    elif cfg.model.attn_impl == "flash_attention_2" and os.environ.get("HEADLINE_DISABLE_FLASH_ATTN", "1") == "1":
        cfg.model.attn_impl = "sdpa"
    # The final runtime target is NVIDIA T4. Keep every Step2 model on fp16;
    # T4 does not provide native bf16 tensor-core acceleration.
    cfg.model.torch_dtype = "float16"
    print(
        f"  model={model_record['id']} backbone={cfg.model.model_path} "
        f"batch_size={cfg.train.batch_size} workers={cfg.data.num_workers} "
        f"attn={cfg.model.attn_impl} dtype={cfg.model.torch_dtype}"
    )
    return cfg


def run_single_model(
    model_record: dict[str, Any],
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    threshold: float,
) -> Path:
    model_id = model_record["id"]
    cfg_path = CONFIG_DIR / Path(model_record["config"]).name
    ckpt_dir = checkpoint_dir_for(model_id, model_path)
    meta_path = checkpoint_meta_path(model_id, ckpt_dir)

    cfg = patch_config_for_runtime(
        load_config(cfg_path),
        model_record=model_record,
        manifest_path=manifest_path,
        model_path=model_path,
    )
    ckpt_meta = read_json(meta_path)
    label_names = ckpt_meta["label_names"]

    preprocess_cfg, load_fn = _dataset_loader(cfg)
    worker_preprocess = use_worker_preprocess(cfg)
    dataset = CTClassificationDataset(
        cfg.data.manifest_path,
        preprocess_cfg,
        split="infer",
        load_fn=load_fn,
        skip_log_path=output_dir / "skipped_nii.jsonl",
        model_cfg=cfg.model if worker_preprocess else None,
        preprocess_in_workers=worker_preprocess,
    )
    dataloader = DataLoader(dataset, **build_dataloader_kwargs(cfg, shuffle=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = build_model(cfg.model, num_labels=len(label_names))
    load_checkpoint(model, ckpt_dir)
    model.to(device)
    model.eval()

    all_ids: list[str] = []
    all_nii_paths: list[str] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            logits = run_model_forward(model, batch.get("videos"), processor, cfg, device, batch=batch)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_ids.extend(batch["ids"])
            all_nii_paths.extend(batch["nii_paths"])
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    predictions = build_prediction_records(
        ids=all_ids,
        nii_paths=all_nii_paths,
        label_names=label_names,
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )

    pred_json_path = output_dir / "predictions.json"
    write_json(pred_json_path, predictions)

    del model, processor, dataloader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_json_path


def align_by_id(preds: list[dict], ids: list[str], source: str) -> list[dict]:
    order = {rec["id"]: rec for rec in preds}
    missing = [sid for sid in ids if sid not in order]
    if missing:
        raise ValueError(f"missing {len(missing)} ids in {source}: {missing[:3]}")
    return [order[sid] for sid in ids]


def ensemble_predictions(model_outputs: list[Path], template_preds: list[dict], threshold: float) -> list[dict]:
    ids = [rec["id"] for rec in template_preds]
    label_names = list(template_preds[0]["labels"].keys())
    prob_sum = np.zeros((len(ids), len(label_names)), dtype=np.float64)

    for path in model_outputs:
        preds = align_by_id(read_json(path), ids, str(path))
        for i, rec in enumerate(preds):
            if list(rec["labels"].keys()) != label_names:
                raise ValueError(f"label order mismatch in {path}")
            for j, name in enumerate(label_names):
                prob_sum[i, j] += float(rec["predictions"][name])

    mean_prob = prob_sum / len(model_outputs)
    pred_binary = (mean_prob >= threshold).astype(np.int32)
    output = []
    for i, rec in enumerate(template_preds):
        output.append(
            {
                "id": rec["id"],
                "nii_path": rec["nii_path"],
                "labels": rec["labels"],
                "predictions": {name: float(mean_prob[i, j]) for j, name in enumerate(label_names)},
                "pred_binary": {name: int(pred_binary[i, j]) for j, name in enumerate(label_names)},
            }
        )
    return output


def run_multilabel_step(base_dataset_path: Path, work_dir: Path, model_path: Path | None = None) -> Path:
    print("Step 2: multi-label classifier ensemble")
    model_path = model_path or Path("/opt/ml/model")
    output_path = work_dir / "multilabel_predictions.json"
    output_root = work_dir / "multilabel"
    threshold = float(os.environ.get("HEADLINE_MULTILABEL_THRESHOLD", "0.57"))

    models = load_models()
    if not models:
        raise RuntimeError("No multilabel models configured")

    first_ckpt_meta = read_json(checkpoint_meta_path(models[0]["id"], checkpoint_dir_for(models[0]["id"], model_path)))
    manifest_path = build_inference_manifest(
        base_dataset_path=base_dataset_path,
        output_path=output_root / "manifest.json",
        label_names=first_ckpt_meta["label_names"],
    )

    model_outputs = []
    for model_record in models:
        model_id = model_record["id"]
        print(f"  running multilabel model: {model_id}")
        model_output = run_single_model(
            model_record=model_record,
            manifest_path=manifest_path,
            model_path=model_path,
            output_dir=output_root / "models" / model_id,
            threshold=threshold,
        )
        model_outputs.append(model_output)

    template_model_id = os.environ.get("HEADLINE_MULTILABEL_TEMPLATE_MODEL", "query_attn_qwen35")
    template_index = next(
        (idx for idx, model_record in enumerate(models) if model_record["id"] == template_model_id),
        0,
    )
    template_preds = read_json(model_outputs[template_index])
    ensemble = ensemble_predictions(model_outputs, template_preds, threshold=threshold)
    write_json(output_path, ensemble)
    print(f"  wrote ensemble predictions: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HEADLINE Step2 classifier ensemble.")
    parser.add_argument("--base-dataset-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_multilabel_step(
        base_dataset_path=args.base_dataset_path,
        work_dir=args.work_dir,
        model_path=args.model_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
