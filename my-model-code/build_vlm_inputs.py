from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pipeline_io import read_json, write_json


LABEL_INFO_PATH = Path(__file__).resolve().parent / "label_info.json"


def load_label_info(path: Path) -> tuple[list[str], list[str]]:
    info = read_json(path)
    finding_labels = [item["label"] for item in info["finding_labels"]]
    diagnosis_labels = [item["label"] for item in info["diagnosis_labels"]]
    return finding_labels, diagnosis_labels


def load_pred_labels(
    pred_json: Path,
    finding_names: list[str],
    diagnosis_names: list[str],
) -> dict[str, dict[str, list[str]]]:
    raw = pred_json.read_bytes()
    start = raw.find(b"[")
    if start == -1:
        raise ValueError(f"Cannot find JSON list in {pred_json}")
    data = read_json_from_bytes(raw[start:])

    result: dict[str, dict[str, list[str]]] = {}
    for item in data:
        sample_id = item["id"]
        binary = item.get("pred_binary")
        if binary is None:
            raise ValueError(f"Missing pred_binary for sample {sample_id} in {pred_json}")
        finding = [name for name in finding_names if binary.get(name, 0) == 1]
        diagnosis = [name for name in diagnosis_names if binary.get(name, 0) == 1]
        result[sample_id] = {"finding": finding, "diagnosis": diagnosis}
    return result


def read_json_from_bytes(raw: bytes):
    import json

    return json.loads(raw)


def make_labeled_prompt(
    original_prompt: str,
    finding_labels: list[str],
    diagnosis_labels: list[str],
) -> str:
    finding_str = "; ".join(finding_labels) if finding_labels else "None"
    diagnosis_str = "; ".join(diagnosis_labels) if diagnosis_labels else "None"
    return (
        f"{original_prompt.rstrip()}\n\n"
        f"Please pay close attention to the following candidate labels:\n"
        f"- Finding-related labels: {finding_str}\n"
        f"- Assessment-related labels: {diagnosis_str}\n\n"
        f"Treat these labels only as auxiliary prior information rather than confirmed findings. "
        f"Verify them carefully against the CT scan before including them in the report."
    )


def build_multimodal_inputs(
    base_dataset_path: Path,
    pred_json_path: Path,
    model_path: Path,
    work_dir: Path,
) -> Path:
    print("Step 3: building multimodal model inputs")
    output_path = work_dir / "multimodal_model_inputs.json"
    if not LABEL_INFO_PATH.is_file():
        raise FileNotFoundError(f"Required label_info.json not found: {LABEL_INFO_PATH}")

    finding_top_n = 10
    diagnosis_top_n = 10
    finding_names, diagnosis_names = load_label_info(LABEL_INFO_PATH)
    labels_by_id = load_pred_labels(
        pred_json_path,
        finding_names[:finding_top_n],
        diagnosis_names[:diagnosis_top_n],
    )
    print(f"  using label info: {LABEL_INFO_PATH}")

    base_dataset = read_json(base_dataset_path)
    converted = []
    for sample in base_dataset:
        sample_id = sample["id"]
        info = labels_by_id.get(sample_id, {"finding": [], "diagnosis": []})
        new_sample = deepcopy(sample)
        messages = new_sample.get("messages", [])
        if not messages or messages[0].get("role") != "user":
            raise ValueError(f"Unexpected messages format for {sample_id}")
        messages[0]["content"] = make_labeled_prompt(
            messages[0].get("content", ""),
            info["finding"][:finding_top_n],
            info["diagnosis"][:diagnosis_top_n],
        )
        converted.append(new_sample)

    write_json(output_path, converted)
    print(f"  wrote: {output_path}")
    return output_path
