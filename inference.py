"""
HEADLINE26 inference entrypoint.

Single supported backend: swift infer with PyTorch backend
(`--infer_backend pt`). Step implementations live in my-model-code/ so they can
be replaced independently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_PATH = Path(__file__).resolve().parent
MODEL_CODE_PATHS = (
    APP_PATH / "my-model-code",
    Path("/opt/app/my-model-code"),
)
for path in MODEL_CODE_PATHS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from build_vlm_inputs import build_multimodal_inputs
from finalize_reports import copy_process_files, write_final_reports
from prepare_dataset import DEFAULT_PROMPT, prepare_base_dataset
from step2_runner import run_multilabel_step
from vlm_generation import run_generation_step


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
WORK_DIR = Path(os.environ.get("HEADLINE_WORK_DIR", "/tmp/headline26"))
FINAL_OUTPUT_NAME = "radiology-text-reports.json"
DEFAULT_STEP4_CALLABLE = "vlm_infer_swift_pt:run_vlm_inference"


def run() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HEADLINE_STEP4_CALLABLE", DEFAULT_STEP4_CALLABLE)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    case_order_path, base_dataset_path, case_order = prepare_base_dataset(
        input_path=INPUT_PATH,
        work_dir=WORK_DIR,
        prompt=DEFAULT_PROMPT,
    )
    multilabel_path = run_multilabel_step(
        base_dataset_path=base_dataset_path,
        work_dir=WORK_DIR,
        model_path=MODEL_PATH,
    )
    multimodal_inputs_path = build_multimodal_inputs(
        base_dataset_path=base_dataset_path,
        pred_json_path=multilabel_path,
        model_path=MODEL_PATH,
        work_dir=WORK_DIR,
    )
    raw_results_path = run_generation_step(
        multimodal_inputs_path=multimodal_inputs_path,
        model_path=MODEL_PATH,
        work_dir=WORK_DIR,
    )
    write_final_reports(
        case_order=case_order,
        raw_results_path=raw_results_path,
        output_path=OUTPUT_PATH,
        final_output_name=FINAL_OUTPUT_NAME,
    )
    copy_process_files(work_dir=WORK_DIR, output_path=OUTPUT_PATH)

    print(f"Case order file: {case_order_path}")
    print("Inference pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
