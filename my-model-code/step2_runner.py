from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_multilabel_step(base_dataset_path: Path, work_dir: Path, model_path: Path | None = None) -> Path:
    """Run Step2 in a disposable process so all CPU/GPU memory is reclaimed."""
    model_path = model_path or Path("/opt/ml/model")
    output_path = work_dir / "multilabel_predictions.json"
    worker_path = Path(__file__).resolve().with_name("multilabel_infer.py")
    command = [
        sys.executable,
        str(worker_path),
        "--base-dataset-path",
        str(base_dataset_path),
        "--work-dir",
        str(work_dir),
        "--model-path",
        str(model_path),
    ]

    env = os.environ.copy()
    model_code_dir = str(worker_path.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (model_code_dir, "/opt/app/my-model-code", existing_pythonpath) if part
    )

    print("Step 2: starting isolated classifier process", flush=True)
    subprocess.run(command, check=True, env=env)
    if not output_path.is_file():
        raise FileNotFoundError(f"Step2 process finished but did not create {output_path}")
    print("Step 2: isolated classifier process exited; memory reclaimed", flush=True)
    return output_path
