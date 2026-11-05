from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def resolve_swift_torch_dtype() -> str:
    # The final runtime target is NVIDIA T4. Do not probe CUDA in this parent
    # process, and do not permit an accidental fallback to bf16.
    return "float16"


def run_vlm_inference(
    input_json: Path,
    output_jsonl: Path,
    model_path: Path,
    work_dir: Path,
) -> None:
    """Run Step 4 with swift using the PyTorch backend.

    This keeps the swift framework and explicitly uses the PyTorch backend:
    `swift infer --infer_backend pt`.

    Container paths:
      input_json   -> /tmp/headline26/multimodal_model_inputs.json
      output_jsonl -> /tmp/headline26/raw_generation_results.jsonl
      model_path   -> /opt/ml/model

    Expected model layout by default:
      /opt/ml/model/vlm/
    """
    swift_bin = os.environ.get("HEADLINE_SWIFT_BIN", "swift")
    resolved_swift = shutil.which(swift_bin)
    if resolved_swift is None:
        raise RuntimeError(
            f"Cannot find swift executable '{swift_bin}'. Add ms-swift/swift to "
            "requirements.txt or set HEADLINE_SWIFT_BIN to the executable path."
        )

    ckpt_path = Path(os.environ.get("HEADLINE_VLM_MODEL_PATH", model_path / "vlm"))
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"VLM checkpoint path does not exist: {ckpt_path}. Expected model "
            "resources under /opt/ml/model/vlm or set HEADLINE_VLM_MODEL_PATH."
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(os.environ.get("TMPDIR", work_dir / "swift_tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    max_model_len = os.environ.get("HEADLINE_MAX_MODEL_LEN", "3584")
    max_new_tokens = os.environ.get("HEADLINE_MAX_NEW_TOKENS", "1024")
    attn_impl = os.environ.get("HEADLINE_SWIFT_ATTN_IMPL", "sdpa")
    device_map = os.environ.get("HEADLINE_SWIFT_DEVICE_MAP", "cuda:0").strip()
    model_kwargs = os.environ.get(
        "HEADLINE_SWIFT_MODEL_KWARGS",
        '{"low_cpu_mem_usage":true}',
    ).strip()
    if model_kwargs:
        parsed_model_kwargs = json.loads(model_kwargs)
        if not isinstance(parsed_model_kwargs, dict):
            raise ValueError("HEADLINE_SWIFT_MODEL_KWARGS must be a JSON object")
    infer_backend = os.environ.get("HEADLINE_SWIFT_INFER_BACKEND", "pt")
    if infer_backend != "pt":
        raise ValueError(
            "Only swift PyTorch backend is supported. Use "
            "HEADLINE_SWIFT_INFER_BACKEND=pt."
        )

    command = [
        resolved_swift,
        "infer",
        "--model",
        str(ckpt_path),
        "--enable_thinking",
        "False",
        "--val_dataset",
        str(input_json),
        "--result_path",
        str(output_jsonl),
        "--max_new_tokens",
        str(max_new_tokens),
        "--num_beams",
        os.environ.get("HEADLINE_NUM_BEAMS", "1"),
        "--temperature",
        os.environ.get("HEADLINE_TEMPERATURE", "0"),
        "--infer_backend",
        "pt",
        "--attn_impl",
        str(attn_impl),
        "--max_model_len",
        str(max_model_len),
    ]

    if device_map:
        command.extend(["--device_map", device_map])
    if model_kwargs:
        command.extend(["--model_kwargs", model_kwargs])

    torch_dtype = resolve_swift_torch_dtype()
    command.extend(["--torch_dtype", torch_dtype])

    extra_args = os.environ.get("HEADLINE_SWIFT_EXTRA_ARGS", "").strip()
    if extra_args:
        command.extend(shlex.split(extra_args))

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TMPDIR", str(tmp_dir))
    env.setdefault("TVM_TEMP_DIR", str(tmp_dir))
    model_code_dir = Path(__file__).resolve().parent
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(model_code_dir), "/opt/app/my-model-code"]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_parts))
    env.setdefault("HEADLINE_PATCH_QWEN_VL", "1")

    print("Running swift inference with PyTorch backend:")
    print(f"  swift:        {resolved_swift}")
    print(f"  model:        {ckpt_path}")
    print(f"  input_json:   {input_json}")
    print(f"  output_jsonl: {output_jsonl}")
    print("  backend:      pt")
    print(f"  attn_impl:    {attn_impl}")
    print(f"  device_map:   {device_map or 'swift/default'}")
    print(f"  model_kwargs: {model_kwargs or 'swift/default'}")
    print(f"  torch_dtype:  {torch_dtype}")
    print(f"  max_len:      {max_model_len}")
    print(f"  max_new:      {max_new_tokens}")
    print(f"  PYTHONPATH:   {env['PYTHONPATH']}")

    subprocess.run(command, check=True, env=env)

    if not output_jsonl.is_file():
        raise FileNotFoundError(f"swift infer finished but did not create {output_jsonl}")
