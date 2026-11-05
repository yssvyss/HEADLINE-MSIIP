from __future__ import annotations

import json
import shutil
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from pipeline_io import read_json


def write_placeholder_generations(multimodal_inputs_path: Path, output_path: Path) -> None:
    samples = read_json(multimodal_inputs_path)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "response": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_step4_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(
            "HEADLINE_STEP4_CALLABLE must use 'module:function' format, "
            f"got: {spec}"
        )
    for import_root in (Path("/opt/app"), Path("/opt/app/my-model-code")):
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)
    module_name, function_name = spec.split(":", 1)
    module = import_module(module_name)
    func = getattr(module, function_name)
    if not callable(func):
        raise TypeError(f"Configured Step4 object is not callable: {spec}")
    return func


def normalize_step4_result(result: Any, output_path: Path) -> Path:
    if result is None:
        if not output_path.is_file():
            raise FileNotFoundError(
                "Step4 callable returned None but did not create "
                f"{output_path}"
            )
        return output_path

    if isinstance(result, (str, Path)):
        result_path = Path(result)
        if not result_path.is_file():
            raise FileNotFoundError(f"Step4 returned missing path: {result_path}")
        if result_path != output_path:
            shutil.copyfile(result_path, output_path)
        return output_path

    if isinstance(result, list):
        with output_path.open("w", encoding="utf-8") as f:
            for item in result:
                if not isinstance(item, dict):
                    raise TypeError("Step4 returned a list containing non-dict items")
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return output_path

    raise TypeError(
        "Step4 callable must return None, a JSONL path, or list[dict]; "
        f"got {type(result).__name__}"
    )


def run_generation_step(
    multimodal_inputs_path: Path,
    model_path: Path,
    work_dir: Path,
) -> Path:
    print("Step 4: multimodal generation")
    output_path = work_dir / "raw_generation_results.jsonl"
    import os

    callable_spec = os.environ.get("HEADLINE_STEP4_CALLABLE")

    if callable_spec:
        func = load_step4_callable(callable_spec)
        result = func(
            input_json=multimodal_inputs_path,
            output_jsonl=output_path,
            model_path=model_path,
            work_dir=work_dir,
        )
        output_path = normalize_step4_result(result, output_path)
        print(f"  wrote from Python callable {callable_spec}: {output_path}")
        return output_path

    write_placeholder_generations(multimodal_inputs_path, output_path)
    print(f"  wrote placeholder generations: {output_path}")
    return output_path
