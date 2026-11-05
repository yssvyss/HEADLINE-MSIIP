from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from pipeline_io import write_json


def extract_report_text(record: dict[str, Any]) -> str:
    for key in (
        "response",
        "prediction",
        "predict",
        "generated_text",
        "text",
        "report",
        "content",
    ):
        value = record.get(key)
        if isinstance(value, str):
            return clean_report_text(value)

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return clean_report_text(message["content"])

    choices = record.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return clean_report_text(content)

    return ""


def clean_report_text(text: str) -> str:
    text = text.strip()
    return re.sub(r"^<think>\s*</think>\s*", "", text, flags=re.IGNORECASE).strip()


def load_generation_results(path: Path) -> tuple[dict[str, str], list[str]]:
    results_by_id: dict[str, str] = {}
    results_by_order: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            report = extract_report_text(record)
            results_by_order.append(report)
            sample_id = record.get("id") or record.get("sample_id") or record.get("query_id")
            if sample_id:
                results_by_id[str(sample_id)] = report
            else:
                print(f"WARNING: missing id in {path}:{line_no}; using JSONL row order")
    return results_by_id, results_by_order


def write_final_reports(
    case_order: list[dict[str, Any]],
    raw_results_path: Path,
    output_path: Path,
    final_output_name: str,
) -> Path:
    print("Step 5: writing final official output")
    results_by_id, results_by_order = load_generation_results(raw_results_path)
    ordered_cases = sorted(case_order, key=lambda item: item["index"])
    reports = []
    for position, item in enumerate(ordered_cases):
        if item["id"] in results_by_id:
            reports.append(results_by_id[item["id"]])
        elif position < len(results_by_order):
            reports.append(results_by_order[position])
        else:
            reports.append("")

    output_path.mkdir(parents=True, exist_ok=True)
    final_path = output_path / final_output_name
    write_json(final_path, reports)
    print(f"  wrote: {final_path} ({len(reports)} reports)")
    return final_path


def copy_process_files(work_dir: Path, output_path: Path) -> None:
    import os

    if os.environ.get("HEADLINE_SAVE_PROCESS_FILES", "0") == "0":
        return

    target = output_path / "process_files"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(work_dir, target)
    print(f"Copied process files to {target}")
