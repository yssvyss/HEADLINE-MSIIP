from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from pipeline_io import read_json, write_json


SOFT_SLUG = "stacked-head-non-contrast-ct-soft-tissue-recon"
SOFT_GEOMETRY_RELPATH = (
    "stacked-head-non-contrast-ct-soft-tissue-reconstruction-volume-geometry.json"
)
DEFAULT_PROMPT = "Generate a radiology report for the given head CT scan."
IDENTITY_DIRECTION_3D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def find_image(input_path: Path, slug: str) -> Path:
    location = input_path / "images" / slug
    for pattern in ("*.mha", "*.nii.gz", "*.nii"):
        matches = sorted(location.glob(pattern))
        if matches:
            return matches[0]
    if slug == SOFT_SLUG:
        return find_soft_image_fallback(input_path, location)
    raise FileNotFoundError(f"No image found under {location}")


def find_soft_image_fallback(input_path: Path, expected_location: Path) -> Path:
    image_root = input_path / "images"
    candidates: list[Path] = []
    for pattern in ("*.mha", "*.nii.gz", "*.nii"):
        candidates.extend(sorted(image_root.rglob(pattern)))

    if not candidates:
        raise FileNotFoundError(
            f"No image found under official soft socket {expected_location} "
            f"or fallback image root {image_root}"
        )

    geometry_path = input_path / SOFT_GEOMETRY_RELPATH
    geometry = read_json(geometry_path) if geometry_path.is_file() else None
    scored: list[tuple[int, Path, str]] = []
    for candidate in candidates:
        score, reason = score_soft_candidate(candidate, geometry)
        if score > 0:
            scored.append((score, candidate, reason))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        raise FileNotFoundError(
            f"Could not identify a soft-window image under {image_root}. "
            f"Expected official path: {expected_location}"
        )
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        details = "\n".join(
            f"  score={score} path={path} reason={reason}"
            for score, path, reason in scored[:5]
        )
        raise RuntimeError(
            "Ambiguous soft-window image candidates. Put the file under the "
            f"official socket path {expected_location}.\n{details}"
        )

    score, chosen, reason = scored[0]
    print(
        "WARNING: official soft socket path was empty; using fallback "
        f"candidate score={score}: {chosen} ({reason})"
    )
    return chosen


def score_soft_candidate(candidate: Path, geometry: Any) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    lower_path = str(candidate).lower()
    lower_name = candidate.name.lower()

    if any(token in lower_name for token in ("soft", "tissue", "wt")):
        score += 4
        reasons.append("soft-like filename")
    if "soft" in lower_path or "tissue" in lower_path:
        score += 3
        reasons.append("soft-like path")

    try:
        img = sitk.ReadImage(str(candidate))
        n_cases = get_subvolume_count(img)
        size = img.GetSize()
        spacing = img.GetSpacing()
        if len(size) >= 3 and size[2] < 100:
            score += 1
            reasons.append("z-size<100")
        if len(spacing) >= 3 and 3.0 <= spacing[2] <= 5.0:
            score += 1
            reasons.append("header z-spacing near 4mm")
    except Exception as exc:
        return 0, f"unreadable image: {exc}"

    if isinstance(geometry, list) and geometry:
        if len(geometry) == n_cases:
            score += 2
            reasons.append("case count matches soft geometry")

        z_spacings = [
            float(item.get("spacing", [0, 0, 0])[2])
            for item in geometry
            if len(item.get("spacing", [])) >= 3
        ]
        z_sizes = [
            int(item.get("unpadded_size", [0, 0, 0])[2])
            for item in geometry
            if len(item.get("unpadded_size", [])) >= 3
        ]
        if z_spacings and 3.0 <= sorted(z_spacings)[len(z_spacings) // 2] <= 5.0:
            score += 3
            reasons.append("geometry z-spacing near 4mm")
        if z_sizes and sorted(z_sizes)[len(z_sizes) // 2] < 100:
            score += 2
            reasons.append("geometry z-size<100")

    return score, "; ".join(reasons) if reasons else "no soft evidence"


def infer_batch_id(path: Path) -> str:
    match = re.search(r"(batch\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return path.name.replace(".nii.gz", "").replace(".mha", "").replace(".nii", "")


def get_subvolume_count(img: sitk.Image) -> int:
    if img.GetDimension() == 4:
        return img.GetSize()[3]
    if img.GetDimension() == 3:
        return 1
    raise ValueError(f"Expected 3D or 4D image, got dimension={img.GetDimension()}")


def build_fallback_geometry(img: sitk.Image, n_cases: int) -> list[dict[str, Any]]:
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    if img.GetDimension() == 4:
        size_3d = size[:3]
        spacing_3d = spacing[:3]
        origin_3d = origin[:3]
        direction_3d = tuple(direction[row * 4 + col] for row in range(3) for col in range(3))
    elif img.GetDimension() == 3:
        size_3d = size
        spacing_3d = spacing
        origin_3d = origin
        direction_3d = direction
    else:
        raise ValueError(f"Expected 3D or 4D image, got dimension={img.GetDimension()}")

    direction_3d = normalize_direction_3d(direction_3d)
    record = {
        "unpadded_size": [int(size_3d[0]), int(size_3d[1]), int(size_3d[2])],
        "spacing": [float(spacing_3d[0]), float(spacing_3d[1]), float(spacing_3d[2])],
        "origin": [float(origin_3d[0]), float(origin_3d[1]), float(origin_3d[2])],
        "direction": [float(value) for value in direction_3d],
    }
    return [dict(record) for _ in range(n_cases)]


def normalize_direction_3d(direction: tuple[float, ...]) -> tuple[float, ...]:
    if len(direction) != 9:
        print(
            "WARNING: fallback geometry found non-3D direction; using identity "
            f"direction instead: len={len(direction)}"
        )
        return IDENTITY_DIRECTION_3D

    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        print(
            "WARNING: fallback geometry found singular 3D direction; using "
            "identity direction instead"
        )
        return IDENTITY_DIRECTION_3D
    return tuple(float(value) for value in direction)


def load_geometry(path: Path, n_cases: int, img: sitk.Image | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        if img is None:
            raise FileNotFoundError(
                f"Missing soft geometry socket: {path}. It is required because the "
                "stacked 4D header does not contain the per-case geometry."
            )
        print(
            "WARNING: missing soft geometry socket; building fallback geometry "
            f"from image header only: {path}"
        )
        return build_fallback_geometry(img, n_cases)
    geometry = read_json(path)
    if not isinstance(geometry, list):
        raise ValueError(f"Geometry socket must be a list: {path}")
    if len(geometry) != n_cases:
        raise ValueError(
            f"Geometry length mismatch: {len(geometry)} entries for {n_cases} sub-volumes."
        )
    return geometry


def extract_soft_subvolume(
    img: sitk.Image,
    index: int,
    geometry: dict[str, Any],
) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img)
    sub = arr[index] if img.GetDimension() == 4 else arr

    size_x, size_y, size_z = geometry["unpadded_size"]
    sub = sub[:size_z, :size_y, :size_x]

    out = sitk.GetImageFromArray(sub.astype(np.float32, copy=False))
    out.SetSpacing(tuple(geometry["spacing"]))
    out.SetOrigin(tuple(geometry["origin"]))
    out.SetDirection(tuple(geometry["direction"]))
    return out


def prepare_base_dataset(
    input_path: Path,
    work_dir: Path,
    prompt: str = DEFAULT_PROMPT,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    print("Step 1: preparing soft-window NIfTI files and base dataset")
    soft_path = find_image(input_path, SOFT_SLUG)
    soft_img = sitk.ReadImage(str(soft_path))
    n_cases = get_subvolume_count(soft_img)
    geometry = load_geometry(input_path / SOFT_GEOMETRY_RELPATH, n_cases, img=soft_img)

    batch_id = infer_batch_id(soft_path)
    nii_dir = work_dir / "nii"
    nii_dir.mkdir(parents=True, exist_ok=True)

    order_records: list[dict[str, Any]] = []
    base_dataset: list[dict[str, Any]] = []

    for index in range(n_cases):
        nii_path = nii_dir / f"{batch_id}_{index:04d}_soft.nii.gz"
        case_id = str(nii_path)
        sub_img = extract_soft_subvolume(soft_img, index, geometry[index])
        sitk.WriteImage(sub_img, str(nii_path), useCompression=True)

        order_records.append(
            {
                "index": index,
                "id": case_id,
                "batch_id": batch_id,
                "source_soft_path": str(soft_path),
                "nii_path": str(nii_path),
            }
        )
        base_dataset.append(
            {
                "id": case_id,
                "messages": [
                    {
                        "role": "user",
                        "content": f"<video>\n{prompt}",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                    },
                ],
                "videos": [str(nii_path)],
            }
        )

    case_order_path = work_dir / "case_order.json"
    base_dataset_path = work_dir / "base_dataset.json"
    write_json(case_order_path, order_records)
    write_json(base_dataset_path, base_dataset)
    print(f"  soft input: {soft_path}")
    print(f"  cases: {n_cases}")
    print(f"  wrote: {case_order_path}")
    print(f"  wrote: {base_dataset_path}")
    return case_order_path, base_dataset_path, order_records
