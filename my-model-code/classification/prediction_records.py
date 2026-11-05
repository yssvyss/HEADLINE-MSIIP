from __future__ import annotations

import numpy as np


def build_prediction_records(
    ids: list[str],
    nii_paths: list[str],
    label_names: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> list[dict]:
    records: list[dict] = []
    y_pred = (y_prob >= threshold).astype(np.int32)
    for idx, sample_id in enumerate(ids):
        records.append(
            {
                "id": sample_id,
                "nii_path": nii_paths[idx],
                "labels": {name: int(y_true[idx, j]) for j, name in enumerate(label_names)},
                "predictions": {name: float(y_prob[idx, j]) for j, name in enumerate(label_names)},
                "pred_binary": {name: int(y_pred[idx, j]) for j, name in enumerate(label_names)},
            }
        )
    return records
