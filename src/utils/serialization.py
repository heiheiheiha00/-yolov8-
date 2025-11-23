from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np

from src.data_processing.features import FeatureSet


def features_to_json(features: Iterable[FeatureSet]) -> list[dict]:
    data: list[dict] = []
    for feature in features:
        record = {
            "frame_index": feature.frame_index,
            "angles": feature.angles,
            "velocities": feature.velocities,
            "forces": feature.forces,
            "keypoints": feature.keypoints.tolist(),
        }
        data.append(record)
    return data


def features_from_json(records: Iterable[dict]) -> List[FeatureSet]:
    feature_sets: list[FeatureSet] = []
    for record in records:
        feature_sets.append(
            FeatureSet(
                frame_index=record["frame_index"],
                angles=record["angles"],
                velocities=record.get("velocities", {}),
                forces=record.get("forces", {}),
                keypoints=np.array(record["keypoints"], dtype=float),
            )
        )
    return feature_sets


def save_features_to_file(features: Iterable[FeatureSet], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = features_to_json(features)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def load_features_from_file(path: str | Path) -> List[FeatureSet]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return features_from_json(data)

