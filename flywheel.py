import os
import json
from pathlib import Path

import cv2

from schema import BottleAnalysis

DATASET_ROOT = Path("dataset")
MIN_CONFIDENCE_TO_SAVE = 0.80


class DataFlywheel:
    """
    Auto-saves cropped bottle images + VLM JSON labels into a structured
    folder dataset. Once enough samples accumulate, this dataset trains
    a local MobileNet classifier that replaces the cloud VLM entirely.

    Folder structure:
        dataset/
            PET/   0001.jpg  0001.json  ...
            HDPE/  ...
            PP/    ...
            PVC/   ...
            unknown/  (low-confidence items for manual review)
    """

    def __init__(self, root: Path = DATASET_ROOT):
        self.root = root
        self.saved_count = 0
        self._ensure_dirs()

    def _ensure_dirs(self):
        for label in ["PET", "HDPE", "PP", "PVC", "unknown"]:
            (self.root / label).mkdir(parents=True, exist_ok=True)

    def save(self, obj_id: int, crop_bgr, result: BottleAnalysis):
        """
        Save crop + sidecar JSON if confidence meets the threshold.
        Low-confidence items go to 'unknown/' for manual review.
        """
        if crop_bgr is None:
            return

        folder = result.material if result.confidence >= MIN_CONFIDENCE_TO_SAVE \
            else "unknown"

        img_path = self.root / folder / f"{obj_id:06d}.jpg"
        json_path = self.root / folder / f"{obj_id:06d}.json"

        cv2.imwrite(str(img_path), crop_bgr)
        with open(json_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

        self.saved_count += 1

    def get_counts(self) -> dict:
        counts = {}
        for label in ["PET", "HDPE", "PP", "PVC", "unknown"]:
            folder = self.root / label
            counts[label] = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        return counts
