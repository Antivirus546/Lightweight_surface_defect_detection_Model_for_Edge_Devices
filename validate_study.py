"""Audit source images, split isolation, masks, and quantization calibration."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from full_quantize import choose_calibration


def validate(data_root: Path, run: Path):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["samples"]
    paths = [row["image"] for row in rows]
    assert len(paths) == len(set(paths)), "Duplicate source image in manifest"
    assert all(row["split"] in ("train", "val", "test") for row in rows)
    assert all(row["original_split"] == "train" and row["split"] == "train"
               for row in rows if row["original_split"] == "train")
    assert all(row["label"] == 0 and row["mask"] is None
               for row in rows if row["original_split"] == "train")
    for row in rows:
        image_path = data_root / row["image"]
        assert image_path.is_file(), image_path
        if row["label"]:
            assert row["mask"] is not None
            mask_path = data_root / row["mask"]
            assert mask_path.is_file(), mask_path
        else:
            assert row["mask"] is None
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["split"]]["defect" if row["label"] else "normal"] += 1
    train = [row for row in rows if row["split"] == "train"]
    calibration = choose_calibration(train, manifest["seed"], 8)
    assert all(row["split"] == "train" for row in calibration)
    assert not set(row["image"] for row in calibration) & set(
        row["image"] for row in rows if row["split"] != "train"
    )
    categories = sorted(set(row["category"] for row in rows))
    if len(categories) == 15:
        assert len(rows) == 5354, f"Expected original MVTec AD 5354 images, found {len(rows)}"
    summary = {
        "images": len(rows), "categories": categories,
        "counts": dict(counts), "calibration_training_images": len(calibration),
        "duplicates": 0, "missing_images_or_masks": 0,
        "original_train_good_in_validation_or_test": 0,
        "test_or_validation_used_for_calibration": 0, "result": "pass",
    }
    (run / "manifest_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    validate(Path(args.data).resolve(), Path(args.out).resolve())
