"""Evaluate the full MVTec AD teacher/student study and saved-image pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from thop import profile
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms

from full_study import MEAN, SIZE, STD, MVTecRows, MultiTaskNet


def auc(labels, scores):
    y = np.asarray(labels, dtype=np.uint8).ravel()
    s = np.asarray(scores, dtype=np.float32).ravel()
    if len(y) == 0 or y.min() == y.max():
        return None
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = np.cumsum(1 - y)[ends]
    tpr = np.r_[0, tp / tp[-1]]
    fpr = np.r_[0, fp / fp[-1]]
    return float(np.trapezoid(tpr, fpr))


def classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float32)
    pred = scores >= threshold
    tp = int(np.sum((labels == 1) & pred))
    fp = int(np.sum((labels == 0) & pred))
    tn = int(np.sum((labels == 0) & ~pred))
    fn = int(np.sum((labels == 1) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(labels), "normal": int(np.sum(labels == 0)),
            "defect": int(np.sum(labels == 1)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(labels), "precision": precision,
            "recall": recall, "f1": f1, "image_auroc": auc(labels, scores)}


def collect(net, loader, rows, device):
    net.eval()
    items = []
    with torch.no_grad():
        for x, label, masks, indexes in loader:
            x = x.to(device, non_blocking=True)
            class_logit, mask_logit = net(x)
            class_scores = torch.sigmoid(class_logit).cpu().numpy().ravel()
            map_scores = torch.sigmoid(mask_logit).cpu().numpy()[:, 0]
            for j, index in enumerate(indexes.tolist()):
                items.append({"row": rows[index], "label": int(label[j, 0]),
                              "score": float(class_scores[j]),
                              "mask": masks[j, 0].numpy().astype(np.uint8),
                              "map": map_scores[j].astype(np.float32)})
    return items


def choose_thresholds(items):
    labels = np.array([item["label"] for item in items], dtype=np.uint8)
    scores = np.array([item["score"] for item in items], dtype=np.float32)
    grid = np.arange(0.05, 0.96, 0.05)
    class_threshold = float(max(
        grid, key=lambda t: classification_metrics(labels, scores, float(t))["f1"]
    ))
    masks = np.stack([item["mask"] for item in items]).astype(bool)
    maps = np.stack([item["map"] for item in items])
    mask_threshold = float(max(
        grid, key=lambda t: np.logical_and(maps >= t, masks).sum() /
        max(np.logical_or(maps >= t, masks).sum(), 1)
    ))
    return class_threshold, mask_threshold


def mask_metrics(items, threshold):
    masks = np.stack([item["mask"] for item in items]).astype(np.uint8)
    maps = np.stack([item["map"] for item in items]).astype(np.float32)
    pred = maps >= threshold
    gt = masks > 0
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return {"pixel_auroc": auc(masks, maps),
            "pixel_iou": intersection / union if union else None,
            "intersection_pixels": intersection, "union_pixels": union}


def summarize(items, class_threshold, mask_threshold):
    labels = [item["label"] for item in items]
    scores = [item["score"] for item in items]
    result = {
        "classification": classification_metrics(labels, scores, class_threshold),
        "localization": mask_metrics(items, mask_threshold),
    }
    by_category = defaultdict(list)
    for item in items:
        by_category[item["row"]["category"]].append(item)
    result["per_category"] = {
        category: {
            "classification": classification_metrics(
                [item["label"] for item in subset],
                [item["score"] for item in subset], class_threshold
            ),
            "localization": mask_metrics(subset, mask_threshold),
        }
        for category, subset in sorted(by_category.items())
    }
    values = [v["localization"]["pixel_auroc"] for v in result["per_category"].values()]
    result["macro_pixel_auroc"] = statistics.mean(v for v in values if v is not None)
    values = [v["localization"]["pixel_iou"] for v in result["per_category"].values()]
    result["macro_pixel_iou"] = statistics.mean(v for v in values if v is not None)
    return result


def save_example(data_root: Path, item, output: Path):
    row = item["row"]
    image = Image.open(data_root / row["image"]).convert("RGB").resize(
        (SIZE, SIZE), Image.Resampling.BICUBIC
    )
    gt = Image.fromarray(item["mask"] * 255).convert("RGB")
    map_array = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    map_array[..., 0] = (item["map"] * 255).astype(np.uint8)
    map_array[..., 2] = ((1 - item["map"]) * 100).astype(np.uint8)
    overlay = Image.blend(image, Image.fromarray(map_array), 0.45)
    panel = Image.new("RGB", (SIZE * 3, SIZE + 24), "white")
    for i, part in enumerate((image, gt, overlay)):
        panel.paste(part, (SIZE * i, 24))
    draw = ImageDraw.Draw(panel)
    for i, label in enumerate(("Input", "Ground truth", "Predicted defect map")):
        draw.text((SIZE * i + 5, 5), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def model_latency(net, device):
    net.eval()
    sample = torch.randn(1, 3, SIZE, SIZE, device=device)
    with torch.no_grad():
        for _ in range(30):
            net(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(150):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            net(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    result = {"device": str(device), "scope": "model forward only, classification and mask",
              "batch_size": 1, "warmup": 30, "trials": 150,
              "mean_ms": statistics.mean(times), "p95_ms": float(np.percentile(times, 95)),
              "fps": 1000 / statistics.mean(times)}
    if device.type == "cuda":
        result["peak_allocated_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    return result


def bbox_from_map(probability: np.ndarray, threshold: float):
    ys, xs = np.where(probability >= threshold)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def pipeline_simulation(data_root: Path, rows: list[dict], checkpoint: Path, kind: str,
                        class_threshold: float, mask_threshold: float, out: Path):
    # Load onto CPU to compare FP32 and later INT8 in the same runtime class.
    net = MultiTaskNet(kind, pretrained=False)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(MEAN, STD)
    times, events = [], []
    with torch.no_grad():
        for row in rows:
            start = time.perf_counter()
            image = Image.open(data_root / row["image"]).convert("RGB")
            image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
            x = normalize(to_tensor(image)).unsqueeze(0)
            class_logit, mask_logit = net(x)
            score = float(torch.sigmoid(class_logit)[0, 0])
            map_score = torch.sigmoid(mask_logit)[0, 0].numpy()
            box = bbox_from_map(map_score, mask_threshold)
            times.append((time.perf_counter() - start) * 1000)
            if score >= class_threshold:
                events.append({
                    "source": "saved-image virtual camera",
                    "image_id": row["image"], "category": row["category"],
                    "score": score, "bbox_224": box,
                })
    with (out / "simulated_events.jsonl").open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")
    return {
        "device": "cpu", "source": "saved-image virtual camera",
        "frames": len(rows), "flagged_events": len(events),
        "event_bytes": (out / "simulated_events.jsonl").stat().st_size,
        "source_image_bytes": sum((data_root / row["image"]).stat().st_size for row in rows),
        "scope": "file load, RGB resize, normalization, model forward, threshold and bounding box; no physical camera or MQTT",
        "mean_ms": statistics.mean(times), "p95_ms": float(np.percentile(times, 95)),
        "fps": 1000 / statistics.mean(times),
    }


def evaluate(args):
    data_root, out = Path(args.data).resolve(), Path(args.out).resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_rows = {
        split: [row for row in manifest["samples"] if row["split"] == split]
        for split in ("val", "test")
    }
    loaders = {
        split: DataLoader(
            MVTecRows(data_root, rows), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, rows in split_rows.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = out / args.model
    checkpoint = model_dir / "weights.pt"
    net = MultiTaskNet(args.model, pretrained=False).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    val_items = collect(net, loaders["val"], split_rows["val"], device)
    class_threshold, mask_threshold = choose_thresholds(val_items)
    test_items = collect(net, loaders["test"], split_rows["test"], device)
    results = summarize(test_items, class_threshold, mask_threshold)
    examples = {}
    for item in test_items:
        category = item["row"]["category"]
        if item["label"] and category not in examples:
            examples[category] = item
    for category, item in examples.items():
        save_example(data_root, item, model_dir / "maps" / f"{category}.png")
    results["model"] = args.model
    results["dataset"] = "MVTec AD"
    results["protocol"] = "custom supervised split"
    results["seed"] = manifest["seed"]
    results["class_threshold_from_validation"] = class_threshold
    results["mask_threshold_from_validation"] = mask_threshold
    results["model_size_mb"] = checkpoint.stat().st_size / 1e6
    results["parameters"] = sum(p.numel() for p in net.parameters())
    macs, _ = profile(
        net, inputs=(torch.randn(1, 3, SIZE, SIZE, device=device),), verbose=False
    )
    results["estimated_macs"] = int(macs)
    results["macs_method"] = "THOP convolution/linear operator estimate; interpolation excluded"
    results["model_only_latency"] = model_latency(net, device)
    results["cpu_pipeline"] = pipeline_simulation(
        data_root, split_rows["test"], checkpoint, args.model,
        class_threshold, mask_threshold, model_dir,
    )
    (model_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    compact = {key: value for key, value in results.items() if key != "per_category"}
    print(json.dumps(compact, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=("teacher", "student"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
