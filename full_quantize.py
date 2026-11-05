"""Export the distilled student and verify static INT8 ONNX Runtime inference."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from PIL import Image
from torchvision import transforms

from full_evaluate import bbox_from_map, choose_thresholds, summarize
from full_study import MEAN, SIZE, STD, MultiTaskNet


def load_input(data_root: Path, row: dict) -> np.ndarray:
    image = Image.open(data_root / row["image"]).convert("RGB")
    image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    tensor = transforms.Normalize(MEAN, STD)(transforms.ToTensor()(image))
    return tensor.unsqueeze(0).numpy()


def load_mask(data_root: Path, row: dict) -> np.ndarray:
    if not row["mask"]:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask = Image.open(data_root / row["mask"]).convert("L")
    mask = mask.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


class Reader(CalibrationDataReader):
    def __init__(self, data_root: Path, rows: list[dict]):
        self.data_root = data_root
        self.rows = rows
        self.index = 0

    def get_next(self):
        if self.index >= len(self.rows):
            return None
        row = self.rows[self.index]
        self.index += 1
        return {"input": load_input(self.data_root, row)}

    def rewind(self):
        self.index = 0


def choose_calibration(rows: list[dict], seed: int, per_category: int = 8):
    by_category = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        by_category[row["category"]][row["label"]].append(row)
    rng = random.Random(seed)
    chosen = []
    for category, groups in sorted(by_category.items()):
        for label in (0, 1):
            candidates = list(groups[label])
            rng.shuffle(candidates)
            chosen.extend(candidates[:per_category // 2])
    return chosen


def collect(session: ort.InferenceSession, data_root: Path, rows: list[dict]):
    items = []
    for row in rows:
        x = load_input(data_root, row)
        class_logit, mask_logit = session.run(None, {"input": x})
        score = float(1 / (1 + np.exp(-np.clip(class_logit[0, 0], -50, 50))))
        map_score = 1 / (1 + np.exp(-np.clip(mask_logit[0, 0], -50, 50)))
        items.append({"row": row, "label": row["label"], "score": score,
                      "mask": load_mask(data_root, row), "map": map_score.astype(np.float32)})
    return items


def model_latency(session: ort.InferenceSession):
    x = np.random.default_rng(42).normal(size=(1, 3, SIZE, SIZE)).astype(np.float32)
    for _ in range(30):
        session.run(None, {"input": x})
    times = []
    for _ in range(150):
        started = time.perf_counter()
        session.run(None, {"input": x})
        times.append((time.perf_counter() - started) * 1000)
    return {"device": "CPUExecutionProvider", "batch_size": 1, "warmup": 30,
            "trials": 150, "scope": "ONNX model forward, classification and mask",
            "mean_ms": statistics.mean(times),
            "p95_ms": float(np.percentile(times, 95)),
            "fps": 1000 / statistics.mean(times)}


def pipeline(session: ort.InferenceSession, data_root: Path, rows: list[dict],
             class_threshold: float, mask_threshold: float, out: Path):
    times, events = [], []
    for row in rows:
        started = time.perf_counter()
        x = load_input(data_root, row)
        class_logit, mask_logit = session.run(None, {"input": x})
        score = float(1 / (1 + np.exp(-np.clip(class_logit[0, 0], -50, 50))))
        map_score = 1 / (1 + np.exp(-np.clip(mask_logit[0, 0], -50, 50)))
        box = bbox_from_map(map_score, mask_threshold)
        times.append((time.perf_counter() - started) * 1000)
        if score >= class_threshold:
            events.append({"source": "saved-image virtual camera",
                           "image_id": row["image"], "category": row["category"],
                           "score": score, "bbox_224": box})
    with out.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")
    return {"device": "CPUExecutionProvider", "frames": len(rows),
            "flagged_events": len(events), "event_bytes": out.stat().st_size,
            "source_image_bytes": sum((data_root / row["image"]).stat().st_size for row in rows),
            "scope": "file load, resize, normalize, ONNX forward, threshold and box; no physical camera or MQTT",
            "mean_ms": statistics.mean(times),
            "p95_ms": float(np.percentile(times, 95)),
            "fps": 1000 / statistics.mean(times)}


def export_and_quantize(args):
    data_root, out = Path(args.data).resolve(), Path(args.out).resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["samples"]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    student_dir = out / "student"
    fp32_path = student_dir / "student_fp32.onnx"
    int8_path = student_dir / "student_int8.onnx"
    model = MultiTaskNet("student", pretrained=False)
    model.load_state_dict(torch.load(student_dir / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    example = torch.randn(1, 3, SIZE, SIZE)
    if not fp32_path.exists() or args.force:
        torch.onnx.export(
            model, example, str(fp32_path), input_names=["input"],
            output_names=["class_logit", "mask_logit"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    onnx.checker.check_model(str(fp32_path))
    calibration = choose_calibration(
        train_rows, manifest["seed"], args.calibration_per_category
    )
    if not int8_path.exists() or args.force:
        quantize_static(
            str(fp32_path), str(int8_path), Reader(data_root, calibration),
            quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8, per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        )
    onnx.checker.check_model(str(int8_path))
    quant_graph = onnx.load(str(int8_path), load_external_data=False)
    q_count = sum(node.op_type == "QuantizeLinear" for node in quant_graph.graph.node)
    dq_count = sum(node.op_type == "DequantizeLinear" for node in quant_graph.graph.node)
    int8_weights = sum(init.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)
                       for init in quant_graph.graph.initializer)
    if q_count == 0 or dq_count == 0 or int8_weights == 0:
        raise RuntimeError("INT8 export did not contain quantized weights and QDQ nodes")
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_class, torch_mask = model(example)
    ort_class, ort_mask = fp32_session.run(None, {"input": example.numpy()})
    parity = max(float(np.max(np.abs(ort_class - torch_class.numpy()))),
                 float(np.max(np.abs(ort_mask - torch_mask.numpy()))))
    if parity > 1e-3:
        raise RuntimeError(f"FP32 ONNX differs from PyTorch by {parity}")
    results = {
        "calibration_images": len(calibration),
        "calibration_split": "training only",
        "calibration_per_category": args.calibration_per_category,
        "onnx_export_max_abs_error": parity,
        "quantization": "static PTQ, QDQ, signed INT8 weights and activations, per-channel",
        "quantize_nodes": q_count, "dequantize_nodes": dq_count,
        "int8_initializers": int8_weights,
    }
    for label, session, path in (
        ("fp32_onnx", fp32_session, fp32_path),
        ("int8_onnx", int8_session, int8_path),
    ):
        val_items = collect(session, data_root, val_rows)
        class_threshold, mask_threshold = choose_thresholds(val_items)
        test_items = collect(session, data_root, test_rows)
        summary = summarize(test_items, class_threshold, mask_threshold)
        summary.update({
            "class_threshold_from_validation": class_threshold,
            "mask_threshold_from_validation": mask_threshold,
            "model_size_mb": path.stat().st_size / 1e6,
            "model_only_latency": model_latency(session),
            "cpu_pipeline": pipeline(
                session, data_root, test_rows, class_threshold, mask_threshold,
                student_dir / f"{label}_events.jsonl",
            ),
        })
        results[label] = summary
    (student_dir / "quantization_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    compact = {k: v for k, v in results.items() if k not in ("fp32_onnx", "int8_onnx")}
    compact["fp32_onnx"] = {k: v for k, v in results["fp32_onnx"].items() if k != "per_category"}
    compact["int8_onnx"] = {k: v for k, v in results["int8_onnx"].items() if k != "per_category"}
    print(json.dumps(compact, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--calibration-per-category", type=int, default=8)
    export_and_quantize(parser.parse_args())


if __name__ == "__main__":
    main()
