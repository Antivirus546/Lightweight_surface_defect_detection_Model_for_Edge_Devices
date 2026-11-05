"""Export the teacher to ONNX for like-for-like CPU pipeline timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from full_evaluate import choose_thresholds, summarize
from full_quantize import collect, model_latency, pipeline
from full_study import SIZE, MultiTaskNet


def run(args):
    data_root, out = Path(args.data).resolve(), Path(args.out).resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    model_dir = out / "teacher"
    path = model_dir / "teacher_fp32.onnx"
    model = MultiTaskNet("teacher", pretrained=False)
    model.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    example = torch.randn(1, 3, SIZE, SIZE)
    if not path.exists() or args.force:
        torch.onnx.export(model, example, str(path), input_names=["input"],
                          output_names=["class_logit", "mask_logit"],
                          opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_class, torch_mask = model(example)
    ort_class, ort_mask = session.run(None, {"input": example.numpy()})
    parity = max(float(np.max(np.abs(ort_class - torch_class.numpy()))),
                 float(np.max(np.abs(ort_mask - torch_mask.numpy()))))
    if parity > 1e-3:
        raise RuntimeError(f"Teacher ONNX differs from PyTorch by {parity}")
    rows = manifest["samples"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    val_items = collect(session, data_root, val_rows)
    class_threshold, mask_threshold = choose_thresholds(val_items)
    result = summarize(collect(session, data_root, test_rows), class_threshold, mask_threshold)
    result.update({
        "onnx_export_max_abs_error": parity,
        "class_threshold_from_validation": class_threshold,
        "mask_threshold_from_validation": mask_threshold,
        "model_size_mb": path.stat().st_size / 1e6,
        "model_only_latency": model_latency(session),
        "cpu_pipeline": pipeline(session, data_root, test_rows,
                                 class_threshold, mask_threshold,
                                 model_dir / "fp32_onnx_events.jsonl"),
    })
    (model_dir / "onnx_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_category"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())
