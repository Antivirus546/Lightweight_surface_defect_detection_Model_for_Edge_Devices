"""Show six held-out MVTec images passing through the trained FP32 student.

Run with the project's virtual environment. The images act as virtual camera
frames; no physical camera or embedded board is involved.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if (PROJECT_ROOT / "work" / "full_run" / "manifest.json").exists():
    DATA = PROJECT_ROOT / "work" / "data" / "mvtec_full"
    RUN = PROJECT_ROOT / "work" / "full_run"
else:
    DATA = SCRIPT_DIR / "demo_data"
    RUN = SCRIPT_DIR / "full_run"
OUT = SCRIPT_DIR / "ES_Simulation_Preview.png"
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((RUN / "student" / "results.json").read_text(encoding="utf-8"))
    class_threshold = results["class_threshold_from_validation"]
    mask_threshold = results["mask_threshold_from_validation"]
    rows = [r for r in manifest["samples"] if r["split"] == "test" and r["category"] == "bottle"]
    good = [r for r in rows if r["label"] == 0][:3]
    defective = [next(r for r in rows if r["subtype"] == subtype)
                 for subtype in ("broken_large", "broken_small", "contamination")]
    selected = good + defective

    session = ort.InferenceSession(
        str(RUN / "student" / "student_fp32.onnx"),
        providers=["CPUExecutionProvider"],
    )
    sheet = Image.new("RGB", (SIZE * 3, (SIZE + 46) * len(selected) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("INPUT IMAGE", "MODEL DEFECT MAP", "DATASET GROUND TRUTH")):
        draw.text((column * SIZE + 8, 6), title, fill="black")
    print("Saved images -> resize/normalize -> MobileNetV3-Small -> score + defect map")
    print(f"Validation thresholds: image={class_threshold:.2f}, pixel={mask_threshold:.2f}")

    for index, row in enumerate(selected):
        started = time.perf_counter()
        image = Image.open(DATA / row["image"]).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = ((array - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        class_logit, mask_logit = session.run(None, {"input": tensor})
        score = float(sigmoid(class_logit[0, 0]))
        probability = sigmoid(mask_logit[0, 0])
        flagged = score >= class_threshold
        overlay = image.copy()
        color = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        color[..., 0] = np.uint8(np.clip(probability * 255, 0, 255))
        overlay = Image.blend(overlay, Image.fromarray(color), 0.48)
        ys, xs = np.where(probability >= mask_threshold)
        if flagged and len(xs):
            ImageDraw.Draw(overlay).rectangle((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), outline="yellow", width=2)
        if row["mask"]:
            ground_truth = Image.open(DATA / row["mask"]).convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST)
            ground_truth = ground_truth.convert("RGB")
        else:
            ground_truth = Image.new("RGB", (SIZE, SIZE), "black")
        y = 30 + index * (SIZE + 46)
        for column, part in enumerate((image, overlay, ground_truth)):
            sheet.paste(part, (column * SIZE, y))
        actual = "DEFECT" if row["label"] else "NORMAL"
        predicted = "DEFECT" if flagged else "NORMAL"
        elapsed_ms = (time.perf_counter() - started) * 1000
        line = f"Frame {index + 1}: actual {actual}, predicted {predicted}, score {score:.3f}"
        draw.text((8, y + SIZE + 5), line, fill="red" if predicted != actual else "black")
        print(f"{line}, demo step {elapsed_ms:.1f} ms", flush=True)
        time.sleep(0.4)

    sheet.save(OUT)
    print(f"Preview saved: {OUT}")
    print("This uses saved test images as virtual frames. The ground-truth panel is for checking the prediction, not an input to the model.")


if __name__ == "__main__":
    main()
