"""Update the original IEEE-style report with the complete measured study."""

from __future__ import annotations

import json
import os
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("ES_REPORT_SOURCE", r"C:\Users\aniru\OneDrive\Desktop\EST (2).pdf"))
OUT = Path(os.environ.get("ES_REPORT_OUT", ROOT.parent / "outputs" / "ES_Final_Report_Full_Study.pdf"))
RUN = Path(os.environ.get("ES_REPORT_RUN", ROOT / "full_run"))
baseline = json.loads((RUN / "teacher" / "results.json").read_text(encoding="utf-8"))
teacher_onnx = json.loads((RUN / "teacher" / "onnx_results.json").read_text(encoding="utf-8"))
small = json.loads((RUN / "student" / "results.json").read_text(encoding="utf-8"))
quant = json.loads((RUN / "student" / "quantization_results.json").read_text(encoding="utf-8"))
int8 = quant["int8_onnx"]
fp32_onnx = quant["fp32_onnx"]
manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
mqtt = json.loads((RUN / "mqtt_test.json").read_text(encoding="utf-8"))
counts = {
    split: {
        label: sum(category[split][label] for category in manifest["counts"].values())
        for label in ("normal", "defect")
    } for split in ("train", "val", "test")
}


def redact(page: fitz.Page, box):
    page.add_redact_annot(fitz.Rect(box), fill=(1, 1, 1))


def box_text(page: fitz.Page, box, text: str, size: float, *, bold=False,
             color=(0, 0, 0), align=fitz.TEXT_ALIGN_JUSTIFY):
    rect = fitz.Rect(box)
    font = "Times-Bold" if bold else "Times-Roman"
    minimum = min(size, 7.4)
    while size >= minimum:
        remaining = page.insert_textbox(rect, text, fontsize=size, fontname=font,
                                        color=color, align=align, lineheight=1.08)
        if remaining >= 0:
            return
        size -= 0.2
    raise RuntimeError(f"Text did not fit in {rect}: need {-remaining:.1f} more points")


doc = fitz.open(SOURCE)
p1, p2, p3, p4, p5 = doc[0], doc[1], doc[2], doc[3], doc[4]

# Preserve the original pages, diagrams and section order; replace only text that
# must change when the proposal becomes a report with a measured pilot.
redact(p1, (48.5, 254.5, 300.5, 464.0))
redact(p2, (48.5, 114.1, 300.5, 306.5))
redact(p3, (48.5, 469.3, 300.5, 720.0))
redact(p3, (48.5, 281.7, 300.5, 388.2))
redact(p3, (311.5, 251.3, 563.5, 310.0))
redact(p3, (311.5, 332.8, 563.5, 427.5))
redact(p3, (311.5, 553.5, 563.5, 612.7))
redact(p3, (311.5, 635.2, 563.5, 721.0))
redact(p3, (430.5, 198.5, 524.0, 208.0))
redact(p3, (140.0, 198.5, 237.0, 225.5))
redact(p3, (323.0, 198.5, 416.0, 225.5))
redact(p4, (48.5, 208.7, 300.5, 267.5))
redact(p4, (48.5, 296.8, 300.5, 439.5))
redact(p4, (48.5, 480.7, 300.5, 720.0))
redact(p4, (311.5, 208.7, 563.5, 231.8))
redact(p4, (311.5, 264.9, 563.5, 435.2))
redact(p4, (311.5, 470.0, 563.5, 615.0))
redact(p4, (311.5, 648.0, 563.5, 720.0))
redact(p5, (48.5, 283.5, 300.5, 646.0))
redact(p5, (71.0, 61.0, 278.0, 71.5))
redact(p5, (90.5, 158.0, 258.0, 168.0))
redact(p5, (177.0, 179.0, 219.0, 199.5))
p4.add_redact_annot(fitz.Rect(418.5, 71.5, 453.0, 87.0), fill=(0.94, 0.94, 1.0))
for rect in p5.search_for("TBD"):
    if rect.y0 < 240:
        redact(p5, (rect.x0 - 0.5, rect.y0 - 0.4, rect.x1 + 0.5, rect.y1 + 0.4))

for page in (p1, p2, p3, p4, p5):
    page.apply_redactions(images=0, graphics=0, text=0)

abstract = (
    f"Abstract: We evaluated a lightweight surface-defect detector across all 15 MVTec AD "
    f"categories using a documented supervised split (seed 42): "
    f"{counts['train']['normal'] + counts['train']['defect']} training, "
    f"{counts['val']['normal'] + counts['val']['defect']} validation and "
    f"{counts['test']['normal'] + counts['test']['defect']} held-out test images. "
    f"An ImageNet-pretrained ResNet18 teacher and MobileNetV3-Small student jointly "
    f"predict defect class and pixel mask; the student learns from teacher logits and maps. "
    f"The student was exported to ONNX and statically quantized to INT8 with training-only "
    f"calibration. Held-out image F1 was {baseline['classification']['f1']*100:.1f}%, "
    f"{small['classification']['f1']*100:.1f}% and {int8['classification']['f1']*100:.1f}% "
    f"for teacher, student FP32 and student INT8. Pixel AUROC was "
    f"{baseline['localization']['pixel_auroc']:.3f}, "
    f"{small['localization']['pixel_auroc']:.3f} and "
    f"{int8['localization']['pixel_auroc']:.3f}; mask IoU was "
    f"{baseline['localization']['pixel_iou']:.3f}, "
    f"{small['localization']['pixel_iou']:.3f} and "
    f"{int8['localization']['pixel_iou']:.3f}. "
    f"ONNX model files were {teacher_onnx['model_size_mb']:.1f}, "
    f"{fp32_onnx['model_size_mb']:.1f} and {int8['model_size_mb']:.1f} MB. "
    f"Saved images simulated camera frames on the laptop CPU; mean INT8 pipeline time was "
    f"{int8['cpu_pipeline']['mean_ms']:.1f} ms, slower than the FP32 student. "
    f"INT8 accuracy loss makes this export unsuitable without further tuning. "
    f"A local MQTT test delivered "
    f"{mqtt['received_events']} defect events. The split differs from the standard "
    f"unsupervised MVTec protocol, so scores are not directly comparable. Physical camera, "
    f"Raspberry Pi, Jetson and power measurements remain future work."
)
box_text(p1, (49, 255, 300, 464), abstract, 8.5, bold=True)

box_text(p3, (140.5, 198.5, 237, 225),
         "Distilled MobileNetV3-Small + INT8", 8.0, align=fitz.TEXT_ALIGN_LEFT)
box_text(p3, (323.5, 198.5, 416, 225),
         f"{int8['model_size_mb']:.1f} MB INT8; laptop latency", 8.0,
         align=fitz.TEXT_ALIGN_LEFT)
box_text(
    p3, (431, 198.5, 524, 225),
    "Physical edge deployment remains open", 8.0, align=fitz.TEXT_ALIGN_LEFT
)

box_text(p2, (49, 115, 300, 306),
         "3) Compress the student with post-training INT8 quantization.\n"
         "4) Benchmark teacher, student and INT8 inference on the available laptop; "
         "embedded-board timing and power tests remain future work.\n"
         "5) Exercise event-only MQTT transport using a saved-image camera simulation.\n\n"
         "Brief contribution. We implemented a supervised 15-category dataset split, "
         "trained a class-and-mask ResNet18 teacher and a distilled MobileNetV3-Small "
         "student, and verified a static INT8 ONNX export. We measured image and pixel "
         "quality on held-out images, model size, laptop GPU model latency, laptop CPU "
         "pipeline latency and local MQTT delivery. The proposed Raspberry Pi or Jetson "
         "deployment in Fig. 2 was not physically evaluated. The code, split manifest, "
         "logs and model outputs accompany this report. Per-category counts and "
         "test metrics are provided in the supplementary table. A common ONNX "
         "CPU runtime times all three model files, while PyTorch GPU tests "
         "measure the teacher and student forward passes. The source-image "
         "simulation includes loading and postprocessing. These measurements "
         "define what can be replicated on the available laptop.", 9.0)

box_text(p3, (49, 282, 300, 388),
         "This study joins supervised class and mask prediction, teacher-to-student "
         "distillation, static INT8 quantization and event-only MQTT transport "
         "in one reproducible pipeline. We measure each model stage on the "
         "available laptop and report both image and pixel quality. Fig. 2 "
         "shows the intended edge deployment; its camera, actuator and cloud "
         "path are design targets rather than completed physical tests. "
         "The saved-image simulation exercises frame processing and local "
         "event delivery without claiming embedded-board timing.", 8.9)

box_text(p3, (49, 470, 300, 719),
         "The MVTec AD dataset [1] provides 15 categories of normal and defective "
         "industrial images, with pixel masks for defects. The original training/good "
         "images stay in the training split. Because this project uses supervised "
         "defect examples, every original test subtype is shuffled with a "
         "SHA256-derived seed from seed 42. Floor(n/2) images enter training, "
         "floor(n/4) validation, and the remainder are held out for testing. "
         "Thus, train, validation and test paths are disjoint within every category "
         "and defect subtype. This custom protocol differs from the normal-only "
         "MVTec benchmark; published unsupervised scores cannot be compared directly. "
         "The archive contains 5354 images. The exact counts for this split are "
         f"{counts['train']['normal']} normal and {counts['train']['defect']} defect "
         f"training images, {counts['val']['normal']} normal and "
         f"{counts['val']['defect']} defect validation images, and "
         f"{counts['test']['normal']} normal and {counts['test']['defect']} defect "
         "held-out images. Checkpoints and class/mask thresholds use validation only; "
         "INT8 calibration uses training only.", 9.0)
box_text(p3, (312, 252, 563, 309),
         "Inputs are resized to 224 x 224 RGB and normalized with ImageNet statistics. "
         "Masks use nearest-neighbor resize. Training uses random horizontal flips "
         "and a class-balanced replacement sampler; no synthetic defects or "
         "brightness/rotation augmentation were used.", 8.9)
box_text(p3, (312, 333, 563, 427),
         "The baseline is an ImageNet-pretrained ResNet18 encoder with a binary "
         "classification head and a skip-connected mask decoder. Both heads are "
         "fine-tuned on the supervised split. Pixel masks are trained directly, "
         "so the output is a segmentation prediction rather than a Grad-CAM "
         "explanation. It establishes classification, localization, model-size "
         "and latency reference values.", 9.0)
box_text(p3, (312, 554, 563, 612),
         "This implementation uses MobileNetV3-Small depthwise-separable features "
         "with a skip-connected mask decoder. No separate width multiplier, pruning "
         "or synthetic-defect generator was applied. The teacher is absent at "
         "student inference time.", 8.9)
box_text(p3, (312, 636, 563, 720),
         "The student minimizes supervised class BCE, weighted mask BCE and Dice, "
         "plus softened teacher class BCE at T = 2 and map MSE. The implemented "
         "objective is:\n"
         "L = BCEc + 0.5 BCEm + 0.5 Dice + 0.3 BCEkd + 0.2 MSEmap.\n"
         "Teacher outputs are detached; the student uses teacher outputs only "
         "during training. Validation loss selects the checkpoint.", 8.8,
         align=fitz.TEXT_ALIGN_LEFT)

box_text(p4, (49, 209, 300, 267),
         "The class term uses sigmoid-scaled logits at temperature 2. Map MSE "
         "compares student and teacher mask probabilities. The training recipe "
         "uses AdamW with a 1e-4 learning rate, balanced image sampling and "
         "early stopping. No test image enters training or model selection.", 8.9)
box_text(p4, (49, 297, 300, 439),
         "After training, the student is exported to FP32 ONNX and converted by "
         "static post-training quantization to signed INT8 QDQ format with per-channel "
         "weights. Each of the 15 categories contributes four normal and four defect "
         "training images to calibration. The ONNX graph is checked for quantize "
         "and dequantize nodes and INT8 initializers; its FP32 output is compared "
         "numerically with PyTorch. The INT8 model is then evaluated on the same "
         "held-out test split, with thresholds selected on validation. Model file "
         "size and CPU inference time are measured. QDQ may add overhead, so "
         "speedup is measured rather than assumed. Pruning and "
         "quantization-aware fine-tuning were not performed.", 9.0)
box_text(p4, (419, 72, 453, 88), "INT8 PTQ", 6.4,
         align=fitz.TEXT_ALIGN_CENTER)

box_text(p4, (49, 481, 300, 719),
         "For the tested runtime, saved held-out images act as virtual camera "
         "frames. Each file is loaded, resized and normalized, then the model "
         "predicts a defect probability and a pixel map. Class and mask "
         "thresholds are selected on validation by image F1 and aggregate "
         "mask IoU, respectively. A bounding box is extracted from mask pixels "
         "above the mask threshold. Only frames whose class score exceeds the "
         "class threshold produce JSON defect events; normal frames generate "
         "no event. The CPU pipeline timer includes image load, preprocessing, "
         "model forward pass and postprocessing. A separate model-only benchmark "
         "measures GPU inference at batch size one after warm-up. An MQTT QoS 1 "
         "test sends the event JSON to a local broker and checks that the "
         "subscriber receives identical messages. This demonstrates event-only "
         "transport in a laptop simulation. The proposed camera, actuator, "
         "cloud dashboard, Raspberry Pi and Jetson in Fig. 2 were not used. "
         "Their latency, memory use and power draw are not inferred from the "
         "laptop results.", 8.9)
box_text(p4, (312, 209, 563, 231),
         "The event log and MQTT check are reproducible from the packaged "
         "scripts and held-out image paths.", 8.6)
box_text(p4, (312, 265, 563, 435),
         "All three stages use one custom supervised split and the same held-out "
         "test images. We report accuracy, precision, recall, F1 and image AUROC "
         "for defect classification, plus pixel AUROC and mask IoU against the "
         "ground-truth masks. Validation chooses both thresholds before test "
         "evaluation. We report parameter count, estimated convolution/linear "
         "MACs and model file size. Speed measurements include mean and 95th "
         "percentile GPU model-only latency after 30 warm-ups and 150 batch-one "
         "trials, and CPU end-to-end saved-image latency over the test set. "
         "The CPU pipeline measurements include disk loading and preprocessing. "
         "We compare the FP32 teacher, FP32 student and INT8 ONNX student under "
         "the measured laptop conditions. Physical edge-device and power "
         "measurements require hardware and are excluded from numeric comparisons.", 8.9)

implementation = (
    "Experiments ran on a Ryzen 7 7840HS laptop with 16 GB RAM and an NVIDIA "
    "RTX 4060 Laptop GPU (8 GB VRAM). Python 3.13, PyTorch 2.11.0+cu128 and "
    "ONNX Runtime 1.31.1 were used. Both encoders started from ImageNet weights "
    "and used 224 x 224 inputs. AdamW used learning rate 1e-4 and weight decay "
    "1e-4; training used batch size 8, up to 8 epochs, patience 3 and seed 42. "
    "The student used T = 2, class distillation weight 0.3 and map weight 0.2. "
    "The exact split, training logs, checkpoint files and validation thresholds "
    "are packaged with this report. Student training reused a lossless 224 x 224 "
    "cache; pipeline timing read original full-size files. Laptop CPU and GPU "
    "timings refer to different scopes and should not be interchanged."
)
box_text(p4, (312, 471, 563, 615), implementation, 8.9)

results = (
    f"Tables II and III report held-out quality and laptop CPU pipeline cost. "
    f"ResNet18, student FP32 and student INT8 image F1 scores were "
    f"{baseline['classification']['f1']*100:.1f}%, "
    f"{small['classification']['f1']*100:.1f}% and "
    f"{int8['classification']['f1']*100:.1f}%. "
    f"Their pixel IoUs were {baseline['localization']['pixel_iou']:.3f}, "
    f"{small['localization']['pixel_iou']:.3f} and "
    f"{int8['localization']['pixel_iou']:.3f}. "
    "INT8 shrank the file but sharply reduced quality and did not speed the "
    "CPU pipeline. The custom split prevents direct comparison with standard MVTec scores."
)
box_text(p4, (312, 649, 563, 720), results, 8.8)

conclusion = (
    f"Across all 15 MVTec AD categories and {baseline['classification']['n']} "
    f"held-out images, the ResNet18 teacher reached "
    f"{baseline['classification']['f1']*100:.1f}% image F1 and "
    f"{baseline['localization']['pixel_iou']:.3f} pixel IoU. The distilled "
    f"MobileNetV3-Small reached {small['classification']['f1']*100:.1f}% F1 and "
    f"{small['localization']['pixel_iou']:.3f} IoU; its INT8 version reached "
    f"{int8['classification']['f1']*100:.1f}% F1 and "
    f"{int8['localization']['pixel_iou']:.3f} IoU. "
    f"The INT8 ONNX file was {int8['model_size_mb']:.1f} MB versus "
    f"{fp32_onnx['model_size_mb']:.1f} MB for FP32 ONNX. Laptop CPU saved-image "
    f"pipeline times were {teacher_onnx['cpu_pipeline']['mean_ms']:.1f}, "
    f"{fp32_onnx['cpu_pipeline']['mean_ms']:.1f} and "
    f"{int8['cpu_pipeline']['mean_ms']:.1f} ms per image. "
    f"The student FP32 saved {teacher_onnx['cpu_pipeline']['mean_ms'] - fp32_onnx['cpu_pipeline']['mean_ms']:.1f} ms "
    f"relative to the teacher, but INT8 lost "
    f"{(small['classification']['f1'] - int8['classification']['f1'])*100:.1f} "
    f"F1 points and added "
    f"{int8['cpu_pipeline']['mean_ms'] - fp32_onnx['cpu_pipeline']['mean_ms']:.1f} ms "
    "relative to FP32 student inference. Static PTQ is therefore not a suitable "
    "deployment choice here without better calibration or quantization-aware training. "
    f"A local MQTT QoS 1 test delivered all {mqtt['received_events']} flagged "
    "events from virtual camera frames. These findings support a smaller software "
    "inspection pipeline, but no production line, physical camera, embedded board, "
    "actuator, cloud dashboard or power meter was tested. The custom supervised "
    "split is not directly comparable with published normal-only MVTec results. "
    "Future work is physical deployment, power measurement, robustness to lighting "
    "and vibration, and testing additional industrial datasets."
)
box_text(p5, (49, 284.5, 300, 484), conclusion, 8.8)
p5.insert_image(fitz.Rect(55, 489, 294, 565),
                filename=str(RUN / "student" / "maps" / "bottle.png"),
                keep_proportion=True)
box_text(
    p5, (55, 568, 294, 627),
    "Fig. 3. Held-out bottle image, ground-truth mask and distilled student's "
    "predicted defect map. The map is a trained segmentation output; other "
    "category examples are in the experiment package.",
    7.6, align=fitz.TEXT_ALIGN_LEFT
)

box_text(p5, (65, 61, 284, 72), "CUSTOM SUPERVISED 15-CATEGORY TEST",
         8.0, align=fitz.TEXT_ALIGN_CENTER)
box_text(p5, (65, 158, 284, 169), "LAPTOP CPU SAVED-IMAGE PIPELINE",
         8.0, align=fitz.TEXT_ALIGN_CENTER)
p5.insert_text((184, 187), "Params", fontname="Times-Bold", fontsize=8)
p5.insert_text((190, 198), "(M)", fontname="Times-Bold", fontsize=8)

class_rows = [
    [f"{baseline['classification'][k] * 100:.1f}" for k in ("accuracy", "precision", "recall", "f1")],
    [f"{small['classification'][k] * 100:.1f}" for k in ("accuracy", "precision", "recall", "f1")],
    [f"{int8['classification'][k] * 100:.1f}" for k in ("accuracy", "precision", "recall", "f1")],
]
efficiency_rows = [
    [f"{teacher_onnx['model_size_mb']:.1f}", f"{baseline['parameters'] / 1e6:.2f}",
     f"{teacher_onnx['cpu_pipeline']['mean_ms']:.1f}", f"{teacher_onnx['cpu_pipeline']['fps']:.1f}"],
    [f"{fp32_onnx['model_size_mb']:.1f}", f"{small['parameters'] / 1e6:.2f}",
     f"{fp32_onnx['cpu_pipeline']['mean_ms']:.1f}", f"{fp32_onnx['cpu_pipeline']['fps']:.1f}"],
    [f"{int8['model_size_mb']:.1f}", f"{small['parameters'] / 1e6:.2f}",
     f"{int8['cpu_pipeline']['mean_ms']:.1f}", f"{int8['cpu_pipeline']['fps']:.1f}"],
]
columns_ii = [157, 201, 239, 269]
columns_iii = [150, 190, 235, 269]
for row_i, values in enumerate(class_rows):
    y = (99, 109.3, 119.7)[row_i] + 7
    for x, value in zip(columns_ii, values):
        p5.insert_text((x, y), value, fontname="Times-Roman", fontsize=8,
                       color=(0, 0, 0))
for row_i, values in enumerate(efficiency_rows):
    y = (206, 216.3, 226.7)[row_i] + 7
    for x, value in zip(columns_iii, values):
        p5.insert_text((x, y), value, fontname="Times-Roman", fontsize=8,
                       color=(0, 0, 0))
p5.insert_text((65, 246), "Red: physical Pi/Jetson and power N/E. Black: measured.",
               fontname="Times-Italic", fontsize=7, color=(0.65, 0, 0))

# Preserve the original IEEE two-column pages and append a matching full-width
# results page for category counts, localization and reproducibility details.
p6 = doc.new_page(width=p5.rect.width, height=p5.rect.height)
p6.insert_text((138, 58), "SUPPLEMENTARY EXPERIMENTAL DETAILS",
               fontname="Times-Bold", fontsize=11)
p6.insert_text((238, 78), "TABLE IV", fontname="Times-Bold", fontsize=8)
p6.insert_text((117, 91), "HELD-OUT RESULTS BY MVTEC AD CATEGORY",
               fontname="Times-Bold", fontsize=8)

starts = [55, 138, 172, 206, 240, 289, 338, 387, 436, 485]
headers = ["Category", "Train", "Val", "Test", "F1 T", "F1 S", "F1 Q", "IoU T", "IoU S", "IoU Q"]
for x, heading in zip(starts, headers):
    p6.insert_text((x + 1, 109), heading, fontname="Times-Bold", fontsize=7.3)
p6.draw_line((54, 96), (538, 96), color=(0, 0, 0), width=0.6)
p6.draw_line((54, 114), (538, 114), color=(0, 0, 0), width=0.4)
for i, category in enumerate(sorted(manifest["counts"])):
    cc = manifest["counts"][category]
    values = [
        category,
        str(sum(cc["train"].values())),
        str(sum(cc["val"].values())),
        str(sum(cc["test"].values())),
        f"{baseline['per_category'][category]['classification']['f1']*100:.1f}",
        f"{small['per_category'][category]['classification']['f1']*100:.1f}",
        f"{int8['per_category'][category]['classification']['f1']*100:.1f}",
        f"{baseline['per_category'][category]['localization']['pixel_iou']:.3f}",
        f"{small['per_category'][category]['localization']['pixel_iou']:.3f}",
        f"{int8['per_category'][category]['localization']['pixel_iou']:.3f}",
    ]
    y = 128 + 14.5 * i
    for x, value in zip(starts, values):
        p6.insert_text((x + 1, y), value, fontname="Times-Roman", fontsize=7.3)
p6.draw_line((54, 335), (538, 335), color=(0, 0, 0), width=0.6)
p6.insert_text((55, 348), "T = ResNet18 teacher; S = MobileNetV3-Small FP32; Q = student INT8. F1 is image-level; IoU is pixel-level.",
               fontname="Times-Italic", fontsize=7.3)

p6.insert_text((239, 375), "TABLE V", fontname="Times-Bold", fontsize=8)
p6.insert_text((116, 388), "MODEL COST AND LAPTOP CPU LATENCY (ONNX RUNTIME)",
               fontname="Times-Bold", fontsize=8)
columns = [55, 164, 222, 283, 344, 411, 477]
headings = ["Model", "Params M", "MACs G", "ONNX MB", "Model ms", "Full ms", "P95 ms"]
p6.draw_line((54, 394), (539, 394), color=(0, 0, 0), width=0.6)
for x, heading in zip(columns, headings):
    p6.insert_text((x + 1, 408), heading, fontname="Times-Bold", fontsize=7.3)
p6.draw_line((54, 413), (539, 413), color=(0, 0, 0), width=0.4)
for y, name, model, runtime in (
    (427, "Teacher FP32", baseline, teacher_onnx),
    (443, "Student FP32", small, fp32_onnx),
    (459, "Student INT8", small, int8),
):
    values = [name, f"{model['parameters']/1e6:.2f}",
              f"{model['estimated_macs']/1e9:.3f}",
              f"{runtime['model_size_mb']:.1f}",
              f"{runtime['model_only_latency']['mean_ms']:.1f}",
              f"{runtime['cpu_pipeline']['mean_ms']:.1f}",
              f"{runtime['cpu_pipeline']['p95_ms']:.1f}"]
    for x, value in zip(columns, values):
        p6.insert_text((x + 1, y), value, fontname="Times-Roman", fontsize=7.4)
p6.draw_line((54, 465), (539, 465), color=(0, 0, 0), width=0.6)

box_text(p6, (55, 488, 294, 719),
         "Experimental protocol. All 5354 source images are assigned once. "
         "Original train/good stays in training; each test subtype is shuffled "
         "with a stable seed and divided 50/25/remainder into train/validation/test. "
         "The split audit found no duplicate paths or missing masks. Validation "
         "selected checkpoints and thresholds; calibration used 120 training "
         "images, four normal and four defective per category. The INT8 graph "
         f"contained {quant['quantize_nodes']} QuantizeLinear, "
         f"{quant['dequantize_nodes']} DequantizeLinear and "
         f"{quant['int8_initializers']} INT8 initializer tensors. "
         "PyTorch/FP32 ONNX parity was checked numerically. Table V uses the same "
         "ONNX CPU runtime for all three model files; its full time includes "
         "saved-image loading, resize and postprocessing.", 8.5)
box_text(p6, (312, 488, 563, 719),
         "Measurement scope. Table III reports CPU saved-image throughput, "
         "not camera or factory throughput. GPU model-only batch-one timing "
         f"was {baseline['model_only_latency']['mean_ms']:.1f} ms for the teacher "
         f"and {small['model_only_latency']['mean_ms']:.1f} ms for the student; "
         "it excludes file loading and preprocessing. MACs are THOP estimates "
         "for convolution and linear layers and exclude interpolation. "
         f"MQTT received {mqtt['received_events']}/{mqtt['published_events']} "
         "QoS 1 events from the saved-image simulation on localhost. "
         "The per-category values above show variation hidden by aggregate "
         "scores. Physical Raspberry Pi and Jetson performance, memory, "
         "power, camera timing, network latency and local actuation remain "
         "unmeasured; the red note on page 5 marks them as planned. "
         "No numeric estimates are presented for those systems.", 8.5)

metadata = dict(doc.metadata)
metadata["title"] = "Lightweight Computer Vision for Surface Defect Detection - Full Study"
doc.set_metadata(metadata)
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT, garbage=4, deflate=True)
doc.close()
print(OUT)
