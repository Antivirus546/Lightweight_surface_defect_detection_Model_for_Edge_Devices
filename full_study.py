"""Full supervised MVTec AD study with classification, masks and distillation.

The original MVTec train/good images remain training-only. Each original test
subtype is deterministically split into train/validation/held-out test images.
This is a custom supervised protocol, not the standard MVTec benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


SIZE = 224
SEED = 42
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stable_seed(seed: int, category: str, subtype: str) -> int:
    digest = hashlib.sha256(f"{seed}/{category}/{subtype}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def prepare(data_root: Path, out: Path, seed: int = SEED,
            categories: tuple[str, ...] = CATEGORIES) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {}
    for category in categories:
        root = data_root / category
        if not (root / "train" / "good").is_dir() or not (root / "test").is_dir():
            raise FileNotFoundError(f"Missing category {category} in {data_root}")
        for image in sorted((root / "train" / "good").glob("*.png")):
            rows.append({"category": category, "subtype": "good", "label": 0,
                         "split": "train", "original_split": "train",
                         "image": image.relative_to(data_root).as_posix(), "mask": None})
        for group in sorted((root / "test").iterdir()):
            if not group.is_dir():
                continue
            images = sorted(group.glob("*.png"))
            random.Random(stable_seed(seed, category, group.name)).shuffle(images)
            n_train, n_val = len(images) // 2, len(images) // 4
            for i, image in enumerate(images):
                split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
                label = int(group.name != "good")
                mask = root / "ground_truth" / group.name / f"{image.stem}_mask.png" if label else None
                if mask is not None and not mask.exists():
                    raise FileNotFoundError(mask)
                rows.append({
                    "category": category, "subtype": group.name, "label": label,
                    "split": split, "original_split": "test",
                    "image": image.relative_to(data_root).as_posix(),
                    "mask": mask.relative_to(data_root).as_posix() if mask else None,
                })
        counts[category] = {
            split: {
                "normal": sum(1 for r in rows if r["category"] == category and r["split"] == split and r["label"] == 0),
                "defect": sum(1 for r in rows if r["category"] == category and r["split"] == split and r["label"] == 1),
            }
            for split in ("train", "val", "test")
        }
    manifest = {
        "dataset": "MVTec AD", "protocol": "custom supervised split",
        "seed": seed, "input_size": SIZE,
        "split_rule": "Original train/good remains train. Each original test subtype is shuffled with SHA256-derived seed; floor(n/2) train, floor(n/4) validation, remainder held-out test.",
        "samples": rows, "counts": counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("Dataset counts:", flush=True)
    for split in ("train", "val", "test"):
        n = sum(v[split]["normal"] for v in counts.values())
        d = sum(v[split]["defect"] for v in counts.values())
        print(f"  {split}: {n} normal, {d} defect, {n+d} total", flush=True)
    return manifest


class MVTecRows(Dataset):
    def __init__(self, data_root: Path, rows: list[dict], train: bool = False):
        self.data_root, self.rows, self.train = data_root, rows, train
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.data_root / row["image"]).convert("RGB")
        image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        if row["mask"]:
            mask = Image.open(self.data_root / row["mask"]).convert("L")
            mask = mask.resize((SIZE, SIZE), Image.Resampling.NEAREST)
            mask_array = (np.asarray(mask) > 0).astype(np.float32)
        else:
            mask_array = np.zeros((SIZE, SIZE), dtype=np.float32)
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask_array = np.fliplr(mask_array).copy()
        x = self.normalize(self.to_tensor(image))
        y = torch.tensor([float(row["label"])], dtype=torch.float32)
        mask = torch.from_numpy(mask_array).unsqueeze(0)
        return x, y, mask, index


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MultiTaskNet(nn.Module):
    def __init__(self, kind: str, pretrained: bool = True):
        super().__init__()
        self.kind = kind
        if kind == "teacher":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.stage1, self.stage2, self.stage3, self.stage4 = (
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            c1, c2, c3, c4 = 64, 128, 256, 512
        elif kind == "student":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.features = models.mobilenet_v3_small(weights=weights).features
            c1, c2, c3, c4 = 16, 24, 48, 576
        else:
            raise ValueError(kind)
        self.classifier = nn.Linear(c4, 1)
        self.d4 = ConvBlock(c4, 128)
        self.d3 = ConvBlock(128 + c3, 128)
        self.d2 = ConvBlock(128 + c2, 64)
        self.d1 = ConvBlock(64 + c1, 32)
        self.mask_head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor):
        if self.kind == "teacher":
            h = self.stem(x)
            f1 = self.stage1(h)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
        else:
            f1 = f2 = f3 = f4 = None
            h = x
            for i, block in enumerate(self.features):
                h = block(h)
                if i == 1:
                    f1 = h
                elif i == 3:
                    f2 = h
                elif i == 8:
                    f3 = h
                elif i == 12:
                    f4 = h
            assert all(t is not None for t in (f1, f2, f3, f4))
        class_logit = self.classifier(F.adaptive_avg_pool2d(f4, 1).flatten(1))
        h = self.d4(f4)
        h = self.d3(torch.cat((F.interpolate(h, size=f3.shape[-2:], mode="bilinear", align_corners=False), f3), dim=1))
        h = self.d2(torch.cat((F.interpolate(h, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2), dim=1))
        h = self.d1(torch.cat((F.interpolate(h, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1), dim=1))
        mask_logit = F.interpolate(self.mask_head(h), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return class_logit, mask_logit


def supervised_loss(class_logit, mask_logit, label, mask):
    class_loss = F.binary_cross_entropy_with_logits(class_logit, label)
    pixel_weight = torch.tensor(5.0, device=mask.device)
    mask_bce = F.binary_cross_entropy_with_logits(mask_logit, mask, pos_weight=pixel_weight)
    probability = torch.sigmoid(mask_logit)
    dims = (1, 2, 3)
    dice = 1 - (2 * (probability * mask).sum(dims) + 1) / (
        probability.sum(dims) + mask.sum(dims) + 1
    )
    return class_loss + 0.5 * mask_bce + 0.5 * dice.mean()


def distillation_loss(student_class, student_mask, teacher_class, teacher_mask):
    temperature = 2.0
    soft_class = torch.sigmoid(teacher_class / temperature)
    class_kd = F.binary_cross_entropy_with_logits(
        student_class / temperature, soft_class
    ) * temperature**2
    mask_kd = F.mse_loss(torch.sigmoid(student_mask), torch.sigmoid(teacher_mask))
    return 0.3 * class_kd + 0.2 * mask_kd


def load_manifest(data_root: Path, out: Path, seed: int):
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else prepare(data_root, out, seed)
    if manifest["seed"] != seed:
        raise ValueError("Manifest seed does not match requested seed")
    return manifest


def make_loaders(data_root: Path, manifest: dict, batch_size: int, workers: int):
    rows = manifest["samples"]
    split_rows = {split: [row for row in rows if row["split"] == split]
                  for split in ("train", "val", "test")}
    counts = Counter(row["label"] for row in split_rows["train"])
    weights = [1 / counts[row["label"]] for row in split_rows["train"]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loaders = {
        "train": DataLoader(
            MVTecRows(data_root, split_rows["train"], train=True),
            batch_size=batch_size, sampler=sampler, num_workers=workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        ),
        "val": DataLoader(
            MVTecRows(data_root, split_rows["val"]), batch_size=batch_size,
            shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        ),
        "test": DataLoader(
            MVTecRows(data_root, split_rows["test"]), batch_size=batch_size,
            shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        ),
    }
    return split_rows, loaders


def train(args):
    data_root, out = Path(args.data).resolve(), Path(args.out).resolve()
    set_seed(args.seed)
    manifest = load_manifest(data_root, out, args.seed)
    split_rows, loaders = make_loaders(data_root, manifest, args.batch_size, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskNet(args.model, pretrained=True).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    teacher = None
    if args.model == "student":
        teacher = MultiTaskNet("teacher", pretrained=False).to(device)
        teacher.load_state_dict(torch.load(out / "teacher" / "weights.pt", map_location=device, weights_only=True))
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model_out = out / args.model
    model_out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    started = time.perf_counter()
    print(f"Training {args.model} on {device}: {len(split_rows['train'])} train, "
          f"{len(split_rows['val'])} val", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, label, mask, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                class_logit, mask_logit = model(x)
                loss = supervised_loss(class_logit, mask_logit, label, mask)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_class, teacher_mask = teacher(x)
                    loss = loss + distillation_loss(class_logit, mask_logit, teacher_class, teacher_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(x)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, label, mask, _ in loaders["val"]:
                x = x.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    class_logit, mask_logit = model(x)
                    loss = supervised_loss(class_logit, mask_logit, label, mask)
                val_total += float(loss.item()) * len(x)
        train_loss = total / len(split_rows["train"])
        val_loss = val_total / len(split_rows["val"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "elapsed_s": time.perf_counter() - started})
        print(f"{args.model} {epoch}/{args.epochs}: train={train_loss:.4f} "
              f"val={val_loss:.4f} elapsed={history[-1]['elapsed_s']:.0f}s", flush=True)
        if val_loss < best:
            best = val_loss
            stale = 0
            torch.save(model.state_dict(), model_out / "weights.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping on validation loss", flush=True)
                break
    with (model_out / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "train_loss", "val_loss", "elapsed_s"))
        writer.writeheader()
        writer.writerows(history)
    settings = {
        "model": args.model, "seed": args.seed, "epochs_requested": args.epochs,
        "epochs_completed": len(history), "batch_size": args.batch_size,
        "learning_rate": args.lr, "weight_decay": 1e-4,
        "optimizer": "AdamW", "input_size": SIZE,
        "training_sampler": "balanced replacement sampler by image label",
        "supervised_loss": "BCE class + 0.5 weighted BCE mask + 0.5 Dice",
        "distillation": "student only: 0.3 soft binary logit BCE at T=2 + 0.2 map MSE",
        "checkpoint_selection": "lowest validation supervised loss",
        "training_seconds": time.perf_counter() - started,
        "device": str(device),
        "pytorch": torch.__version__,
    }
    if device.type == "cuda":
        settings["gpu_name"] = torch.cuda.get_device_name(0)
        settings["peak_allocated_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    (model_out / "training_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(json.dumps(settings, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=("teacher", "student"))
    parser.add_argument("--categories", default=",".join(CATEGORIES),
                        help="Comma-separated categories for prepare")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(Path(args.data).resolve(), Path(args.out).resolve(), args.seed,
                tuple(x.strip() for x in args.categories.split(",") if x.strip()))
    else:
        if args.model is None:
            parser.error("--model is required for train")
        train(args)


if __name__ == "__main__":
    main()
