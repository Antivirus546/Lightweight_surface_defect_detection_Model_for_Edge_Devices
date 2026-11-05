"""Record the laptop/runtime used for the reported experiment."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import onnxruntime
import psutil
import torch
import torchvision

result = {
    "platform": platform.platform(),
    "python": sys.version,
    "cpu": platform.processor(),
    "physical_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
    "pytorch": torch.__version__,
    "torchvision": torchvision.__version__,
    "onnxruntime": onnxruntime.__version__,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    result.update({
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        "torch_cuda_runtime": torch.version.cuda,
    })
try:
    result["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True, timeout=15,
    ).strip()
except Exception as error:
    result["nvidia_smi_error"] = str(error)
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
