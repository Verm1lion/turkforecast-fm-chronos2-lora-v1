"""TurkForecast-FM-Chronos2-LoRA-v1 minimal inference example.

Loads the LoRA-adapted Chronos-2 from HF Hub and produces a 9-quantile
forecast on a synthetic context. For full GIFT-Eval reproduction (V4 router
+ bucket calibration), see replicate.py.

Usage:
    python inference.py

Requires Path B locked stack (see requirements.txt).
"""

import os
import numpy as np
import torch
from chronos import BaseChronosPipeline

# Tier 1.5 deterministic environment (must be set before any CUDA op)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

REPO_ID = "Verm1ion/turkforecast-fm-chronos2-lora-v1"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PREDICTION_LENGTH = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    print(f"[+] Loading {REPO_ID} on {DEVICE} ...")
    pipe = BaseChronosPipeline.from_pretrained(
        REPO_ID,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    )
    pipe.model.eval()

    # Synthetic sinusoidal context (replace with your Turkish-vertical series)
    t = np.linspace(0, 8 * np.pi, 512, dtype=np.float32)
    context = np.sin(t) + 0.3 * np.sin(3 * t) + 0.1 * np.random.RandomState(SEED).randn(512).astype(np.float32)

    print(f"[+] Forecasting {PREDICTION_LENGTH} steps, 9 quantiles ...")
    quantiles, _means = pipe.predict_quantiles(
        inputs=[context],
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
    )

    arr = quantiles[0].cpu().float().numpy() if hasattr(quantiles[0], "cpu") else np.asarray(quantiles[0])
    print(f"[+] Output shape: {arr.shape}  (expected [{PREDICTION_LENGTH}, 9])")
    print(f"[+] Median (q=0.5) forecast first 8 steps: {arr[:8, 4]}")
    print(f"[+] 80% interval width (q0.9 - q0.1) first 8 steps: {(arr[:8, 8] - arr[:8, 0])}")
    print(f"[+] Non-crossing check (all q_{{k+1}} >= q_k): {np.all(np.diff(arr, axis=1) >= -1e-6)}")


if __name__ == "__main__":
    main()
