# Environment Notes

This repository supports two hardware configurations with different CUDA versions.

## Option A — DGX Spark with CUDA 13.0 (original target)

Use the pinned environment:

```bash
micromamba create -n qwen-ft -f environment.yml
micromamba activate qwen-ft
```

Or install into a clean virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Option B — RTX PRO 6000 Blackwell with CUDA 12.8 (validated)

This is the environment used for the FP8 MoE training pipeline
(`configs/lora_huihui_fp8.yaml`, model `edp1096/Huihui-Qwen3.6-35B-A3B-abliterated-FP8`).

Use the existing `pytorch` micromamba environment with `uv pip` for package management:

```bash
# Install / update packages into the shared pytorch env
uv pip install --python /zfs/micromamba/envs/pytorch/bin/python \
    transformers>=5.4 trl datasets accelerate peft bitsandbytes unsloth

# Run training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
UNSLOTH_MOE_BACKEND=native_torch \
micromamba run -n pytorch python scripts/train.py \
    --config configs/lora_huihui_fp8.yaml
```

## Validated Training Stack (CUDA 12.8 / RTX PRO 6000)

The following package versions were validated for the FP8 MoE training pipeline:

| Package | Version |
|---|---|
| `torch` | `2.9.1+cu128` |
| `transformers` | `5.5.0` |
| `trl` | `0.23.0` |
| `unsloth` | `2026.4.8` |
| `peft` | `0.19.1` |
| `datasets` | `4.3.0` |
| `accelerate` | `1.12.0` |
| `bitsandbytes` | `0.48.2` |
| `CUDA` | `12.8` |

## Validated Training Stack (CUDA 13.0 / DGX Spark)

| Package | Version |
|---|---|
| `torch` | `2.10.0+cu130` |
| `torchvision` | `0.25.0+cu130` |
| `transformers` | `5.5.0` |
| `trl` | `0.24.0` |
| `unsloth` | `2026.4.8` |
| `datasets` | `4.3.0` |
| `accelerate` | `1.12.0` |
| `bitsandbytes` | `0.49.2` |
| `CUDA` | `13.0` |

## CUDA Check

After installing, confirm CUDA is active:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

## Notes

- `transformers>=5.4` is required for Qwen3.6 model support.
- The `kernels` package (used by `transformers` for FP8 DeepGEMM kernels) fails
  with `httpx.LocalProtocolError` in air-gapped environments.  `train.py`
  patches this automatically by injecting PyTorch fallbacks before any download
  is attempted.  No action is needed.
- Flash Attention 2 may not be available on all setups.  Unsloth falls back to
  xformers automatically with no loss of correctness.
- `requirements.txt` targets CUDA 13.0 (`+cu130`).  On CUDA 12.8 hardware, use
  the existing `pytorch` micromamba environment instead.
