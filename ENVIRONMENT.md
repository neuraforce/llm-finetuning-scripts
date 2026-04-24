# Environment Notes

This repository is easiest to run in a fresh isolated environment.
The default environment targets DGX Spark with CUDA 13.0.

## Recommended Setup

Use the pinned environment:

```bash
micromamba create -n qwen-ft -f environment.yml
micromamba activate qwen-ft
```

Or create a clean virtualenv and install directly:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Validated Training Stack

The following package versions were validated together for local training in
this repository:

- `unsloth==2026.4.8`
- `unsloth_zoo==2026.4.9`
- `transformers==5.5.0`
- `trl==0.24.0`
- `datasets==4.3.0`
- `accelerate==1.12.0`
- `bitsandbytes==0.49.2`
- `torch==2.10.0+cu130`
- `torchvision==0.25.0+cu130`

## CUDA Check

After installing, confirm the CUDA 13.0 wheel is active:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

Expected output includes `+cu130`, CUDA `13.0`, and `True`.

## Notes

- Use an isolated environment. A shared environment can pull in unrelated
  packages that interfere with `transformers`, `vllm`, or quantization.
- `requirements.txt` remains the quick-install path and uses the PyTorch cu130
  wheel index.
- `environment.yml` is the reproducible path for this repository.
