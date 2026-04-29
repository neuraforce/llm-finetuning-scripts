# LLM Fine-tuning Scripts

Fine-tuning scripts for [Qwen 3.6](https://huggingface.co/Qwen) and derivatives (vision, thinking, tool-calling) using [Unsloth](https://github.com/unslothai/unsloth) on a DGX Spark (128 GB GPU memory).

Supports **LoRA** and **full fine-tuning**, all via a single entry point.

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA B200 SXM (128 GB HBM3e) |
| System | DGX Spark |
| CUDA | 13.0 |
| Python | 3.12 |
| PyTorch | 2.10.0 + cu130 |

---

## Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 13.0 PyTorch wheels (`torch==2.10.0+cu130`,
`torchvision==0.25.0+cu130`) before installing the pinned Unsloth stack.
This matches the CUDA 13.0 DGX Spark setup recommended by NVIDIA's DGX Spark
Unsloth guide and Unsloth's CUDA 13 install notes.

### Reproducible Environment

For a clean local setup, prefer a fresh conda or micromamba environment instead
of installing into a shared Python environment:

```bash
micromamba create -n qwen-ft -f environment.yml
micromamba activate qwen-ft
```

If you prefer pip-only setup in an existing clean environment:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository was validated with the package set recorded in
`environment.yml`, which pins the training-critical stack (`unsloth`,
`transformers`, `trl`, `datasets`, `accelerate`, `bitsandbytes`, `torch`,
`torchvision`) to a known-compatible configuration.

After installation, verify CUDA 13.0 is active:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

Expected: a `+cu130` Torch build, CUDA `13.0`, and `True`.

---

## Quick Start

### LoRA fine-tuning

```bash
python scripts/train.py --config configs/lora_qwen3.yaml
```

### Full fine-tuning

```bash
python scripts/train.py --config configs/full_ft_qwen3.yaml
```

### Override config values on the fly

```bash
python scripts/train.py --config configs/lora_qwen3.yaml \
    training.num_train_epochs=5 \
    data.train_file=data/my_custom_train.jsonl
```

### With accelerate / DeepSpeed (full FT of large models)

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py --config configs/full_ft_qwen3.yaml
```

### Monitor training

```bash
tensorboard --logdir outputs/
```

---

## Supported Models

All configs default to `Qwen/Qwen2.5-VL-7B-Instruct` as a placeholder.
Override `model.name` on the command line to train any Qwen model without
editing YAML files.

### Qwen3.6-27B (BF16)

The full-precision 27 B model fits comfortably in 128 GB.  Use **LoRA** for
fast iteration or **full fine-tuning** for maximum accuracy.

```bash
# LoRA (recommended for quick experiments)
python scripts/train.py --config configs/lora_qwen3.yaml \
    model.name=Qwen/Qwen3.6-27B

# Full fine-tuning
python scripts/train.py --config configs/full_ft_qwen3.yaml \
    model.name=Qwen/Qwen3.6-27B
```

> **Note:** At 27 B in BF16 (~54 GB), there is still ample headroom for
> optimizer states, activations, and LoRA adapters on the 128 GB DGX Spark.

### Qwen3.6-27B-FP8

The FP8-quantized variant cuts weight memory roughly in half (~27 GB).
Pass `dtype=auto` so Unsloth preserves the native FP8 weights; LoRA
adapters are still trained in BF16.

```bash
# LoRA on the FP8 model
python scripts/train.py --config configs/lora_qwen3.yaml \
    model.name=Qwen/Qwen3.6-27B-FP8 \
    model.dtype=auto
```

> **Note:** Full fine-tuning of an FP8 model is not supported by Unsloth.
> Use LoRA or merge the adapter after training.

### Changing the model in a YAML config

To permanently set a model in a config file, edit the `model.name` field:

```yaml
model:
  name: "Qwen/Qwen3.6-27B"   # or Qwen/Qwen3.6-27B-FP8
  dtype: "bfloat16"           # use "auto" for FP8 models
  max_seq_length: 8192
```

---

## Export

After LoRA training, merge the adapter into the base model:

```bash
# Standard model
python scripts/export_model.py \
    --base-model Qwen/Qwen3.6-27B \
    --adapter outputs/lora_qwen3/adapter \
    --output outputs/lora_qwen3_merged

# FP8 model — pass --dtype auto to preserve native FP8 weights
python scripts/export_model.py \
    --base-model Qwen/Qwen3.6-27B-FP8 \
    --adapter outputs/lora_qwen3/adapter \
    --output outputs/lora_qwen3_merged \
    --dtype auto
```

Optional GGUF export:

```bash
python scripts/export_model.py \
    --base-model Qwen/Qwen3.6-27B \
    --adapter outputs/lora_qwen3/adapter \
    --output outputs/lora_qwen3_merged \
    --gguf q8_0
```

---

## Dataset Format

All datasets are JSONL files with **ShareGPT-style** conversation records.

### Plain text (instruction / multi-turn chat)

```jsonl
{"conversations": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is 2+2?"},
  {"role": "assistant", "content": "4"}
]}
```

### Thinking / reasoning (`<think>` tokens are preserved as-is)

```jsonl
{"conversations": [
  {"role": "user", "content": "What is 15% of 240?"},
  {"role": "assistant", "content": "<think>\n0.15 × 240 = 36\n</think>\n\n15% of 240 is **36**."}
]}
```

### Vision (image + text)

```jsonl
{"conversations": [
  {"role": "user", "content": [
    {"type": "image", "image": "data/my_image.jpg"},
    {"type": "text",  "text": "What is in this image?"}
  ]},
  {"role": "assistant", "content": "A mountain lake surrounded by pine trees."}
]}
```

### Tool use

Tool calls use Qwen's native `<tool_call>` / `<tool_response>` format embedded in assistant and tool messages.

```jsonl
{"conversations": [
  {"role": "system", "content": "You have access to tools."},
  {"role": "user", "content": "What is the weather in Berlin?"},
  {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Berlin\"}}\n</tool_call>"},
  {"role": "tool", "content": "<tool_response>\n{\"temperature\": 18, \"condition\": \"Cloudy\"}\n</tool_response>"},
  {"role": "assistant", "content": "It is 18°C and cloudy in Berlin."}
]}
```

See `data/example_dataset.jsonl` for a complete reference with all task types.

---

## Docs Dataset (`.md` + `.txt` pairs)

The `scripts/prepare_docs_dataset.py` script converts a directory of paired
document files into a ShareGPT JSONL dataset:

```bash
python scripts/prepare_docs_dataset.py \
    --docs-dir ~/docs \
    --output data/train.jsonl

# Dry run: print stats without writing
python scripts/prepare_docs_dataset.py --docs-dir ~/docs --dry-run
```

Then point your training config at the output file:

```bash
python scripts/train.py --config configs/lora_qwen3.yaml \
    data.train_file=data/train.jsonl
```

### File pair format

For each document, two files must share the same base name:

| File | Role |
|---|---|
| `<name>.md` | Document content — combined with `DOCS_SYSTEM_PREAMBLE` to form the **system** message |
| `<name>.txt` | Q&A pairs — parsed into **user / assistant** turns |

`.pdf` files in the same directory are ignored.

### `.txt` format

```
> Question text here

Answer text here.
It may span multiple lines.

--
> Second question?

Second answer.
--
```

- Samples are separated by a line containing only `--`.
- The question line is prefixed with `> `.
- The answer is all text after the question line until the next `--` or EOF.
- Blank lines between question and answer are ignored.
- With `--split`, train/eval splitting is done by document, not by Q&A row, so
  all questions for a document stay in the same split.

### Option expansion (`{opt1, opt2, ...}`)

A question may contain one or more option blocks to generate multiple training
samples from a single template:

```
> Was ist das Thema{_, dieses Briefs, dieses Dokuments}?
```

Expands to **three** questions, each paired with the same answer:

1. `Was ist das Thema?`
2. `Was ist das Thema dieses Briefs?`
3. `Was ist das Thema dieses Dokuments?`

Rules:

- `_` (underscore) denotes the **empty** option — the block is removed.
- Any other option is prefixed with a space before substitution.
- After substitution, consecutive spaces are collapsed to one and the string is trimmed.
- Multiple `{...}` blocks in one question produce the **cartesian product** of all options.

---

## Bewerbungen Dataset (CV extraction)

The `scripts/prepare_bewerbungen_dataset.py` script converts a directory of
**one folder per CV sample** into a ShareGPT JSONL dataset for a structured
information-extraction task: given the markdown text of a CV, output a JSON
object with the candidate's key attributes.

```bash
python scripts/prepare_bewerbungen_dataset.py \
    --bewerbungen-dir ~/bewerbungen \
    --output data/bewerbungen_train.jsonl

# Dry run: count samples without writing
python scripts/prepare_bewerbungen_dataset.py \
    --bewerbungen-dir ~/bewerbungen \
    --dry-run
```

### Folder structure

Each subdirectory is one training sample:

```
~/bewerbungen/
├── Firstname Lastname/
│   ├── Firstname Lastname.md        # CV text extracted from PDF (markdown) — INPUT
│   ├── Firstname Lastname.json      # Structured parsed output — TARGET
│   ├── Firstname Lastname_metadata.json  # Ignored (technical PDF stats)
│   └── *.webp                       # Ignored (image files)
└── ...
```

The script warns and skips any folder that is missing either the `.md` or
the `.json` file.  It also validates that ground-truth `.json` files contain
exactly the expected 10 fields with correct types, printing a warning and
skipping invalid samples.

### Conversation format

Each folder is converted into a single-turn ShareGPT conversation:

| Role | Content |
|---|---|
| `system` | German extraction prompt listing the 10 fields in a specific order |
| `user` | Full markdown text of the CV |
| `assistant` | Pretty-printed JSON object with keys in the **same order** as the system prompt |

**Example record:**

```jsonl
{"conversations": [
  {"role": "system",    "content": "Du bist ein Assistent zur Analyse von Bewerbungsunterlagen. Extrahiere die strukturierten Informationen aus dem folgenden Lebenslauf und gib sie als JSON-Objekt mit diesen Feldern zurück:\n\n  name ...\n\nAntworte ausschließlich mit dem JSON-Objekt, ohne zusätzlichen Text."},
  {"role": "user",      "content": "# Anna Müller\n\nE-Mail: anna.mueller@example.de\n..."},
  {"role": "assistant", "content": "{\n  \"name\": \"Anna Müller\",\n  \"gender\": \"female\",\n  ..."}
]}
```

The system prompt lists fields in a specific order and the assistant output uses
the **same field order**.  This teaches the model to treat the system prompt as a
schema template: at inference time you can list fields in any order and the model
will honour that order.

### JSON output schema

The assistant always returns a JSON object with exactly these 10 fields:

```json
{
  "name":          "Vollständiger Name (string)",
  "gender":        "\"male\" | \"female\"",
  "email":         "E-Mail-Adresse (string)",
  "address":       "Wohnort / Adresse (string)",
  "date_of_birth": "TT.MM.JJJJ (string)",
  "nationality":   "Nationalität (string)",
  "languages":     ["Sprache Niveau", "..."],
  "education":     ["Abschluss / Studiengang", "..."],
  "skills":        ["Fachliche Fähigkeit", "..."],
  "products":      ["Software / Tool / Produkt", "..."]
}
```

### Field-order augmentation

Use `--augment N` to generate additional training samples per CV with randomly
shuffled field orders.  Each variant produces a different system prompt and a
correspondingly reordered JSON output:

```bash
# 1 canonical + 4 shuffled permutations per CV = 5× more training data
python scripts/prepare_bewerbungen_dataset.py \
    --bewerbungen-dir ~/bewerbungen \
    --output data/bewerbungen_train.jsonl \
    --augment 4

# With train/eval split (only train split is augmented)
python scripts/prepare_bewerbungen_dataset.py \
    --bewerbungen-dir ~/bewerbungen \
    --output data/bewerbungen_train.jsonl \
    --augment 4 --split 0.1
```

When combined with `--split`, the eval split always uses the canonical field
order so evaluation is comparable across runs.

### Inference prompt

At inference time, use `make_system_prompt(field_order)` from
`scripts/prepare_bewerbungen_dataset.py` to build a system prompt for any field
order you need.  The backward-compatible `SYSTEM_PROMPT` constant uses the
canonical field order defined by `FIELD_DEFINITIONS`.

---

## Joint Training

Fine-tune on **both** the docs Q&A task and the bewerbungen CV-extraction task
simultaneously.  Because the two tasks use different system prompts, the model
learns to switch behaviour at inference time based solely on the prompt — no
architecture changes are needed.

### 1. Prepare each dataset

```bash
# CV extraction dataset with augmentation and eval split
python scripts/prepare_bewerbungen_dataset.py \
    --bewerbungen-dir ~/bewerbungen \
    --output data/bewerbungen_train.jsonl \
    --split 0.1 --augment 4

# Docs Q&A dataset with eval split
python scripts/prepare_docs_dataset.py \
    --docs-dir ~/docs \
    --output data/docs_train.jsonl \
    --split 0.1
```

### 2. Merge into training and eval files

```bash
# Joint training file (concatenate + shuffle)
python scripts/merge_datasets.py \
    --inputs data/bewerbungen_train.jsonl data/docs_train.jsonl \
    --output data/joint_train.jsonl \
    --shuffle \
    --seed 42

# Joint eval file (merge the _eval splits)
python scripts/merge_datasets.py \
    --inputs data/bewerbungen_train_eval.jsonl data/docs_train_eval.jsonl \
    --output data/joint_eval.jsonl \
    --shuffle \
    --seed 42
```

`merge_datasets.py` prints the record count of each input file and the total,
so you can verify the dataset balance before training.

### 3. Train

```bash
# Joint LoRA training
python scripts/train.py --config configs/lora_joint.yaml

# With a specific model
python scripts/train.py --config configs/lora_joint.yaml \
    model.name=Qwen/Qwen3.6-27B

# FP8 variant
python scripts/train.py --config configs/lora_joint.yaml \
    model.name=Qwen/Qwen3.6-27B-FP8 \
    model.dtype=auto
```

### Task disambiguation at inference time

The fine-tuned model responds to whichever task its system prompt describes:

| Task | System prompt |
|---|---|
| CV extraction | German extraction prompt from `SYSTEM_PROMPT` in `prepare_bewerbungen_dataset.py` |
| Docs Q&A | German extraction preamble (`DOCS_SYSTEM_PREAMBLE`) + the full `.md` document content |

Both tasks return plain text (JSON string for CV extraction, prose for Q&A) so
no special decoding logic is required.

---

## Config Reference

All configs (`lora_qwen3.yaml`, `lora_bewerbungen.yaml`, `lora_joint.yaml`,
`lora_huihui_fp8.yaml`, `full_ft_qwen3.yaml`) share the same top-level structure:

| Section | Key | Description |
|---|---|---|
| `model` | `name` | HuggingFace model ID or local path |
| `model` | `max_seq_length` | Maximum token length |
| `model` | `dtype` | `bfloat16` (standard) or `auto` (FP8 / quantized models) |
| `model` | `load_in_4bit` | Enable 4-bit quantization (LoRA only; not needed on 128 GB) |
| `model` | `offload_folder` | Optional disk offload directory for large MoE model loading |
| `lora` | `r` | LoRA rank (LoRA configs only) |
| `lora` | `lora_alpha` | LoRA scaling factor |
| `lora` | `use_rslora` | Rank-stabilized LoRA (better for high ranks) |
| `lora` | `target_modules` | List of module names to apply LoRA to |
| `data` | `train_file` | Path to training JSONL |
| `data` | `eval_file` | Path to eval JSONL (optional) |
| `data` | `max_samples` | Limit train and eval dataset size (`null` = all) |
| `data` | `dataset_format` | Always `"sharegpt"` — informational only, not validated by code |
| `training` | `output_dir` | Where to save checkpoints and final model |
| `training` | `max_steps` | Optional hard stop for smoke tests (`-1` or omitted = use epochs) |
| `training` | `report_to` | `tensorboard` (default; no wandb) |
| `training` | `logging_dir` | TensorBoard log directory |
| `training` | `packing` | Sequence packing for higher GPU utilisation |
| `export` | `merge_adapter` | Auto-merge adapter after LoRA training |
| `export` | `merged_output_dir` | Destination for merged weights |

---

## Repository Structure

```
llm-finetuning-scripts/
├── configs/
│   ├── lora_qwen3.yaml              # LoRA hyperparameters (docs Q&A)
│   ├── lora_bewerbungen.yaml        # LoRA hyperparameters (CV extraction)
│   ├── lora_joint.yaml              # LoRA hyperparameters (joint training)
│   ├── lora_huihui_fp8.yaml         # LoRA for Huihui-Qwen3.6-35B FP8 MoE
│   └── full_ft_qwen3.yaml           # Full fine-tune hyperparameters
├── data/
│   └── example_dataset.jsonl        # Reference dataset with all task types
├── scripts/
│   ├── train.py                     # Main training entry point
│   ├── dataset_utils.py             # Dataset loading and formatting helpers
│   ├── export_model.py              # Merge adapter / export to GGUF
│   ├── prepare_docs_dataset.py      # Convert .md/.txt pairs → ShareGPT JSONL
│   ├── prepare_bewerbungen_dataset.py  # Convert bewerbungen folders → ShareGPT JSONL
│   ├── merge_datasets.py            # Concatenate JSONL files for joint training
│   ├── evaluate_model.py            # Evaluate deployed model against ~/docs/
│   ├── evaluate_bewerbungen.py      # Evaluate deployed model on CV extraction
│   └── eval_metrics.py              # Metric helpers (ROUGE, exact match, token F1)
├── outputs/                         # Training checkpoints, eval logs, plots
├── ENVIRONMENT.md                    # CUDA 13 environment notes
├── environment.yml                   # Reproducible CUDA 13 environment
├── requirements.txt
└── README.md
```

---

## Evaluation

Test a deployed vLLM model against the `~/docs/` dataset.  For every `.md`/`.txt`
pair the markdown content is combined with a German extraction preamble to form
the system prompt; all expanded Q&A pairs are then sent to the model.  Responses
are compared to ground-truth answers using ROUGE-L, exact match, token F1, and
edit similarity.

```bash
# Quick smoke test (5 samples)
python scripts/evaluate_model.py --max-samples 5

# Full evaluation with default settings
python scripts/evaluate_model.py

# Higher concurrency for faster runs
python scripts/evaluate_model.py --concurrency 8

# Enable chain-of-thought (uses chat_template_kwargs enable_thinking=True)
python scripts/evaluate_model.py --thinking

# Custom endpoint / model
python scripts/evaluate_model.py \
    --endpoint http://192.168.35.8:8002 \
    --model qwen3
```

Results are written to `outputs/eval_TIMESTAMP.jsonl` (one JSON record per
sample, plus a summary line) and `outputs/eval_TIMESTAMP_plot.png` (a 2×2
figure with per-document ROUGE-L, score distributions, metric comparison, and
latency histogram).

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--docs-dir` | `~/docs` | Directory of `.md`/`.txt` pairs |
| `--endpoint` | `http://192.168.35.8:8002` | vLLM base URL |
| `--model` | `qwen3` | Model ID served by vLLM |
| `--output-dir` | `outputs/` | Where logs and plots are written |
| `--concurrency` | `4` | Max parallel requests |
| `--max-samples` | `0` (all) | Cap for quick smoke tests |
| `--temperature` | `0.0` | Greedy decoding |
| `--max-tokens` | `1024` | Response token cap |
| `--thinking` | flag | Enable chain-of-thought (off by default) |
| `--timeout` | `120` | Per-request timeout (seconds) |
| `--retries` | `3` | Retry attempts per request |
| `--seed` | `42` | Random seed for reproducibility |

---

## Deployment (vLLM)

### Starting the server

Use `scripts/serve_vllm.sh` to start the vLLM OpenAI-compatible server:

```bash
# Base model only
./scripts/serve_vllm.sh

# With a LoRA adapter
LORA_ADAPTER_PATH=/path/to/adapter ./scripts/serve_vllm.sh
```

Key environment variables (all optional, defaults shown):

| Variable | Default | Description |
|---|---|---|
| `CONTAINER_NAME` | `vllm-qwen3.6` | Docker container name |
| `GPU_DEVICE` | `2` | CUDA device number |
| `HOST_PORT` | `8002` | Host port (maps to container 8000) |
| `HF_CACHE` | `/zfs/.cache/huggingface` | HuggingFace model cache |
| `IMAGE` | `vllm/vllm-openai:v0.19.0` | vLLM Docker image |
| `MODEL` | `edp1096/Huihui-Qwen3.6-35B-A3B-abliterated-FP8` | Model to serve |
| `SERVED_MODEL_NAME` | `qwen3` | API model alias |
| `MAX_MODEL_LEN` | `262144` | Max context length |
| `LORA_ADAPTER_PATH` | _(empty)_ | Path to adapter dir; unset = no LoRA |
| `LORA_NAME` | `cv-extraction` | Name to serve the LoRA under |

### Dynamic LoRA loading

If the server is already running with `--enable-lora` (set automatically
when `LORA_ADAPTER_PATH` is passed to `serve_vllm.sh`), you can load or
unload adapters at runtime without restarting:

```bash
# Load an adapter
python scripts/load_lora.py load --name cv-extraction --path /path/to/adapter

# Unload an adapter
python scripts/load_lora.py unload --name cv-extraction

# List currently available models/adapters
python scripts/load_lora.py list
```

### JSON parsing

All evaluation scripts use `scripts/json_utils.parse_llm_json()` which applies
an 8-step recovery chain (think-block stripping → fence stripping → direct parse
→ embedded-object extraction → trailing-comma removal → single-quote fix →
brace balancing → optional `json_repair` library).  Import it in your own
scripts:

```python
from json_utils import parse_llm_json

result = parse_llm_json(raw_response)  # dict | None
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

