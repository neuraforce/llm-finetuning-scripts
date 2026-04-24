# Copilot Instructions

This repository contains Python fine-tuning scripts for **Qwen 3.6** (vision, thinking, tool-calling) using **Unsloth** on a **DGX Spark (128 GB GPU)**. Supports LoRA and full fine-tuning.

---

## Commands

```bash
# Install
pip install -r requirements.txt

# LoRA training
python scripts/train.py --config configs/lora_qwen3.yaml

# Full fine-tuning
python scripts/train.py --config configs/full_ft_qwen3.yaml

# Override config keys inline (key=value pairs after the config path)
python scripts/train.py --config configs/lora_qwen3.yaml training.num_train_epochs=5

# Merge LoRA adapter after training
python scripts/export_model.py \
    --base-model Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter outputs/lora_qwen3/adapter \
    --output outputs/lora_qwen3_merged

# Monitor with TensorBoard
tensorboard --logdir outputs/

# Convert docs folder to training JSONL
python scripts/prepare_docs_dataset.py --docs-dir ~/docs --output data/train.jsonl
```

---

## Architecture

- **`scripts/train.py`** — unified entry point. Detects LoRA vs full FT by presence of `lora` key in the YAML config. Loads model via `unsloth.FastVisionModel`, formats data with `dataset_utils`, trains with `trl.SFTTrainer`.
- **`scripts/dataset_utils.py`** — loads JSONL datasets, applies Qwen 3 chat template via `tokenizer.apply_chat_template`, returns a `formatting_func` for `SFTTrainer`.
- **`scripts/export_model.py`** — merges LoRA adapter into base weights; optional GGUF export.
- **`scripts/prepare_docs_dataset.py`** — converts paired `.md`/`.txt` document files into ShareGPT JSONL; expands `{opt1, opt2}` option blocks.
- **`configs/*.yaml`** — all hyperparameters live here; config is a plain dict passed through the call chain (no dataclass).

---

## Key Conventions

- **Model loader**: always use `unsloth.FastVisionModel` (not `AutoModelForCausalLM`) — it applies Qwen-specific patches for vision and thinking.
- **LoRA detection**: `"lora" in cfg` — a config with a `lora:` block trains LoRA; without it, all weights are updated.
- **CUDA stack**: dependency files target CUDA 13.0 on DGX Spark (`torch==2.10.0+cu130`, `torchvision==0.25.0+cu130`). Do not switch back to CUDA 12.x selectors.
- **Precision**: `bf16=True` always; never `fp16` on B200/DGX Spark.
- **Logging**: TensorBoard only (`report_to="tensorboard"`). Do not add wandb.
- **Output**: local disk only. No HuggingFace Hub push.
- **`sys.path`**: `scripts/` is inserted at runtime so cross-script imports work without a package install.

---

## Dataset Format (ShareGPT JSONL)

Each line: `{"conversations": [{"role": "...", "content": "..."}, ...]}`

| Role | Content type | Notes |
|---|---|---|
| `system` / `user` / `assistant` | string | Plain text |
| `user` | list of `{"type": "image"/"text", ...}` | Vision turns |
| `assistant` | string with `<think>...</think>` | Thinking / reasoning |
| `assistant` | string with `<tool_call>...</tool_call>` | Tool invocation |
| `tool` | string with `<tool_response>...</tool_response>` | Tool result |

See `data/example_dataset.jsonl` for a complete reference.

---

## Docs Dataset (`.md` + `.txt` pairs)

`scripts/prepare_docs_dataset.py` converts a directory of paired document files
into a ShareGPT JSONL dataset.  The `.md` file becomes the **system** message;
each Q&A pair from the `.txt` file becomes a **user / assistant** turn.

### `.txt` format

```
> Question text here

Answer text here.
May be multi-line.

--
> Next question?

Next answer.
--
```

### Option expansion

Question templates may contain `{opt1, opt2, ...}` blocks:

- `_` → empty string (the block is removed).
- Other options → prefixed with a space, then substituted.
- Multiple `{...}` blocks generate the **cartesian product** of all options.
- After substitution, consecutive spaces collapse to one and the result is stripped.

Example: `> Was ist das Thema{_, dieses Briefs}?` → two training samples:
- `Was ist das Thema?`
- `Was ist das Thema dieses Briefs?`

`.pdf` files in the docs directory are ignored.
