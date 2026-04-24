# AGENTS.md — Codex / Jules Agent Instructions

This file gives AI coding agents the context needed to work effectively in this repository.

## Purpose

This repo contains Python scripts for fine-tuning **Qwen 3.6** (vision, thinking, tool-calling) using **Unsloth** on a **DGX Spark (128 GB GPU)**. Supports LoRA and full fine-tuning via YAML config files.

---

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run LoRA training (docs Q&A task)
python scripts/train.py --config configs/lora_qwen3.yaml

# Run LoRA training (CV extraction task)
python scripts/train.py --config configs/lora_bewerbungen.yaml

# Run joint LoRA training (both tasks combined)
python scripts/train.py --config configs/lora_joint.yaml

# Run full fine-tuning
python scripts/train.py --config configs/full_ft_qwen3.yaml

# Override a config key inline
python scripts/train.py --config configs/lora_qwen3.yaml training.num_train_epochs=5

# Train with a specific model (override placeholder)
python scripts/train.py --config configs/lora_qwen3.yaml model.name=Qwen/Qwen3.6-27B

# Merge LoRA adapter into base model (BF16)
python scripts/export_model.py \
    --base-model Qwen/Qwen3.6-27B \
    --adapter outputs/lora_qwen3/adapter \
    --output outputs/lora_qwen3_merged

# Merge LoRA adapter into FP8 model (pass --dtype auto)
python scripts/export_model.py \
    --base-model Qwen/Qwen3.6-27B-FP8 \
    --adapter outputs/lora_qwen3/adapter \
    --output outputs/lora_qwen3_merged \
    --dtype auto

# Launch TensorBoard
tensorboard --logdir outputs/

# Convert a docs folder (.md/.txt pairs) into a training JSONL
python scripts/prepare_docs_dataset.py --docs-dir ~/docs --output data/train.jsonl

# Dry-run (counts samples without writing)
python scripts/prepare_docs_dataset.py --docs-dir ~/docs --dry-run

# Convert a bewerbungen folder (CV extraction) into a training JSONL
python scripts/prepare_bewerbungen_dataset.py \
    --bewerbungen-dir ~/bewerbungen \
    --output data/bewerbungen_train.jsonl

# Dry-run (counts samples without writing)
python scripts/prepare_bewerbungen_dataset.py --bewerbungen-dir ~/bewerbungen --dry-run

# Merge multiple JSONL datasets for joint training
python scripts/merge_datasets.py \
    --inputs data/bewerbungen_train.jsonl data/docs_train.jsonl \
    --output data/joint_train.jsonl \
    --shuffle

# Dry-run merge (print counts without writing)
python scripts/merge_datasets.py \
    --inputs data/bewerbungen_train.jsonl data/docs_train.jsonl \
    --output data/joint_train.jsonl \
    --dry-run

# Evaluate deployed model against ~/docs/ Q&A dataset
python scripts/evaluate_model.py

# Evaluate deployed model on bewerbungen CV extraction
python scripts/evaluate_bewerbungen.py --bewerbungen-dir ~/bewerbungen
```

No build step, no test suite to run — verify changes by doing a short training dry-run or by importing the affected module.

---

## Architecture

```
scripts/train.py
  └─ loads YAML config
  └─ detects LoRA vs full FT (presence of "lora" block in config)
  └─ calls train_lora() or train_full()
       └─ unsloth.FastVisionModel.from_pretrained()   ← model loader
       └─ dataset_utils.load_dataset_from_config()    ← data
       └─ dataset_utils.make_formatting_func()        ← tokenization
       └─ trl.SFTTrainer                              ← training loop
       └─ model.save_pretrained()                     ← checkpoint

scripts/export_model.py
  └─ merge_from_disk() or merge_and_save()  ← called post-training
  └─ --dtype auto required for FP8 models  ← preserves native FP8 weights
  └─ optional: export_to_gguf()             ← via Unsloth's GGUF support

scripts/dataset_utils.py
  └─ load_jsonl()                           ← raw file loader
  └─ load_dataset_from_config()             ← returns HF Dataset objects
  └─ make_formatting_func()                 ← returns SFTTrainer-compatible formatter
  └─ format_conversation_for_qwen()         ← applies Qwen chat template

scripts/prepare_docs_dataset.py
  └─ find_pairs()        ← discover .md/.txt pairs in docs directory
  └─ parse_txt()         ← split .txt into (question, answer) tuples
  └─ expand_options()    ← expand {opt1, opt2} blocks (cartesian product)
  └─ DOCS_SYSTEM_PREAMBLE ← German preamble prepended to every doc system message
  └─ convert()           ← orchestrates conversion, writes JSONL

scripts/prepare_bewerbungen_dataset.py
  └─ FIELD_DEFINITIONS   ← ordered list of (field_name, description) tuples
  └─ FIELD_NAMES         ← field names in canonical order (derived from FIELD_DEFINITIONS)
  └─ make_system_prompt(field_order) ← build system prompt for any field permutation
  └─ make_assistant_json(data, field_order) ← output JSON with keys in given order
  └─ SYSTEM_PROMPT       ← backward-compatible constant (canonical field order)
  └─ validate_json_record() ← warn on missing/wrong-type fields in ground truth
  └─ find_pairs()        ← discover .md/.json pairs in bewerbungen directory
  └─ convert()           ← reads each folder, supports --augment N field permutations

scripts/merge_datasets.py
  └─ load_jsonl()        ← load each input file
  └─ main()              ← concatenate + optional shuffle, write output
  └─ --dry-run           ← print counts without writing

scripts/evaluate_model.py
  └─ _build_tasks()      ← load .md/.txt pairs, expand Q&A into task list
  └─ _run_eval()         ← async evaluation loop (concurrency via semaphore)
  └─ _make_plot()        ← generate 2×2 matplotlib summary figure
  └─ imports DOCS_SYSTEM_PREAMBLE from prepare_docs_dataset

scripts/eval_metrics.py
  └─ compute_all()       ← returns dict with exact_match, token_f1, rouge*, edit_sim
  └─ strip_thinking()    ← removes <think>...</think> from model responses

scripts/evaluate_bewerbungen.py
  └─ _build_tasks_from_dir()   ← load .md/.json pairs from bewerbungen directory
  └─ _build_tasks_from_jsonl() ← load from prepared eval JSONL
  └─ _compare_fields()         ← compare predicted JSON fields against ground truth
  └─ _run_eval()               ← async evaluation loop
  └─ imports SYSTEM_PROMPT, FIELD_NAMES from prepare_bewerbungen_dataset
```

### Config → code mapping

The YAML config is loaded as a plain Python dict and passed through the entire call chain. There is no config class or dataclass — keys are accessed by string. When adding new config keys, also update the relevant section in `README.md`.

---

## Dataset Format

All datasets are JSONL. Each line is a JSON object with a `conversations` list (ShareGPT-style):

```json
{"conversations": [
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "..." },
  {"role": "assistant", "content": "..."}
]}
```

- `content` can be a string (text) or a list of `{"type": "image"|"text", ...}` dicts (vision turns).
- `<think>...</think>` tokens in assistant messages are preserved as plain text — no special handling needed.
- Tool calls use Qwen's `<tool_call>` / `<tool_response>` XML-like tags embedded in message content.
- The `role` value `"tool"` is used for tool response turns.

See `data/example_dataset.jsonl` for examples of every task type.

---

## Bewerbungen Dataset (CV Extraction)

`scripts/prepare_bewerbungen_dataset.py` converts a directory of per-sample folders into ShareGPT JSONL:

```bash
python scripts/prepare_bewerbungen_dataset.py \
    --bewerbungen-dir ~/bewerbungen \
    --output data/bewerbungen_train.jsonl

# With field-order augmentation (1 canonical + 4 shuffled permutations per CV)
python scripts/prepare_bewerbungen_dataset.py \
    --bewerbungen-dir ~/bewerbungen \
    --output data/bewerbungen_train.jsonl \
    --augment 4 --split 0.1
```

Each folder contains one `.md` (CV markdown) and one `.json` (structured output). The script:

- Validates ground-truth `.json` fields and types, printing a warning for anomalies.
- Outputs single-turn conversations where the system prompt lists fields in a specific order and the assistant JSON uses the **same field order** (schema-template pattern).
- With `--augment N`: generates 1 canonical + N randomly shuffled field-order variants per CV, multiplying training data by up to N+1.  Only the train split is augmented; the eval split always uses the canonical order.
- `FIELD_DEFINITIONS` in the script is the single source of truth for field names and descriptions.  `FIELD_NAMES`, `SYSTEM_PROMPT`, `make_system_prompt()`, and `make_assistant_json()` are all derived from it.

---

## Joint Training

Combine docs Q&A and CV extraction into one training run:

```bash
python scripts/merge_datasets.py \
    --inputs data/bewerbungen_train.jsonl data/docs_train.jsonl \
    --output data/joint_train.jsonl \
    --shuffle

python scripts/train.py --config configs/lora_joint.yaml model.name=Qwen/Qwen3.6-27B
```

Task routing at inference time is done purely by system prompt — no architecture changes needed.

---

## Docs Dataset (`.md` + `.txt` pairs)

`scripts/prepare_docs_dataset.py` converts paired document files into ShareGPT JSONL:

```bash
python scripts/prepare_docs_dataset.py --docs-dir ~/docs --output data/train.jsonl
```

### Input format

- `<name>.md` — document content; combined with `DOCS_SYSTEM_PREAMBLE` (a module-level constant in `prepare_docs_dataset.py`) as the **system** message.
- `<name>.txt` — Q&A pairs for that document.

### `.txt` format

Samples are separated by a line containing only `--`.  The question line is prefixed with `> `.  The answer follows on subsequent lines.

### Option expansion

Question templates may contain `{opt1, opt2, ...}` blocks.  All combinations across all blocks are expanded into separate training samples:

- `_` (underscore) → empty string; the block vanishes.
- Other options → a leading space is prepended, then the option is substituted.
- After substitution, consecutive spaces collapse to one and the result is stripped.
- Multiple `{...}` blocks produce the **cartesian product** of all options.

`.pdf` files in the docs directory are ignored.

---

## Key Conventions

- **Config drives behaviour**: add new features via YAML config keys, not CLI flags (except `--config` and overrides).
- **`lora` block = LoRA mode**: `train.py` detects the method by checking `if "lora" in cfg`, not by a separate `--method` flag.
- **`scripts/` on sys.path**: `train.py` inserts its own directory into `sys.path` so `dataset_utils` and `export_model` can be imported without a package install.
- **Unsloth `FastVisionModel`**: always use this loader (not `AutoModelForCausalLM`) — it applies Qwen-specific patches and enables vision + thinking.
- **CUDA 13.0 environment**: dependency files target DGX Spark CUDA 13.0 with `torch==2.10.0+cu130` and `torchvision==0.25.0+cu130`; do not reintroduce CUDA 12.x package selectors.
- **bfloat16 everywhere**: do not use float16 on B200/DGX Spark hardware; always pass `bf16=True` in `TrainingArguments`.
- **FP8 models need `dtype=auto`**: pass `model.dtype=auto` on the CLI (or in YAML) when training on `Qwen3.6-27B-FP8`; pass `--dtype auto` to `export_model.py` when merging.
- **No wandb**: logging is TensorBoard only (`report_to="tensorboard"`). Do not add wandb dependencies.
- **Local output only**: no HuggingFace Hub push. Keep `push_to_hub=False` (default).
- **`dataset_format` key is informational**: all configs declare `dataset_format: "sharegpt"` for documentation purposes; `dataset_utils.py` does not read or validate it — ShareGPT is always assumed.
- **`DOCS_SYSTEM_PREAMBLE` is the single source of truth**: defined in `prepare_docs_dataset.py` and imported by `evaluate_model.py`. If changed, regenerate the training JSONL.

---

## Adding a New Task Type or Dataset

1. Add representative examples to `data/example_dataset.jsonl`.
2. If the task requires special tokenization (e.g., a new tool format), add a helper in `scripts/dataset_utils.py` and call it from `format_conversation_for_qwen()`.
3. Update the Dataset Format section in `README.md`.

## Adding a New Model

1. Duplicate a config in `configs/` and update `model.name`.
2. Check `lora.target_modules` — these differ per architecture (run `model.print_trainable_parameters()` to verify).
3. If the model needs a different Unsloth loader, update the `from_pretrained` call in `train.py`.
