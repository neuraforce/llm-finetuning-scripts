#!/usr/bin/env python3
"""
Unified fine-tuning entry point for Qwen 3.6 using Unsloth.

Usage:
    # LoRA fine-tuning
    python scripts/train.py --config configs/lora_qwen3.yaml

    # Full fine-tuning
    python scripts/train.py --config configs/full_ft_qwen3.yaml

    # Override config values on the CLI
    python scripts/train.py --config configs/lora_qwen3.yaml \\
        training.num_train_epochs=5 \\
        data.train_file=data/my_train.jsonl

    # With accelerate (e.g. DeepSpeed)
    accelerate launch --config_file configs/accelerate_config.yaml \\
        scripts/train.py --config configs/full_ft_qwen3.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure scripts/ is on sys.path so sibling modules can be imported
# regardless of the working directory the script is invoked from.
_SCRIPTS_DIR = str(Path(__file__).parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _deep_set(d: dict, dotted_key: str, value: str) -> None:
    """Set a nested dict value using a dot-separated key, e.g. 'training.lr'."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Attempt numeric coercion so overrides like num_train_epochs=5 work correctly
    try:
        value = yaml.safe_load(value)
    except Exception:
        pass
    d[keys[-1]] = value


def load_config(config_path: str, overrides: list[str]) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for override in overrides:
        if "=" not in override:
            print(f"Warning: ignoring malformed override '{override}' (expected key=value)")
            continue
        key, _, val = override.partition("=")
        _deep_set(cfg, key.strip(), val.strip())
    return cfg


def is_lora_config(cfg: dict) -> bool:
    return "lora" in cfg


def _build_sft_config(train_cfg: dict, model_cfg: dict, eval_ds, cfg: dict | None = None) -> "SFTConfig":
    """Build a SFTConfig from the training and model sections of the YAML config."""
    from trl import SFTConfig

    load_best_model_at_end = train_cfg.get("load_best_model_at_end", False) and eval_ds is not None
    eval_steps = train_cfg.get("eval_steps", 200)
    save_steps = train_cfg.get("save_steps", 500)

    # HuggingFace Trainer requires save_steps == eval_steps when load_best_model_at_end=True
    if load_best_model_at_end:
        save_steps = eval_steps

    return SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        optim=train_cfg.get("optim", "adamw_8bit"),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=eval_steps if eval_ds is not None else None,
        save_steps=save_steps,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
        report_to=train_cfg.get("report_to", "tensorboard"),
        logging_dir=train_cfg.get("logging_dir"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        packing=train_cfg.get("packing", True),
        dataset_text_field=None,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        seed=train_cfg.get("seed", 42),
        max_seq_length=model_cfg["max_seq_length"],
        deepspeed=train_cfg.get("deepspeed", (cfg or {}).get("deepspeed")),
    )


def _build_trainer_kwargs(model, tokenizer, train_ds, eval_ds, trainer_args, model_cfg: dict) -> dict:
    """Build SFTTrainer kwargs, selecting a vision collator for multimodal data."""
    from dataset_utils import has_multimodal_content, make_formatting_func

    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": trainer_args,
    }

    if has_multimodal_content(train_ds) or has_multimodal_content(eval_ds):
        from unsloth import UnslothVisionDataCollator

        print("Detected multimodal records — using UnslothVisionDataCollator")
        kwargs["data_collator"] = UnslothVisionDataCollator(
            model,
            tokenizer,
            max_seq_length=model_cfg["max_seq_length"],
        )
    else:
        kwargs["formatting_func"] = make_formatting_func(tokenizer)

    return kwargs


def train_lora(cfg: dict, resume_from_checkpoint: str | None = None) -> None:
    from unsloth import FastVisionModel
    from trl import SFTTrainer

    from dataset_utils import load_dataset_from_config

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    print(f"Loading model: {model_cfg['name']} (LoRA, dtype={model_cfg['dtype']})")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=None if model_cfg["dtype"] == "auto" else model_cfg["dtype"],
        load_in_4bit=model_cfg.get("load_in_4bit", False),
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        use_rslora=lora_cfg.get("use_rslora", True),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
    )

    train_ds, eval_ds = load_dataset_from_config(cfg)
    trainer_args = _build_sft_config(train_cfg, model_cfg, eval_ds, cfg)

    callbacks = []
    patience = train_cfg.get("early_stopping_patience")
    if patience is not None and eval_ds is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))
        print(f"Early stopping enabled: patience={patience} eval steps")

    trainer = SFTTrainer(
        **_build_trainer_kwargs(model, tokenizer, train_ds, eval_ds, trainer_args, model_cfg),
        callbacks=callbacks or None,
    )

    print("Starting LoRA training...")
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(
        f"Training complete — loss: {result.training_loss:.4f}, "
        f"runtime: {result.metrics.get('train_runtime', 0):.1f}s"
    )

    adapter_dir = Path(train_cfg["output_dir"]) / "adapter"
    print(f"Saving LoRA adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    export_cfg = cfg.get("export", {})
    if export_cfg.get("merge_adapter", False):
        from export_model import merge_and_save
        merged_dir = export_cfg.get("merged_output_dir", str(Path(train_cfg["output_dir"]) / "merged"))
        merge_and_save(model, tokenizer, merged_dir)


def train_full(cfg: dict, resume_from_checkpoint: str | None = None) -> None:
    from unsloth import FastVisionModel
    from trl import SFTTrainer

    from dataset_utils import load_dataset_from_config

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    print(f"Loading model: {model_cfg['name']} (full FT, dtype={model_cfg['dtype']})")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=None if model_cfg["dtype"] == "auto" else model_cfg["dtype"],
        load_in_4bit=model_cfg.get("load_in_4bit", False),
        full_finetuning=True,
    )

    train_ds, eval_ds = load_dataset_from_config(cfg)
    trainer_args = _build_sft_config(train_cfg, model_cfg, eval_ds, cfg)

    callbacks = []
    patience = train_cfg.get("early_stopping_patience")
    if patience is not None and eval_ds is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))
        print(f"Early stopping enabled: patience={patience} eval steps")

    trainer = SFTTrainer(
        **_build_trainer_kwargs(model, tokenizer, train_ds, eval_ds, trainer_args, model_cfg),
        callbacks=callbacks or None,
    )

    print("Starting full fine-tuning...")
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(
        f"Training complete — loss: {result.training_loss:.4f}, "
        f"runtime: {result.metrics.get('train_runtime', 0):.1f}s"
    )

    print(f"Saving model to {train_cfg['output_dir']}")
    model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen 3.6 with Unsloth (LoRA or full FT)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Resume training from a saved checkpoint directory. "
            "Pass the path to a Trainer checkpoint (e.g. outputs/lora_qwen3/checkpoint-500)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides in key=value format, e.g. training.learning_rate=1e-4",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    if is_lora_config(cfg):
        train_lora(cfg, resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        train_full(cfg, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
