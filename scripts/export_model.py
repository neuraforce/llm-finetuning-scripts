#!/usr/bin/env python3
"""
Merge a LoRA adapter into the base model weights and save locally.

Usage:
    # After training with LoRA (BF16 model):
    python scripts/export_model.py \\
        --base-model Qwen/Qwen3.6-27B \\
        --adapter outputs/lora_qwen3/adapter \\
        --output outputs/lora_qwen3_merged

    # FP8 model — pass --dtype auto so weights are loaded in their native format:
    python scripts/export_model.py \\
        --base-model Qwen/Qwen3.6-27B-FP8 \\
        --adapter outputs/lora_qwen3/adapter \\
        --output outputs/lora_qwen3_merged \\
        --dtype auto

    # Export to GGUF (requires llama.cpp conversion tools):
    python scripts/export_model.py \\
        --base-model Qwen/Qwen3.6-27B \\
        --adapter outputs/lora_qwen3/adapter \\
        --output outputs/lora_qwen3_merged \\
        --max-seq-length 8192 \\
        --gguf q8_0
"""

from __future__ import annotations

import argparse
from pathlib import Path


def merge_and_save(model, tokenizer, output_dir: str) -> None:
    """
    Merge a loaded PEFT model's LoRA adapter into base weights and save.
    Called programmatically from train.py when export.merge_adapter=true.
    """
    print(f"Merging LoRA adapter into base weights → {output_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Merge complete.")


def merge_from_disk(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
    max_seq_length: int = 8192,
    dtype: str = "bfloat16",
) -> None:
    """
    Load a base model + saved adapter from disk, merge, and save.
    Used when calling this script standalone after training.

    Pass dtype="auto" for FP8 models (e.g. Qwen/Qwen3.6-27B-FP8) so that
    Unsloth preserves the native FP8 weights instead of upcasting to BF16.
    """
    from unsloth import FastVisionModel

    resolved_dtype = None if dtype == "auto" else dtype
    print(f"Loading base model: {base_model} (dtype={dtype})")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=resolved_dtype,
        load_in_4bit=False,
    )

    print(f"Loading LoRA adapter: {adapter_dir}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_dir)

    print(f"Merging and saving to: {output_dir}")
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Merge complete.")


def export_to_gguf(merged_dir: str, quant_type: str) -> None:
    """
    Convert a merged HuggingFace model to GGUF using Unsloth's built-in method.

    quant_type examples: "q8_0", "q4_k_m", "f16"
    See https://github.com/unslothai/unsloth for the full list.

    The merged model is always loaded in bfloat16 regardless of the original
    training dtype — after merging, adapters are baked in at BF16 precision.
    """
    try:
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(merged_dir, dtype="bfloat16")
        gguf_dir = str(Path(merged_dir).parent / (Path(merged_dir).name + f"_{quant_type}"))
        print(f"Exporting GGUF ({quant_type}) → {gguf_dir}")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant_type)
        print("GGUF export complete.")
    except AttributeError:
        print(
            "GGUF export not available via this Unsloth version. "
            "Use llama.cpp convert scripts manually."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export / merge LoRA adapter into base model")
    parser.add_argument("--base-model", required=True, help="HF model ID or local path of the base model")
    parser.add_argument("--adapter", required=True, help="Path to the saved LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output directory for the merged model")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help=(
            "Model dtype for loading the base model. "
            "Use 'auto' for FP8 models (e.g. Qwen/Qwen3.6-27B-FP8) to preserve "
            "native FP8 weights. Default: bfloat16"
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length to pass to the model loader (default: 8192)",
    )
    parser.add_argument(
        "--gguf",
        metavar="QUANT",
        default=None,
        help="Also export to GGUF with this quantization type (e.g. q8_0, q4_k_m, f16)",
    )
    args = parser.parse_args()

    merge_from_disk(args.base_model, args.adapter, args.output, max_seq_length=args.max_seq_length, dtype=args.dtype)

    if args.gguf:
        export_to_gguf(args.output, args.gguf)


if __name__ == "__main__":
    main()
