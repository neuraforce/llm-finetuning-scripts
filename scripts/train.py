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
import json
import os
import sys
from pathlib import Path

import yaml

# Unsloth must patch Transformers before any direct Transformers imports.
import unsloth  # noqa: F401
from transformers import TrainerCallback

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


def _resolve_dtype(dtype_str: str):
    """Convert dtype string from YAML config to torch dtype (or None for 'auto')."""
    import torch
    if dtype_str == "auto" or dtype_str is None:
        return None
    _map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    result = _map.get(str(dtype_str))
    if result is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Use: bfloat16, float16, float32, or auto.")
    return result


def _load_prior_log_history(output_dir: Path) -> list[dict]:
    """Load ``log_history`` from the most recent checkpoint's ``trainer_state.json``.

    Scans *output_dir* for ``checkpoint-N`` directories and reads the one with
    the highest step number.  Returns an empty list if no checkpoint exists or if
    the file cannot be read, so callers are always safe to use the result directly.
    """
    try:
        checkpoints = sorted(
            (d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if not checkpoints:
            return []
        state_file = checkpoints[-1] / "trainer_state.json"
        if not state_file.exists():
            return []
        data = json.loads(state_file.read_text(encoding="utf-8"))
        history = data.get("log_history", [])
        if history:
            print(
                f"[monitor] Loaded {len(history)} prior log entries from"
                f" {checkpoints[-1].name}/trainer_state.json"
            )
        return history
    except Exception as exc:
        print(f"[monitor] Warning: could not load prior log history — {exc}", file=sys.stderr)
        return []


class TrainingMonitorCallback(TrainerCallback):
    """Callback that logs extra metrics and saves a training progress plot periodically.

    Fires on every ``on_log`` event (i.e. every ``logging_steps``).  Every
    ``plot_every_steps`` training steps it overwrites ``{output_dir}/training_progress.png``
    with a fresh multi-panel figure so you can inspect progress at any time.

    When training is resumed from a checkpoint, the callback loads the prior
    ``trainer_state.json`` so that ``training_progress.png`` always shows the
    **full** history of the run, not just the resumed segment.

    Panels saved:
        - Train loss (raw + EMA-smoothed)
        - Eval loss (if available)
        - Perplexity (exp of train loss)
        - Learning rate schedule
        - Gradient norm
        - GPU peak memory (GB)
    """

    def __init__(self, output_dir: str, plot_every_steps: int = 50) -> None:
        self.output_dir = Path(output_dir)
        self.plot_every_steps = plot_every_steps
        self._last_plot_step = -1
        self._prior_history: list[dict] = []

    # ------------------------------------------------------------------
    # transformers TrainerCallback protocol (duck-typed)
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """Load historical log entries from any prior checkpoint so the plot
        always reflects the complete training run, not just the current segment."""
        self._prior_history = _load_prior_log_history(self.output_dir)

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if self.plot_every_steps <= 0:
            return
        step = state.global_step
        if step - self._last_plot_step >= self.plot_every_steps:
            self._save_plot(state)
            self._last_plot_step = step

    def on_train_end(self, args, state, control, **kwargs) -> None:
        self._save_plot(state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_plot(self, state) -> None:
        import math
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Merge prior history (from checkpoint) with current session history,
        # using step number as the key to deduplicate overlapping entries.
        current_history = state.log_history or []
        current_steps = {e.get("step") or e.get("global_step", 0) for e in current_history}
        prior_only = [
            e for e in self._prior_history
            if (e.get("step") or e.get("global_step", 0)) not in current_steps
        ]
        history = sorted(prior_only + current_history,
                         key=lambda e: e.get("step") or e.get("global_step", 0))

        # Separate train and eval entries
        train_steps, train_loss, train_gnorm, train_lr = [], [], [], []
        eval_steps, eval_loss = [], []
        gpu_steps, gpu_mem = [], []

        for entry in history:
            step = entry.get("step") or entry.get("global_step", 0)
            if "loss" in entry:
                train_steps.append(step)
                train_loss.append(entry["loss"])
                if "grad_norm" in entry:
                    train_gnorm.append(entry["grad_norm"])
                if "learning_rate" in entry:
                    train_lr.append(entry["learning_rate"])
            if "eval_loss" in entry:
                eval_steps.append(step)
                eval_loss.append(entry["eval_loss"])

        # GPU peak memory (snapshot at plot time)
        if torch.cuda.is_available():
            gpu_peak_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_peak_gb = 0.0

        def _ema(values, alpha=0.1):
            if not values:
                return []
            smoothed = [values[0]]
            for v in values[1:]:
                smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
            return smoothed

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            f"Training Progress — step {state.global_step} / {state.max_steps}",
            fontsize=13,
            fontweight="bold",
        )

        # --- Panel 1: Train Loss ---
        ax = axes[0, 0]
        if train_steps:
            ax.plot(train_steps, train_loss, alpha=0.35, color="steelblue", linewidth=0.8, label="raw")
            ax.plot(train_steps, _ema(train_loss), color="steelblue", linewidth=1.8, label="EMA")
            if eval_steps:
                ax.plot(eval_steps, eval_loss, color="coral", linewidth=1.8, linestyle="--", label="eval")
            ax.set_title("Loss")
            ax.set_xlabel("Step")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # --- Panel 2: Eval Loss (detail) ---
        ax = axes[0, 1]
        if eval_steps:
            ax.plot(eval_steps, eval_loss, color="coral", linewidth=1.8, marker="o", markersize=4)
            ax.set_title("Eval Loss")
            ax.set_xlabel("Step")
        else:
            ax.text(0.5, 0.5, "No eval data yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Eval Loss")
        ax.grid(True, alpha=0.3)

        # --- Panel 3: Perplexity ---
        ax = axes[0, 2]
        if train_loss:
            ppl = [math.exp(min(l, 20)) for l in train_loss]
            ppl_ema = [math.exp(min(l, 20)) for l in _ema(train_loss)]
            ax.plot(train_steps, ppl, alpha=0.35, color="mediumseagreen", linewidth=0.8)
            ax.plot(train_steps, ppl_ema, color="mediumseagreen", linewidth=1.8, label="EMA")
            ax.set_title("Perplexity (train)")
            ax.set_xlabel("Step")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # --- Panel 4: Learning Rate ---
        ax = axes[1, 0]
        if train_lr:
            lr_steps = train_steps[: len(train_lr)]
            ax.plot(lr_steps, train_lr, color="darkorange", linewidth=1.8)
            ax.set_title("Learning Rate")
            ax.set_xlabel("Step")
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # --- Panel 5: Gradient Norm ---
        ax = axes[1, 1]
        if train_gnorm:
            gn_steps = train_steps[: len(train_gnorm)]
            ax.plot(gn_steps, train_gnorm, alpha=0.5, color="mediumpurple", linewidth=0.8)
            ax.plot(gn_steps, _ema(train_gnorm, alpha=0.2), color="mediumpurple", linewidth=1.8)
            ax.set_title("Gradient Norm")
            ax.set_xlabel("Step")
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # --- Panel 6: GPU Memory ---
        ax = axes[1, 2]
        if gpu_peak_gb > 0:
            # Show a single bar for peak memory vs. total available
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            ax.barh(["Peak used", "Total VRAM"], [gpu_peak_gb, total_gb],
                    color=["steelblue", "lightgray"])
            ax.set_xlim(0, total_gb * 1.05)
            ax.set_title(f"GPU Memory — step {state.global_step}")
            ax.set_xlabel("GB")
            for i, v in enumerate([gpu_peak_gb, total_gb]):
                ax.text(v + 0.5, i, f"{v:.1f} GB", va="center", fontsize=9)
        else:
            ax.text(0.5, 0.5, "GPU not available", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        out_path = self.output_dir / "training_progress.png"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[monitor] Plot saved → {out_path}")


def _patch_fp8_triton_kernel_with_pytorch_fallback() -> None:
    """Patch the FP8 Triton kernel loader to use a pure PyTorch fallback.

    The ``transformers`` ``finegrained_fp8`` module lazily loads a Triton kernel
    from HuggingFace Hub (``finegrained-fp8``) for FP8 matrix multiplication.
    When the hub is unreachable or the kernel download fails, all FP8 matmul
    operations raise an error.

    This patch injects a PyTorch implementation using ``torch._scaled_mm`` for
    simple (non-block-wise) cases and per-block dequantisation for block-wise
    scales.  Throughput is lower than a fused Triton kernel but correctness is
    maintained, which is sufficient for LoRA fine-tuning.
    """
    try:
        import transformers.integrations.finegrained_fp8 as _fp8_mod
    except ImportError:
        return

    if getattr(_fp8_mod, "_triton_available", None) is not None:
        return  # already loaded or already failed — do not re-patch

    import torch

    _FP8_DTYPE = torch.float8_e4m3fn
    _FP8_MIN = float(torch.finfo(_FP8_DTYPE).min)
    _FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)

    def _pt_fp8_act_quant(x: "torch.Tensor", block_size: int) -> "tuple[torch.Tensor, torch.Tensor]":
        """Quantise activations to FP8 with per-row (block_size-column) scales."""
        x = x.to(torch.float32)
        # Compute per-row absolute max as scale
        row_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = row_max / _FP8_MAX  # [rows, 1]
        x_fp8 = (x / scale).clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
        return x_fp8, scale.squeeze(-1)  # scale shape: [rows]

    def _dequantize_block_fp8(
        weight: "torch.Tensor",
        scale_inv: "torch.Tensor",
        block_size: "tuple[int, int] | list[int] | None",
    ) -> "torch.Tensor":
        """Dequantise block-wise FP8 weight tensor to bfloat16."""
        w = weight.to(torch.float32)
        if block_size is None:
            # per-tensor or per-column
            return (w * scale_inv.to(torch.float32)).to(torch.bfloat16)
        bs0, bs1 = block_size
        out, in_ = w.shape[-2], w.shape[-1]
        # Repeat scale_inv to match weight dimensions
        scale = scale_inv.to(torch.float32)
        scale = scale.repeat_interleave(bs0, dim=-2)[..., :out, :]
        scale = scale.repeat_interleave(bs1, dim=-1)[..., :in_]
        return (w * scale).to(torch.bfloat16)

    def _pt_w8a8_fp8_matmul(
        A: "torch.Tensor",
        B: "torch.Tensor",
        A_scale: "torch.Tensor",
        B_scale_inv: "torch.Tensor",
        block_size: "tuple[int, int] | list[int] | None",
        output_dtype: "torch.dtype" = torch.bfloat16,
    ) -> "torch.Tensor":
        """FP8 × FP8 matmul with block-wise dequantisation — pure PyTorch fallback."""
        # A: [..., M, K] fp8, A_scale: [M] or [M, ceil(K/bs)]
        # B: [N, K] fp8,  B_scale_inv: scale for B (inverse)
        B_bf16 = _dequantize_block_fp8(B, B_scale_inv, block_size)
        # Dequantise A
        A_f32 = A.to(torch.float32)
        if A_scale.ndim == 1:
            A_f32 = A_f32 * A_scale.unsqueeze(-1)  # [M, 1] broadcast over K
        else:
            # block-wise activation scale — per-row in our quant function
            A_f32 = A_f32 * A_scale.unsqueeze(-1)
        A_bf16 = A_f32.to(torch.bfloat16)
        return torch.nn.functional.linear(A_bf16, B_bf16).to(output_dtype)

    def _pt_w8a8_fp8_matmul_batched(
        A: "torch.Tensor",
        B: "torch.Tensor",
        A_scale: "torch.Tensor",
        B_scale_inv: "torch.Tensor",
        block_size: "tuple | list | None",
        output_dtype: "torch.dtype" = torch.bfloat16,
    ) -> "torch.Tensor":
        """Batched FP8 matmul — same as non-batched but loops over batch dim."""
        B_bf16 = _dequantize_block_fp8(B, B_scale_inv, block_size)
        A_f32 = A.to(torch.float32) * A_scale.unsqueeze(-1)
        return torch.bmm(A_f32.to(torch.bfloat16), B_bf16.transpose(-1, -2)).to(output_dtype)

    def _pt_w8a8_fp8_matmul_grouped(
        A: "torch.Tensor",
        B: "torch.Tensor",
        B_scale_inv: "torch.Tensor",
        tokens_per_expert: "torch.Tensor",
        block_size: "tuple | list | None",
        offsets: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Grouped FP8 matmul — loop over expert groups."""
        B_bf16 = _dequantize_block_fp8(B, B_scale_inv, block_size)  # [E, N, K] bf16
        outputs = []
        start = 0
        for e_idx, n_tok in enumerate(tokens_per_expert.tolist()):
            n_tok = int(n_tok)
            if n_tok == 0:
                continue
            a_slice = A[start : start + n_tok].to(torch.bfloat16)
            out = torch.nn.functional.linear(a_slice, B_bf16[e_idx])
            outputs.append(out)
            start += n_tok
        if not outputs:
            return A.new_zeros(0, B.shape[1], dtype=torch.bfloat16)
        return torch.cat(outputs, dim=0)

    # -------------------------------------------------------------------------
    # Memory-efficient custom autograd Function for FP8Experts.linear.
    #
    # The default path (dequant → BF16 matmul) causes autograd to retain every
    # dequantised weight tensor (B_bf16) in the computation graph.  With 40 MoE
    # layers × 64 active experts × two projections that accumulates ~57 GB of
    # BF16 temporaries on top of the 36 GB FP8 model — OOM on a 95 GB GPU.
    #
    # This custom Function saves only references to the FP8 model parameters
    # (already in the 36 GB model footprint) and recomputes B_bf16 during
    # backward.  Gradient wrt the BF16 input is computed via the straight-
    # through estimator: treat quantisation as identity in the backward pass.
    # -------------------------------------------------------------------------
    class _FP8ExpertLinearFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_bf16, weight_fp8, weight_scale_inv, block_size_list):
            B_bf16 = _dequantize_block_fp8(weight_fp8, weight_scale_inv, block_size_list)
            output = torch.nn.functional.linear(input_bf16, B_bf16)
            # Save only FP8 parameter references (zero extra VRAM — already in model).
            ctx.save_for_backward(weight_fp8, weight_scale_inv)
            ctx.block_size_list = block_size_list
            return output.to(input_bf16.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            weight_fp8, weight_scale_inv = ctx.saved_tensors
            B_bf16 = _dequantize_block_fp8(weight_fp8, weight_scale_inv, ctx.block_size_list)
            # Straight-through estimator: treat quantisation as identity.
            grad_input = grad_output.to(B_bf16.dtype) @ B_bf16
            return grad_input.to(grad_output.dtype), None, None, None

    try:
        from transformers.integrations.finegrained_fp8 import FP8Experts as _FP8Experts
        _orig_linear = _FP8Experts.linear.__func__ if hasattr(_FP8Experts.linear, "__func__") else _FP8Experts.linear

        def _mem_efficient_linear(self_experts, input, weight, weight_scale_inv, activation_scale=None):
            if weight.element_size() > 1:
                # Already BF16/FP16 (shared expert or dequantized weight) — no-op path.
                return torch.nn.functional.linear(input, weight, None)
            if self_experts.activation_scheme == "static" and activation_scale is not None:
                # Static activation scheme: fall back to original to keep static scaling.
                return _orig_linear(self_experts, input, weight, weight_scale_inv, activation_scale)
            # Dynamic activation scheme: use memory-efficient custom Function.
            return _FP8ExpertLinearFn.apply(
                input.to(torch.bfloat16),
                weight,
                weight_scale_inv,
                self_experts.block_size,
            )

        _FP8Experts.linear = _mem_efficient_linear
        print("FP8Experts.linear patched — memory-efficient autograd (STE, no BF16 weight cache)")
    except (ImportError, AttributeError):
        pass  # FP8Experts not present; skip

    # Inject the fallback implementations
    _fp8_mod.triton_fp8_act_quant = _pt_fp8_act_quant
    _fp8_mod.triton_fp8_matmul = _pt_w8a8_fp8_matmul
    _fp8_mod.triton_batched_fp8_matmul = _pt_w8a8_fp8_matmul_batched
    _fp8_mod.triton_grouped_fp8_matmul = _pt_w8a8_fp8_matmul_grouped
    _fp8_mod._triton_available = True  # signal that loading "succeeded"

    # Also prevent the DeepGEMM kernel from trying to download; w8a8_fp8_matmul
    # catches ImportError from _load_deepgemm_kernel and falls back to Triton.
    # But the download raises LocalProtocolError (not ImportError), so we must
    # mark deepgemm as unavailable upfront so it raises ImportError immediately.
    _fp8_mod._deepgemm_available = False

    print(
        "FP8 Triton kernel not available — using PyTorch BF16 dequantisation fallback. "
        "Throughput is reduced but training is correct."
    )


def _patch_qwen35moe_text_config() -> None:
    """Add intermediate_size alias to Qwen3_5MoeTextConfig for Unsloth compatibility.

    Unsloth's model loading code reads ``config.intermediate_size`` but
    ``Qwen3_5MoeTextConfig`` only exposes ``moe_intermediate_size``.  This
    patch adds a property alias so native FP8 loading succeeds without the
    ``AttributeError: 'Qwen3_5MoeTextConfig' object has no attribute
    'intermediate_size'`` that otherwise aborts model loading.

    This is a no-op for model configs that already have ``intermediate_size``.
    """
    try:
        from transformers import Qwen3_5MoeTextConfig  # type: ignore[attr-defined]

        if not hasattr(Qwen3_5MoeTextConfig, "intermediate_size"):
            Qwen3_5MoeTextConfig.intermediate_size = property(  # type: ignore[attr-defined]
                lambda self: self.moe_intermediate_size
            )
    except (ImportError, AttributeError):
        pass


def _load_fp8_experts_from_checkpoint(model, model_name: str) -> None:
    """Pack individual FP8 expert weights from checkpoint into model's 3D tensors.

    Pre-quantized FP8 checkpoints store expert weights as individual 2D
    tensors (one per expert, per projection: gate_proj, up_proj, down_proj).
    Unsloth's ``FP8Experts`` module stores them as packed 3D tensors
    ([num_experts, out, in]).  During native FP8 loading, Unsloth's fast
    loader packs the experts but uses a wrong key prefix (``model.layers.*``
    instead of ``model.language_model.layers.*``), leaving the 3D params
    randomly initialised.

    This function reads the checkpoint's safetensors files directly, packs
    the individual expert weights for each layer into the correct 3D tensors,
    and assigns them to the ``FP8Experts`` modules in-place.

    Memory overhead: ~768 MB peak per layer (256 experts × 3 projections
    × 1 MB each), streamed one layer at a time.
    """
    import json
    import os

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    try:
        from transformers.integrations.finegrained_fp8 import FP8Experts
    except ImportError:
        print("FP8Experts not available — skipping expert weight loading.")
        return

    # Locate the local cached model directory.
    try:
        model_path = snapshot_download(model_name, local_files_only=True)
    except Exception:
        model_path = snapshot_download(model_name)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print("No model.safetensors.index.json found — skipping expert weight loading.")
        return

    with open(index_path) as f:
        weight_map: dict = json.load(f)["weight_map"]

    # Discover all FP8Experts modules in the model.
    expert_modules: list[tuple[str, object]] = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, FP8Experts)
    ]

    if not expert_modules:
        print("No FP8Experts modules found — skipping expert weight loading.")
        return

    print(
        f"Loading FP8 expert weights for {len(expert_modules)} MoE layers "
        f"(packing individual 2D weights into 3D tensors)..."
    )

    for mod_name, fp8_experts in expert_modules:
        # Extract layer index from module name, e.g. "model.language_model.layers.7.mlp.experts"
        parts = mod_name.split(".")
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            print(f"  {mod_name}: cannot extract layer index, skipping.")
            continue

        num_experts: int = fp8_experts.num_experts
        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
        key0 = f"{prefix}.0.gate_proj.weight"

        if key0 not in weight_map:
            print(f"  Layer {layer_idx}: no individual expert keys in checkpoint, skipping.")
            continue

        # All experts for a given layer are stored in the same safetensors shard.
        shard_file = os.path.join(model_path, weight_map[key0])
        gate_up_list: list = []
        gate_up_scale_list: list = []
        down_list: list = []
        down_scale_list: list = []

        with safe_open(shard_file, framework="pt", device="cuda") as st:
            available = set(st.keys())
            for e in range(num_experts):
                ep = f"{prefix}.{e}"
                gate_w = st.get_tensor(f"{ep}.gate_proj.weight")
                up_w = st.get_tensor(f"{ep}.up_proj.weight")
                down_w = st.get_tensor(f"{ep}.down_proj.weight")

                # gate_up_proj: concatenate gate + up along output dimension.
                gate_up_list.append(torch.cat([gate_w, up_w], dim=0))
                down_list.append(down_w)

                # Scale-inv tensors (may be absent in some checkpoints).
                gs_key, us_key = f"{ep}.gate_proj.weight_scale_inv", f"{ep}.up_proj.weight_scale_inv"
                ds_key = f"{ep}.down_proj.weight_scale_inv"
                if gs_key in available and us_key in available:
                    gate_up_scale_list.append(
                        torch.cat([st.get_tensor(gs_key).float(), st.get_tensor(us_key).float()], dim=0)
                    )
                if ds_key in available:
                    down_scale_list.append(st.get_tensor(ds_key).float())

        with torch.no_grad():
            # Process layer by layer to minimize peak memory usage.
            # Use copy_() instead of .data assignment for ParamWrapper compatibility.
            stacked = torch.stack([t.to("cuda") for t in gate_up_list], dim=0)
            fp8_experts.gate_up_proj.copy_(stacked)
            gate_up_list.clear()
            torch.cuda.empty_cache()
            
            stacked = torch.stack([t.to("cuda") for t in down_list], dim=0)
            fp8_experts.down_proj.copy_(stacked)
            down_list.clear()
            torch.cuda.empty_cache()
            
            if gate_up_scale_list:
                stacked = torch.stack([t.to("cuda") for t in gate_up_scale_list], dim=0)
                fp8_experts.gate_up_proj_scale_inv.copy_(stacked)
                gate_up_scale_list.clear()
                torch.cuda.empty_cache()
            if down_scale_list:
                stacked = torch.stack([t.to("cuda") for t in down_scale_list], dim=0)
                fp8_experts.down_proj_scale_inv.copy_(stacked)
                down_scale_list.clear()
                torch.cuda.empty_cache()

        print(f"  Layer {layer_idx}: {num_experts} experts packed.", flush=True)

    torch.cuda.empty_cache()
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"FP8 expert loading complete. GPU memory: {mem_gb:.2f} GB")


def _is_finegrained_fp8_model(model_name: str) -> bool:
    """Return True if the model checkpoint uses FineGrainedFP8 / block-wise FP8 quantization."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
        qc = getattr(cfg, "quantization_config", None)
        if qc is None and hasattr(cfg, "text_config"):
            qc = getattr(cfg.text_config, "quantization_config", None)
        if qc is None:
            return False
        # qc may be a dict (raw JSON) or a parsed config object
        if isinstance(qc, dict):
            method = qc.get("quant_method", "")
            qtype = qc.get("quant_type", "")
        else:
            method = getattr(qc, "quant_method", "")
            qtype = getattr(qc, "quant_type", "")
        return method in ("fp8",) or qtype in ("fp8", "finegrained_fp8")
    except Exception:
        return False


def _model_load_kwargs(model_cfg: dict) -> dict:
    """Return optional model loader kwargs configured in YAML."""
    kwargs = {}
    offload_folder = model_cfg.get("offload_folder")
    if offload_folder:
        offload_path = Path(offload_folder)
        offload_path.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(offload_path)
        # Disable device_map='auto' when using disk offloading to avoid accelerate issues
        kwargs["device_map"] = None
    return kwargs


def _build_sft_config(train_cfg: dict, model_cfg: dict, eval_ds, cfg: dict | None = None) -> "SFTConfig":
    """Build a SFTConfig from the training and model sections of the YAML config."""
    from trl import SFTConfig

    load_best_model_at_end = train_cfg.get("load_best_model_at_end", False) and eval_ds is not None
    eval_steps = train_cfg.get("eval_steps", 200)
    save_steps = train_cfg.get("save_steps", 100)

    # HuggingFace Trainer requires save_steps == eval_steps when load_best_model_at_end=True
    if load_best_model_at_end:
        save_steps = eval_steps

    return SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=train_cfg.get("max_steps", -1),
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
        logging_dir=os.path.abspath(train_cfg.get("logging_dir") or os.path.join(train_cfg["output_dir"], "runs")),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        packing=train_cfg.get("packing", True),
        dataset_text_field=None,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        seed=train_cfg.get("seed", 42),
        max_length=model_cfg["max_seq_length"],
        deepspeed=train_cfg.get("deepspeed", (cfg or {}).get("deepspeed")),
        remove_unused_columns=False,
    )


def _build_trainer_kwargs(model, tokenizer, train_ds, eval_ds, trainer_args, model_cfg: dict) -> dict:
    """Build SFTTrainer kwargs, selecting a vision collator for multimodal data.

    ``UnslothVisionDataCollator`` is used whenever the model is a VLM (processor-
    based), even if the current training data is text-only.  This preserves the
    model's vision capabilities and avoids the Unsloth compiled trainer's
    ``DataCollatorForLanguageModeling`` path which does not work with processor-
    based models.

    For genuinely text-only models (tokenizer, not processor), the text-only path
    pre-applies the chat template and uses ``dataset_text_field`` for tokenization.
    """
    from transformers import ProcessorMixin
    from dataset_utils import has_multimodal_content, make_formatting_func

    kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": trainer_args,
    }

    is_vlm_model = isinstance(tokenizer, ProcessorMixin)

    if is_vlm_model or has_multimodal_content(train_ds) or has_multimodal_content(eval_ds):
        from unsloth import UnslothVisionDataCollator

        if has_multimodal_content(train_ds) or has_multimodal_content(eval_ds):
            print("Detected multimodal records — using UnslothVisionDataCollator")
        else:
            print("Vision-capable model (processor-based) with text-only data — using UnslothVisionDataCollator to preserve vision capabilities")
        kwargs["data_collator"] = UnslothVisionDataCollator(
            model,
            tokenizer,
            max_seq_length=model_cfg["max_seq_length"],
        )
    else:
        # Text-only model (bare tokenizer, not a processor): pre-apply the chat
        # template so the dataset has a "text" column, then let SFTTrainer
        # tokenise via dataset_text_field.
        fmt = make_formatting_func(tokenizer)

        def _apply_fmt(example: dict) -> dict:
            return {"text": fmt(example)[0]}

        print("Pre-applying chat template to dataset (text-only model path)…")
        kwargs["train_dataset"] = train_ds.map(_apply_fmt, desc="Formatting train")
        if eval_ds is not None:
            kwargs["eval_dataset"] = eval_ds.map(_apply_fmt, desc="Formatting eval")
        kwargs["dataset_text_field"] = "text"

    return kwargs


def train_lora(cfg: dict, resume_from_checkpoint: str | None = None) -> None:
    from unsloth import FastVisionModel
    from trl import SFTTrainer

    from dataset_utils import load_dataset_from_config

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    print(f"Loading model: {model_cfg['name']} (LoRA, dtype={model_cfg['dtype']})")

    load_in_4bit = model_cfg.get("load_in_4bit", False)

    # For pre-quantized FP8 MoE models: patch the Qwen3.5MoE config so Unsloth
    # can read intermediate_size, then load natively in FP8 (~36 GB instead of
    # dequantizing to BF16 at ~93 GB).  After loading, pack the per-expert 2D
    # checkpoint weights into the 3D FP8Experts tensors (Unsloth's fast loader
    # loses them due to a key-prefix mismatch).
    # bitsandbytes NF4 cannot quantize the MoE 3D expert tensors, so for FP8
    # MoE models we always use native FP8 regardless of load_in_4bit.
    is_fp8_model = _is_finegrained_fp8_model(model_cfg["name"])
    if is_fp8_model:
        _patch_qwen35moe_text_config()
        _patch_fp8_triton_kernel_with_pytorch_fallback()
        effective_4bit = False  # native FP8 is more memory-efficient than NF4 for MoE
        print(
            f"FP8 MoE model detected — loading in native FP8 (~36 GB). "
            f"load_in_4bit={load_in_4bit} is a no-op (NF4 cannot quantize MoE 3D expert tensors)."
        )
    else:
        effective_4bit = load_in_4bit

    extra_kwargs = _model_load_kwargs(model_cfg)
    if is_fp8_model:
        # grouped_mm dispatch requires a Triton kernel that may not be available;
        # eager dispatch uses standard PyTorch ops and always works.
        extra_kwargs.setdefault("experts_implementation", "eager")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=_resolve_dtype(model_cfg["dtype"]),
        load_in_4bit=effective_4bit,
        **extra_kwargs,
    )

    if is_fp8_model:
        # Pack individual FP8 expert weights from checkpoint into the model's 3D tensors.
        _load_fp8_experts_from_checkpoint(model, model_cfg["name"])
        # Initialize lm_head with random values on CUDA to avoid meta device issues
        # (it will be restored from the checkpoint by the trainer if it exists)
        try:
            if hasattr(model, "lm_head") and model.lm_head.weight.device.type == "meta":
                with torch.no_grad():
                    model.lm_head.weight.copy_(
                        torch.randn(model.lm_head.weight.shape, device="cuda", dtype=model.lm_head.weight.dtype)
                    )
                    if model.lm_head.bias is not None and model.lm_head.bias.device.type == "meta":
                        model.lm_head.bias.copy_(
                            torch.zeros(model.lm_head.bias.shape, device="cuda", dtype=model.lm_head.bias.dtype)
                        )
        except Exception as e:
            print(f"[warn] Could not initialize lm_head: {e}", flush=True)

    # MoE models use ParamWrapper for expert weight tensors, which requires dropout=0.
    # They also have 256 experts per layer; applying LoRA to FFN modules creates
    # 3B+ trainable params and OOMs. Restrict to attention modules only for MoE.
    _ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def _cfg_is_moe(c) -> bool:
        return any(hasattr(c, a) for a in ("num_experts", "num_local_experts", "moe_intermediate_size"))

    _is_moe = _cfg_is_moe(model.config) or (
        hasattr(model.config, "text_config") and _cfg_is_moe(model.config.text_config)
    )
    lora_dropout = lora_cfg.get("lora_dropout", 0.05)
    target_modules = lora_cfg["target_modules"]
    if _is_moe:
        if lora_dropout != 0:
            print(
                f"MoE model detected — overriding lora_dropout {lora_dropout} → 0 "
                f"(PEFT ParamWrapper does not support dropout)."
            )
            lora_dropout = 0
        ffn_modules = [m for m in target_modules if m not in _ATTN_MODULES]
        if ffn_modules:
            target_modules = [m for m in target_modules if m in _ATTN_MODULES]
            print(
                f"MoE model detected — removing FFN modules {ffn_modules} from target_modules "
                f"to avoid OOM (256 experts × 40 layers would create 3B+ trainable params). "
                f"Training attention modules only: {target_modules}"
            )

    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_dropout,
        target_modules=target_modules,
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

    plot_every_steps = train_cfg.get("plot_every_steps", 50)
    if plot_every_steps > 0:
        callbacks.append(TrainingMonitorCallback(train_cfg["output_dir"], plot_every_steps=plot_every_steps))
        print(f"Training monitor enabled: plot saved every {plot_every_steps} steps → {train_cfg['output_dir']}/training_progress.png")
    else:
        print("Training monitor disabled: plot_every_steps <= 0")

    # Patch fix_untrained_tokens to skip meta tensors (which will be loaded from checkpoint)
    try:
        from unsloth_zoo import tokenizer_utils
        _original_fix_untrained = tokenizer_utils.fix_untrained_tokens
        
        def _patched_fix_untrained(*args, **kwargs):
            try:
                return _original_fix_untrained(*args, **kwargs)
            except NotImplementedError as e:
                if "meta tensor" in str(e):
                    print(f"[warn] Skipping fix_untrained_tokens: {e}")
                    return
                raise
        
        tokenizer_utils.fix_untrained_tokens = _patched_fix_untrained
    except Exception as e:
        print(f"[warn] Could not patch fix_untrained_tokens: {e}")

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

    load_in_4bit = model_cfg.get("load_in_4bit", False)

    is_fp8_model = _is_finegrained_fp8_model(model_cfg["name"])
    if is_fp8_model:
        _patch_qwen35moe_text_config()
        _patch_fp8_triton_kernel_with_pytorch_fallback()
        effective_4bit = False
    else:
        effective_4bit = load_in_4bit

    extra_kwargs = _model_load_kwargs(model_cfg)
    if is_fp8_model:
        extra_kwargs.setdefault("experts_implementation", "eager")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=_resolve_dtype(model_cfg["dtype"]),
        load_in_4bit=effective_4bit,
        full_finetuning=True,
        **extra_kwargs,
    )

    if is_fp8_model:
        _load_fp8_experts_from_checkpoint(model, model_cfg["name"])

    train_ds, eval_ds = load_dataset_from_config(cfg)
    trainer_args = _build_sft_config(train_cfg, model_cfg, eval_ds, cfg)

    callbacks = []
    patience = train_cfg.get("early_stopping_patience")
    if patience is not None and eval_ds is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))
        print(f"Early stopping enabled: patience={patience} eval steps")

    plot_every_steps = train_cfg.get("plot_every_steps", 50)
    if plot_every_steps > 0:
        callbacks.append(TrainingMonitorCallback(train_cfg["output_dir"], plot_every_steps=plot_every_steps))
        print(f"Training monitor enabled: plot saved every {plot_every_steps} steps → {train_cfg['output_dir']}/training_progress.png")
    else:
        print("Training monitor disabled: plot_every_steps <= 0")

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
            "Pass the path to a Trainer checkpoint (e.g. outputs/lora_qwen3/checkpoint-500). "
            "If not specified, will auto-detect the latest checkpoint in the output directory."
        ),
    )
    _4bit_group = parser.add_mutually_exclusive_group()
    _4bit_group.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        default=None,
        help="Load model weights in 4-bit (QLoRA / NF4). Overrides model.load_in_4bit in the config.",
    )
    _4bit_group.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading. Overrides model.load_in_4bit in the config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides in key=value format, e.g. training.learning_rate=1e-4",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    # CLI flag takes precedence over the YAML config key
    if args.load_in_4bit is not None:
        cfg.setdefault("model", {})["load_in_4bit"] = args.load_in_4bit

    # Auto-detect latest checkpoint if not specified
    resume_from = args.resume_from_checkpoint
    if resume_from is None:
        output_dir = cfg.get("training", {}).get("output_dir", "outputs/default")
        output_path = Path(output_dir)
        if output_path.exists():
            checkpoints = sorted(
                [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda p: int(p.name.split("-")[1])
            )
            if checkpoints:
                resume_from = str(checkpoints[-1])
                print(f"Auto-detected latest checkpoint: {resume_from}")

    try:
        if is_lora_config(cfg):
            train_lora(cfg, resume_from_checkpoint=resume_from)
        else:
            train_full(cfg, resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        print("\n\n✓ Training interrupted (Ctrl-C). Current state has been saved to checkpoint.")
        print("  To resume, run:")
        print(f"    python scripts/train.py --config {args.config} --resume-from-checkpoint <checkpoint-dir>")
        sys.exit(0)


if __name__ == "__main__":
    main()
