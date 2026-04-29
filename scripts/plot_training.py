#!/usr/bin/env python3
"""
Standalone training progress plotter.

Reads TensorBoard event files from a training run and generates a multi-panel
matplotlib figure showing loss, eval loss, perplexity, learning rate, gradient
norm, and GPU memory (if recorded).

Usage:
    # Plot the latest run in an output directory
    python scripts/plot_training.py --log-dir outputs/lora_joint_abliterated/runs

    # Specify output PNG path
    python scripts/plot_training.py \\
        --log-dir outputs/lora_joint_abliterated/runs \\
        --output my_plot.png

    # Plot all runs merged (useful when training resumed from checkpoint)
    python scripts/plot_training.py \\
        --log-dir outputs/lora_joint_abliterated/runs \\
        --merge-runs
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path


def _ema(values: list[float], alpha: float = 0.1) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def _load_scalars(log_dir: str, merge_runs: bool = False) -> dict[str, list[tuple[int, float]]]:
    """Load scalar events from TensorBoard event files.

    Returns a dict mapping tag → list of (step, value) tuples.
    When merge_runs=True, all run subdirectories are combined and sorted by step.
    """
    from tensorboard.backend.event_processing import event_accumulator

    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Discover event files — either directly in log_dir or in run subdirs
    event_files = sorted(glob.glob(str(log_path / "**" / "*.tfevents*"), recursive=True))
    if not event_files:
        event_files = sorted(glob.glob(str(log_path / "*.tfevents*")))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {log_dir}")

    if not merge_runs:
        # Use only the latest event file (by filename, which encodes timestamp)
        event_files = [sorted(event_files)[-1]]

    print(f"Loading {len(event_files)} event file(s)...")

    result: dict[str, list[tuple[int, float]]] = {}
    for ef in event_files:
        ea = event_accumulator.EventAccumulator(ef)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            pairs = [(e.step, e.value) for e in ea.Scalars(tag)]
            result.setdefault(tag, []).extend(pairs)

    # Sort all series by step and deduplicate (keep last value per step)
    for tag in result:
        seen: dict[int, float] = {}
        for step, val in result[tag]:
            seen[step] = val
        result[tag] = sorted(seen.items())

    return result


def plot(
    log_dir: str,
    output: str = "training_progress.png",
    merge_runs: bool = False,
    title: str | None = None,
) -> None:
    """Generate a training progress plot from TensorBoard event files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    scalars = _load_scalars(log_dir, merge_runs=merge_runs)

    def _get(tag_candidates: list[str]) -> tuple[list[int], list[float]]:
        for tag in tag_candidates:
            if tag in scalars:
                pairs = scalars[tag]
                return [p[0] for p in pairs], [p[1] for p in pairs]
        return [], []

    train_steps, train_loss = _get(["train/loss"])
    eval_steps, eval_loss = _get(["eval/loss", "eval_loss"])
    lr_steps, lr_values = _get(["train/learning_rate"])
    gn_steps, gn_values = _get(["train/grad_norm"])

    max_step = max(
        (train_steps[-1] if train_steps else 0),
        (eval_steps[-1] if eval_steps else 0),
    )
    plot_title = title or f"Training Progress — step {max_step}"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(plot_title, fontsize=13, fontweight="bold")

    # --- Panel 1: Train + Eval Loss ---
    ax = axes[0, 0]
    if train_steps:
        ax.plot(train_steps, train_loss, alpha=0.35, color="steelblue", linewidth=0.8, label="train (raw)")
        ax.plot(train_steps, _ema(train_loss), color="steelblue", linewidth=1.8, label="train (EMA)")
    if eval_steps:
        ax.plot(eval_steps, eval_loss, color="coral", linewidth=1.8, linestyle="--",
                marker="o", markersize=4, label="eval")
    if not train_steps and not eval_steps:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Loss")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Eval Loss (detail) ---
    ax = axes[0, 1]
    if eval_steps:
        ax.plot(eval_steps, eval_loss, color="coral", linewidth=1.8, marker="o", markersize=5)
        ax.set_title("Eval Loss")
    else:
        ax.text(0.5, 0.5, "No eval data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Eval Loss")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Perplexity ---
    ax = axes[0, 2]
    if train_loss:
        ppl = [math.exp(min(l, 20)) for l in train_loss]
        ppl_ema = [math.exp(min(l, 20)) for l in _ema(train_loss)]
        ax.plot(train_steps, ppl, alpha=0.35, color="mediumseagreen", linewidth=0.8)
        ax.plot(train_steps, ppl_ema, color="mediumseagreen", linewidth=1.8, label="EMA")
        ax.set_title("Perplexity (train)")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Perplexity")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Learning Rate ---
    ax = axes[1, 0]
    if lr_values:
        ax.plot(lr_steps, lr_values, color="darkorange", linewidth=1.8)
        ax.set_title("Learning Rate")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
    else:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Gradient Norm ---
    ax = axes[1, 1]
    if gn_values:
        ax.plot(gn_steps, gn_values, alpha=0.5, color="mediumpurple", linewidth=0.8)
        ax.plot(gn_steps, _ema(gn_values, alpha=0.2), color="mediumpurple", linewidth=1.8, label="EMA")
        ax.set_title("Gradient Norm")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Gradient Norm")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # --- Panel 6: All available scalar tags (info panel) ---
    ax = axes[1, 2]
    ax.axis("off")
    lines = [f"Event files: {'merged' if merge_runs else 'latest only'}"]
    lines.append(f"Log dir: {Path(log_dir).name}")
    lines.append("")
    lines.append("Available tags:")
    for tag in sorted(scalars.keys()):
        pairs = scalars[tag]
        lines.append(f"  {tag}: {len(pairs)} points")
        if pairs:
            lines.append(f"    last value: {pairs[-1][1]:.4f} @ step {pairs[-1][0]}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Run Info")

    plt.tight_layout()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a training progress plot from TensorBoard event files."
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Directory containing TensorBoard event files (or a parent with run subdirectories).",
    )
    parser.add_argument(
        "--output",
        default="training_progress.png",
        help="Output PNG file path (default: training_progress.png).",
    )
    parser.add_argument(
        "--merge-runs",
        action="store_true",
        help="Merge data from all run subdirectories (useful when training was resumed).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title.",
    )
    args = parser.parse_args()
    plot(args.log_dir, output=args.output, merge_runs=args.merge_runs, title=args.title)


if __name__ == "__main__":
    main()
