#!/usr/bin/env python3
"""
Evaluate a deployed Qwen3.6 model against the ~/docs/ dataset.

For each .md/.txt pair the markdown content is used as the document context
(appended to a tight extraction system prompt) and every expanded question from
the .txt file is sent to the model.  Responses are compared to ground-truth
answers using multiple metrics and results are written to a JSONL log file and
a summary PNG plot.

By default chain-of-thought (thinking) is disabled so the model answers
directly without <think> overhead consuming the token budget.  Pass --thinking
to re-enable it if needed.

Usage examples
--------------
# Smoke-test: 10 samples only
python scripts/evaluate_model.py --max-samples 10

# Full run with higher concurrency
python scripts/evaluate_model.py --concurrency 6

# Enable chain-of-thought (off by default)
python scripts/evaluate_model.py --thinking

# Custom endpoint / model
python scripts/evaluate_model.py --endpoint http://192.168.35.8:8002 --model qwen3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tqdm.asyncio import tqdm as atqdm

# Import from the same directory at runtime
sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import compute_all, strip_thinking
from prepare_docs_dataset import DOCS_SYSTEM_PREAMBLE, expand_options, find_pairs, parse_txt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_KEYS = ("exact_match", "token_f1", "rouge1_f", "rouge2_f", "rougeL_f", "edit_sim")
METRIC_LABELS = ("Exact\nMatch", "Token\nF1", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Edit\nSim")
PRIMARY_METRIC = "rougeL_f"

# DOCS_SYSTEM_PREAMBLE is imported from prepare_docs_dataset so that training
# data and evaluation always use the exact same system prompt prefix.

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate vLLM-deployed model against ~/docs/ Q&A dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--docs-dir", type=Path, default=Path("~/docs"),
                   help="Directory containing .md/.txt pairs (default: ~/docs)")
    p.add_argument("--endpoint", default="http://192.168.35.8:8002",
                   help="vLLM base URL (default: http://192.168.35.8:8002)")
    p.add_argument("--model", default="qwen3",
                   help="Model ID served by vLLM (default: qwen3)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Directory for log and plot files (default: outputs/)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max parallel requests (default: 4)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Stop after N samples — 0 means all (default: 0)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy (default: 0.0)")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Max tokens in model response (default: 1024)")
    p.add_argument("--thinking", action="store_true", default=False,
                   help="Enable chain-of-thought (thinking mode). Default: off for evaluation.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout in seconds (default: 120)")
    p.add_argument("--retries", type=int, default=3,
                   help="Retry attempts per request (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


async def _call_model(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    retries: int,
    timeout: float,
    enable_thinking: bool = False,
) -> tuple[str, float]:
    """
    Send a chat completion request and return (raw_response_text, latency_seconds).
    Raises RuntimeError after all retries are exhausted.

    enable_thinking=False suppresses Qwen3's chain-of-thought via vLLM's
    chat_template_kwargs — cleaner than appending /no_think to the system prompt.
    """
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    }

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            t0 = time.monotonic()
            resp = await client.post(url, json=payload, timeout=timeout)
            latency = time.monotonic() - t0
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            return text, round(latency, 3)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                wait = 2 ** attempt
                await asyncio.sleep(wait)

    raise RuntimeError(f"Request failed after {retries} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Build task list
# ---------------------------------------------------------------------------


def _build_tasks(docs_dir: Path, max_samples: int) -> list[dict]:
    """
    Return a list of task dicts, one per expanded Q&A sample.

    Each dict contains: doc_name, system, question, ground_truth.
    The system message is: SYSTEM_PREAMBLE + document markdown content.
    """
    pairs = find_pairs(docs_dir)
    if not pairs:
        print(f"No .md/.txt pairs found in {docs_dir}", file=sys.stderr)
        sys.exit(1)

    tasks: list[dict] = []
    for md_path, txt_path in pairs:
        doc_content = md_path.read_text(encoding="utf-8")
        system = DOCS_SYSTEM_PREAMBLE + doc_content
        txt_content = txt_path.read_text(encoding="utf-8")
        raw_samples = parse_txt(txt_content)
        for raw_question, answer in raw_samples:
            for question in expand_options(raw_question):
                tasks.append({
                    "doc_name": md_path.stem,
                    "system": system,
                    "question": question,
                    "ground_truth": answer,
                })
                if max_samples and len(tasks) >= max_samples:
                    return tasks
    return tasks


# ---------------------------------------------------------------------------
# Async evaluation loop
# ---------------------------------------------------------------------------


async def _run_eval(
    tasks: list[dict],
    *,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    retries: int,
    timeout: float,
    concurrency: int,
    enable_thinking: bool,
    log_path: Path,
    plot_path: Path,
) -> list[dict]:
    """
    Evaluate all tasks asynchronously.  Results are appended to log_path after
    each sample so the log survives a crash mid-run.

    When concurrency > 1, results are written in completion order, not task
    order — the log is therefore nondeterministic in sequence across runs.

    Every 100 samples, updates the metrics plot (overwrites existing file).
    Returns the list of result dicts.
    """
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    results: list[dict] = []

    limits = httpx.Limits(max_connections=concurrency + 2, max_keepalive_connections=concurrency)
    log_file = log_path.open("a", encoding="utf-8")
    try:
        async with httpx.AsyncClient(limits=limits) as client:

            async def process(task: dict) -> dict:
                async with sem:
                    ts = datetime.now(timezone.utc).isoformat()
                    try:
                        raw_response, latency = await _call_model(
                            client,
                            endpoint=endpoint,
                            model=model,
                            system=task["system"],
                            user=task["question"],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            seed=seed,
                            retries=retries,
                            timeout=timeout,
                            enable_thinking=enable_thinking,
                        )
                        metrics = compute_all(raw_response, task["ground_truth"])
                        record = {
                            "doc": task["doc_name"],
                            "question": task["question"],
                            "ground_truth": task["ground_truth"],
                            "raw_response": raw_response,
                            "response": strip_thinking(raw_response),
                            **metrics,
                            "latency_s": latency,
                            "status": "ok",
                            "timestamp": ts,
                        }
                    except Exception as exc:  # noqa: BLE001
                        record = {
                            "doc": task["doc_name"],
                            "question": task["question"],
                            "ground_truth": task["ground_truth"],
                            "raw_response": "",
                            "response": "",
                            **{k: None for k in METRIC_KEYS},
                            "latency_s": None,
                            "status": f"error: {exc}",
                            "timestamp": ts,
                        }
                    async with write_lock:
                        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log_file.flush()
                        results.append(record)
                        if len(results) % 100 == 0:
                            _make_plot(results, plot_path)
                            print(f"Updated metrics plot at {len(results)} samples")
                    return record

            coros = [process(t) for t in tasks]
            await atqdm.gather(*coros, desc="Evaluating", unit="sample")
    finally:
        log_file.close()

    return list(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _truncate(s: str, max_len: int = 45) -> str:
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def _make_plot(results: list[dict], plot_path: Path) -> None:
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        print("No successful results to plot.", file=sys.stderr)
        return

    # --- Aggregate per-document ---
    doc_scores: dict[str, list[float]] = {}
    for r in ok_results:
        doc_scores.setdefault(r["doc"], []).append(r[PRIMARY_METRIC])
    doc_names = list(doc_scores.keys())
    doc_means = [float(np.mean(doc_scores[d])) for d in doc_names]
    # Sort by score ascending (worst at top in horizontal bar)
    order = np.argsort(doc_means)
    doc_names_sorted = [_truncate(doc_names[i]) for i in order]
    doc_means_sorted = [doc_means[i] for i in order]

    colors = ["#e74c3c" if v < 0.5 else "#f39c12" if v < 0.8 else "#27ae60"
              for v in doc_means_sorted]

    # --- All primary metric values ---
    primary_vals = [r[PRIMARY_METRIC] for r in ok_results]
    em_vals = [r["exact_match"] for r in ok_results]

    # --- Per-metric global means ---
    metric_means = [float(np.mean([r[k] for r in ok_results if r[k] is not None]))
                    for k in METRIC_KEYS]

    # --- Latencies ---
    latencies = [r["latency_s"] for r in ok_results if r["latency_s"] is not None]

    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Model Evaluation — {len(ok_results)} samples / {len(doc_names)} documents\n"
        f"(Primary metric: ROUGE-L, mean = {np.mean(primary_vals):.3f})",
        fontsize=14,
        fontweight="bold",
    )

    # --- Subplot 1: Per-document ROUGE-L ---
    ax1 = axes[0, 0]
    bars = ax1.barh(range(len(doc_names_sorted)), doc_means_sorted, color=colors, edgecolor="white")
    ax1.set_yticks(range(len(doc_names_sorted)))
    ax1.set_yticklabels(doc_names_sorted, fontsize=8)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Mean ROUGE-L")
    ax1.set_title("Per-Document ROUGE-L (mean)")
    ax1.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 threshold")
    ax1.axvline(0.8, color="gray", linestyle=":",  linewidth=0.8, label="0.8 threshold")
    ax1.legend(fontsize=7, loc="lower right")
    for bar, val in zip(bars, doc_means_sorted):
        ax1.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=7)

    # --- Subplot 2: Score distributions ---
    ax2 = axes[0, 1]
    bins = np.linspace(0, 1, 26)
    ax2.hist(primary_vals, bins=bins, alpha=0.65, label="ROUGE-L", color="#2980b9", edgecolor="white")
    ax2.hist(em_vals, bins=bins, alpha=0.65, label="Exact Match", color="#e67e22", edgecolor="white")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("# Samples")
    ax2.set_title("Score Distribution (all samples)")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # --- Subplot 3: Metric comparison ---
    ax3 = axes[1, 0]
    bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#1abc9c"]
    bars3 = ax3.bar(METRIC_LABELS, metric_means, color=bar_colors, edgecolor="white", width=0.6)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Mean Score")
    ax3.set_title("Global Metric Comparison (mean over all samples)")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    for bar, val in zip(bars3, metric_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- Subplot 4: Latency distribution ---
    ax4 = axes[1, 1]
    if latencies:
        ax4.hist(latencies, bins=30, color="#8e44ad", edgecolor="white", alpha=0.8)
        ax4.axvline(float(np.median(latencies)), color="red", linestyle="--",
                    linewidth=1.5, label=f"median {np.median(latencies):.1f}s")
        ax4.axvline(float(np.mean(latencies)), color="orange", linestyle=":",
                    linewidth=1.5, label=f"mean {np.mean(latencies):.1f}s")
        ax4.legend(fontsize=8)
    ax4.set_xlabel("Latency (s)")
    ax4.set_ylabel("# Samples")
    ax4.set_title("Response Latency Distribution")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]
    n = len(results)
    print(f"\n{'='*60}")
    print(f"  Evaluation complete: {n} samples, {len(ok)} ok, {len(errors)} errors")
    if ok:
        for key, label in zip(METRIC_KEYS, METRIC_LABELS):
            vals = [r[key] for r in ok if r[key] is not None]
            mean_v = float(np.mean(vals)) if vals else 0.0
            print(f"  {label.replace(chr(10), ' '):14s}  {mean_v:.4f}")
    if errors:
        print(f"\n  First error: {errors[0]['status']}")
    print(f"{'='*60}\n")


def _write_summary(results: list[dict], log_path: Path) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    summary: dict = {
        "_summary": True,
        "n_total": len(results),
        "n_ok": len(ok),
        "n_error": len(results) - len(ok),
    }
    for key in METRIC_KEYS:
        vals = [r[key] for r in ok if r[key] is not None]
        summary[f"{key}_mean"] = round(float(np.mean(vals)), 6) if vals else None
    if ok:
        latencies = [r["latency_s"] for r in ok if r["latency_s"] is not None]
        if latencies:
            summary["latency_mean_s"] = round(float(np.mean(latencies)), 3)
            summary["latency_median_s"] = round(float(np.median(latencies)), 3)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    docs_dir = args.docs_dir.expanduser()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"eval_{stamp}.jsonl"
    plot_path = output_dir / f"eval_{stamp}_plot.png"

    print(f"Docs dir   : {docs_dir}")
    print(f"Endpoint   : {args.endpoint}")
    print(f"Model      : {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Thinking   : {'on' if args.thinking else 'off (default)'}")
    print(f"Log file   : {log_path}")
    print(f"Plot file  : {plot_path}")
    print()

    tasks = _build_tasks(docs_dir, max_samples=args.max_samples)
    print(f"Total samples to evaluate: {len(tasks)}")
    if args.max_samples:
        print(f"(capped at --max-samples {args.max_samples})")
    print()

    results = asyncio.run(
        _run_eval(
            tasks,
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            retries=args.retries,
            timeout=args.timeout,
            concurrency=args.concurrency,
            enable_thinking=args.thinking,
            log_path=log_path,
            plot_path=plot_path,
        )
    )

    _write_summary(results, log_path)
    _print_summary(results)
    _make_plot(results, plot_path)
    print(f"Log saved  → {log_path}")


if __name__ == "__main__":
    main()
