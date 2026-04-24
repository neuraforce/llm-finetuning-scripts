#!/usr/bin/env python3
"""
Evaluate a deployed Qwen3.6 model on the bewerbungen CV-extraction task.

For each sample the CV markdown is sent to the model with the same fixed
German system prompt that was used during training.  The model's JSON
response is parsed and compared field-by-field against the ground-truth JSON.

Results are written to a JSONL log file (one record per sample plus a summary
line) and a per-field accuracy table is printed to stdout.

Usage examples
--------------
# Evaluate against a bewerbungen directory (.md + .json pairs)
python scripts/evaluate_bewerbungen.py --bewerbungen-dir ~/bewerbungen

# Evaluate against a prepared eval JSONL (output of --split)
python scripts/evaluate_bewerbungen.py --eval-file data/bewerbungen_eval.jsonl

# Quick smoke test (10 samples only)
python scripts/evaluate_bewerbungen.py --bewerbungen-dir ~/bewerbungen --max-samples 10

# Custom endpoint / model
python scripts/evaluate_bewerbungen.py \\
    --endpoint http://192.168.35.8:8002 --model qwen3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import strip_thinking
from prepare_bewerbungen_dataset import FIELD_NAMES, SYSTEM_PROMPT, find_pairs

# ---------------------------------------------------------------------------
# Field comparison
# ---------------------------------------------------------------------------

#: Expected JSON fields, derived from the shared FIELD_DEFINITIONS source of truth.
EXPECTED_FIELDS: list[str] = FIELD_NAMES


def _strip_json_fence(text: str) -> str:
    """Remove optional ```json / ``` markdown code fences from *text*."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _compare_fields(predicted: dict, ground_truth: dict) -> dict:
    """
    Compare predicted JSON object against ground truth field-by-field.

    List fields (languages, education, skills, products) are compared
    order-insensitively.  String fields are compared case-insensitively
    after stripping whitespace.

    Returns a dict with:
      - field_<name>: bool  — whether each expected field matched
      - fields_matched: int
      - fields_total: int
      - field_accuracy: float
    """
    results: dict = {}
    matched = 0
    total = len(EXPECTED_FIELDS)

    for field in EXPECTED_FIELDS:
        pred_val = predicted.get(field)
        gt_val = ground_truth.get(field)

        if isinstance(pred_val, list) and isinstance(gt_val, list):
            match = (
                sorted(str(x).strip().lower() for x in pred_val)
                == sorted(str(x).strip().lower() for x in gt_val)
            )
        elif isinstance(pred_val, str) and isinstance(gt_val, str):
            match = pred_val.strip().lower() == gt_val.strip().lower()
        else:
            match = pred_val == gt_val

        results[f"field_{field}"] = match
        if match:
            matched += 1

    results["fields_matched"] = matched
    results["fields_total"] = total
    results["field_accuracy"] = round(matched / total, 4) if total > 0 else 0.0
    return results


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


async def _call_model(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    cv_text: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    retries: int,
    timeout: float,
) -> tuple[str, float]:
    """
    Send a CV extraction request and return (raw_response_text, latency_seconds).
    Raises RuntimeError after all retries are exhausted.
    """
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cv_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
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
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"Request failed after {retries} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Build task list
# ---------------------------------------------------------------------------


def _build_tasks_from_dir(bewerbungen_dir: Path, max_samples: int) -> list[dict]:
    """Build task list by scanning a bewerbungen directory for .md/.json pairs."""
    pairs = find_pairs(bewerbungen_dir)
    if not pairs:
        print(f"No complete .md/.json pairs found in {bewerbungen_dir}", file=sys.stderr)
        sys.exit(1)

    tasks: list[dict] = []
    for md_path, json_path in pairs:
        try:
            cv_text = md_path.read_text(encoding="utf-8").strip()
            ground_truth = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: skipping {md_path.parent.name!r} — {exc}", file=sys.stderr)
            continue
        tasks.append({
            "sample_name": md_path.parent.name,
            "cv_text": cv_text,
            "ground_truth": ground_truth,
        })
        if max_samples and len(tasks) >= max_samples:
            break
    return tasks


def _build_tasks_from_jsonl(eval_file: Path, max_samples: int) -> list[dict]:
    """
    Build task list from a prepared bewerbungen eval JSONL file.

    Expects ShareGPT format with system / user (CV markdown) / assistant (JSON)
    turns as produced by prepare_bewerbungen_dataset.py.
    """
    tasks: list[dict] = []
    with eval_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            convs = record.get("conversations", [])
            conv_map = {c["role"]: c["content"] for c in convs if "role" in c}
            cv_text = conv_map.get("user", "")
            gt_str = conv_map.get("assistant", "{}")
            try:
                ground_truth = json.loads(gt_str)
            except json.JSONDecodeError:
                ground_truth = {}
            tasks.append({
                "sample_name": f"sample_{len(tasks)}",
                "cv_text": cv_text,
                "ground_truth": ground_truth,
            })
            if max_samples and len(tasks) >= max_samples:
                break
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
    log_path: Path,
) -> list[dict]:
    """
    Evaluate all tasks asynchronously.  Results are written to log_path after
    each sample (append mode) so the log survives a crash mid-run.

    When concurrency > 1, results are written in completion order, not task
    order — the log is therefore nondeterministic in sequence across runs.

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
                            cv_text=task["cv_text"],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            seed=seed,
                            retries=retries,
                            timeout=timeout,
                        )
                        try:
                            response_text = strip_thinking(raw_response)
                            predicted = json.loads(_strip_json_fence(response_text))
                            parse_status = "ok"
                        except json.JSONDecodeError:
                            predicted = {}
                            parse_status = "parse_error"

                        field_metrics = _compare_fields(predicted, task["ground_truth"])
                        record = {
                            "sample": task["sample_name"],
                            "raw_response": raw_response,
                            **field_metrics,
                            "latency_s": latency,
                            "status": parse_status,
                            "timestamp": ts,
                        }
                    except Exception as exc:  # noqa: BLE001
                        record = {
                            "sample": task["sample_name"],
                            "raw_response": "",
                            "fields_matched": None,
                            "fields_total": len(EXPECTED_FIELDS),
                            "field_accuracy": None,
                            **{f"field_{f}": None for f in EXPECTED_FIELDS},
                            "latency_s": None,
                            "status": f"error: {exc}",
                            "timestamp": ts,
                        }

                    async with write_lock:
                        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log_file.flush()
                        results.append(record)
                    return record

            coros = [process(t) for t in tasks]
            await atqdm.gather(*coros, desc="Evaluating", unit="sample")
    finally:
        log_file.close()

    return list(results)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    parse_errors = [r for r in results if r["status"] == "parse_error"]
    other_errors = [r for r in results if r["status"] not in ("ok", "parse_error")]
    n = len(results)

    print(f"\n{'='*60}")
    print(
        f"  Evaluation complete: {n} samples, {len(ok)} ok"
        + (f", {len(parse_errors)} parse errors" if parse_errors else "")
        + (f", {len(other_errors)} request errors" if other_errors else "")
    )

    if ok:
        accuracies = [r["field_accuracy"] for r in ok if r["field_accuracy"] is not None]
        if accuracies:
            mean_acc = sum(accuracies) / len(accuracies)
            print(f"  Mean field accuracy : {mean_acc:.4f} ({mean_acc * 100:.1f}%)")

        print("\n  Per-field accuracy:")
        for field in EXPECTED_FIELDS:
            key = f"field_{field}"
            vals = [r[key] for r in ok if r.get(key) is not None]
            if vals:
                field_acc = sum(vals) / len(vals)
                bar = "█" * round(field_acc * 20)
                print(f"    {field:<16} {field_acc:.3f}  {bar}")

    if parse_errors:
        print(f"\n  First parse error — raw response: {parse_errors[0]['raw_response'][:120]!r}")
    if other_errors:
        print(f"\n  First request error: {other_errors[0]['status']}")
    print(f"{'='*60}\n")


def _write_summary(results: list[dict], log_path: Path) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    summary: dict = {
        "_summary": True,
        "n_total": len(results),
        "n_ok": len(ok),
        "n_error": len(results) - len(ok),
    }

    accuracies = [r["field_accuracy"] for r in ok if r.get("field_accuracy") is not None]
    summary["field_accuracy_mean"] = (
        round(sum(accuracies) / len(accuracies), 6) if accuracies else None
    )

    for field in EXPECTED_FIELDS:
        key = f"field_{field}"
        vals = [r[key] for r in ok if r.get(key) is not None]
        summary[f"{key}_accuracy"] = round(sum(vals) / len(vals), 6) if vals else None

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate vLLM-deployed model on the bewerbungen CV-extraction task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bewerbungen-dir",
        type=Path,
        help="Directory of per-folder CV samples (.md + .json pairs).",
    )
    source.add_argument(
        "--eval-file",
        type=Path,
        help="Prepared eval JSONL file (e.g. output of prepare_bewerbungen_dataset.py --split).",
    )
    p.add_argument("--endpoint", default="http://192.168.35.8:8002",
                   help="vLLM base URL (default: http://192.168.35.8:8002)")
    p.add_argument("--model", default="qwen3",
                   help="Model ID served by vLLM (default: qwen3)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Directory for log files (default: outputs/)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max parallel requests (default: 4)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Stop after N samples — 0 means all (default: 0)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy (default: 0.0)")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Max tokens in model response (default: 1024)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout in seconds (default: 120)")
    p.add_argument("--retries", type=int, default=3,
                   help="Retry attempts per request (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be greater than or equal to 1.")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be greater than or equal to 0.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"eval_bewerbungen_{stamp}.jsonl"

    if args.bewerbungen_dir:
        bew_dir = args.bewerbungen_dir.expanduser()
        tasks = _build_tasks_from_dir(bew_dir, max_samples=args.max_samples)
        print(f"Bewerbungen dir: {bew_dir}")
    else:
        eval_file = args.eval_file.expanduser()
        tasks = _build_tasks_from_jsonl(eval_file, max_samples=args.max_samples)
        print(f"Eval file      : {eval_file}")

    if not tasks:
        raise SystemExit("No evaluation tasks found; check the input data.")

    print(f"Endpoint       : {args.endpoint}")
    print(f"Model          : {args.model}")
    print(f"Concurrency    : {args.concurrency}")
    print(f"Log file       : {log_path}")
    print(f"Total samples  : {len(tasks)}")
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
            log_path=log_path,
        )
    )

    _write_summary(results, log_path)
    _print_summary(results)
    print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()
