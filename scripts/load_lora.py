#!/usr/bin/env python3
"""
Dynamically load or unload a LoRA adapter into a running vLLM instance.

The vLLM server must have been started with ``--enable-lora``
(done automatically by ``serve_vllm.sh`` when ``LORA_ADAPTER_PATH`` is set).

Usage examples
--------------
# Load a LoRA adapter
python scripts/load_lora.py load --name cv-extraction --path /path/to/adapter

# Load to a specific endpoint
python scripts/load_lora.py load \\
    --endpoint http://192.168.35.8:8002 \\
    --name cv-extraction \\
    --path /path/to/adapter

# Unload a LoRA adapter
python scripts/load_lora.py unload --name cv-extraction

# List currently loaded LoRA adapters (via /v1/models)
python scripts/load_lora.py list
"""

from __future__ import annotations

import argparse
import sys

import httpx

_DEFAULT_ENDPOINT = "http://192.168.35.8:8002"
_DEFAULT_NAME = "cv-extraction"


def _load(client: httpx.Client, endpoint: str, name: str, path: str) -> None:
    url = f"{endpoint.rstrip('/')}/v1/load_lora_adapter"
    resp = client.post(url, json={"lora_name": name, "lora_path": path})
    if resp.status_code == 200:
        print(f"Loaded LoRA adapter '{name}' from '{path}'")
    else:
        print(f"ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)


def _unload(client: httpx.Client, endpoint: str, name: str) -> None:
    url = f"{endpoint.rstrip('/')}/v1/unload_lora_adapter"
    resp = client.post(url, json={"lora_name": name})
    if resp.status_code == 200:
        print(f"Unloaded LoRA adapter '{name}'")
    else:
        print(f"ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)


def _list_models(client: httpx.Client, endpoint: str) -> None:
    resp = client.get(f"{endpoint.rstrip('/')}/v1/models")
    resp.raise_for_status()
    models = resp.json().get("data", [])
    if not models:
        print("No models found.")
        return
    print("Available models / LoRA adapters:")
    for m in models:
        mid = m.get("id", "?")
        parent = m.get("parent")
        print(f"  {mid}  (LoRA on {parent})" if parent else f"  {mid}  (base)")


def _add_endpoint_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--endpoint", default=_DEFAULT_ENDPOINT,
                   help=f"vLLM base URL (default: {_DEFAULT_ENDPOINT})")


def _add_name_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", default=_DEFAULT_NAME,
                   help=f"LoRA adapter name as served by vLLM (default: {_DEFAULT_NAME})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage LoRA adapters on a running vLLM instance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # --- load ---
    p_load = sub.add_parser("load", help="Load a LoRA adapter into vLLM")
    _add_endpoint_arg(p_load)
    _add_name_arg(p_load)
    p_load.add_argument("--path", required=True,
                        help="Path to the LoRA adapter directory (adapter_config.json must be present)")

    # --- unload ---
    p_unload = sub.add_parser("unload", help="Unload a LoRA adapter from vLLM")
    _add_endpoint_arg(p_unload)
    _add_name_arg(p_unload)

    # --- list ---
    p_list = sub.add_parser("list", help="List all currently available models / adapters")
    _add_endpoint_arg(p_list)

    args = parser.parse_args()

    with httpx.Client(timeout=30.0) as client:
        if args.action == "load":
            _load(client, args.endpoint, args.name, args.path)
        elif args.action == "unload":
            _unload(client, args.endpoint, args.name)
        elif args.action == "list":
            _list_models(client, args.endpoint)


if __name__ == "__main__":
    main()

