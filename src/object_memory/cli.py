"""Minimal command-line entry point for the memory store."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from pydantic import ValidationError

from .assets import MemoryPaths
from .config import DEFAULT_CONFIG_PATH, load_config, resolve_memory_root
from .memory_store import MemoryStore, MemoryStoreError, StoreStatus


def _add_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--memory-root",
        help="Override the configured memory root for this command.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="object-memory",
        description="Initialize and inspect the explicit object-memory store.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser(
        "init", help="Create or safely reopen the memory store."
    )
    _add_store_arguments(init_parser)

    status_parser = commands.add_parser(
        "status", help="Show schema version and record counts."
    )
    _add_store_arguments(status_parser)
    return parser


def _build_store(config_path: str, memory_root: str | None) -> MemoryStore:
    config = load_config(config_path)
    root = resolve_memory_root(config, memory_root)
    paths = MemoryPaths(root, config.storage.database_filename)
    return MemoryStore(paths)


def _print_status(status: StoreStatus, as_json: bool, initialized: bool) -> None:
    if as_json:
        payload = status.as_dict()
        payload["status"] = "ready"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    action = "initialized" if initialized else "ready"
    print(f"Memory store {action}: {status.database_path}")
    print(f"Schema version: {status.schema_version}")
    print("Records:")
    for table, count in status.counts.items():
        print(f"  {table}: {count}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        store = _build_store(args.config, args.memory_root)
        initialized = args.command == "init"
        status = store.initialize() if initialized else store.status()
        _print_status(status, args.json, initialized)
        return 0
    except (FileNotFoundError, ValueError, ValidationError, MemoryStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
