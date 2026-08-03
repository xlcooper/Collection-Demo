#!/usr/bin/env python3
"""Serve the object-memory experiment UI on one AutoDL instance."""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from object_memory.web_service import create_app, default_web_settings  # noqa: E402


PASSWORD_ENV = "OBJECT_MEMORY_WEB_PASSWORD"
USERNAME_ENV = "OBJECT_MEMORY_WEB_USERNAME"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the single-user object-memory experiment interface."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Listen address. Non-loopback addresses require a Basic password.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        choices=(6006, 6008),
        help="AutoDL custom-service port.",
    )
    return parser


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validated_credentials(
    parser: argparse.ArgumentParser,
    host: str,
) -> tuple[str, str | None]:
    username = os.environ.get(USERNAME_ENV, "object-memory").strip()
    password = os.environ.get(PASSWORD_ENV)
    if not username or ":" in username or any(ord(char) < 32 for char in username):
        parser.error(f"{USERNAME_ENV} must be a safe non-empty Basic username")
    if password is not None and len(password) < 12:
        parser.error(f"{PASSWORD_ENV} must contain at least 12 characters")
    if not is_loopback_host(host) and password is None:
        parser.error(
            f"non-loopback listening requires {PASSWORD_ENV}; "
            "use 127.0.0.1 with an SSH tunnel otherwise"
        )
    return username, password


def configure_model_cache() -> None:
    """Keep every Web-launched Qwen process on the project data disk."""

    hf_home = PROJECT_ROOT / "weights" / "qwen"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    username, password = _validated_credentials(parser, args.host)
    configure_model_cache()

    try:
        import uvicorn
    except ImportError as exc:
        parser.error(f"Web dependencies are not installed: {exc}")

    settings = default_web_settings(
        basic_username=username,
        basic_password=password,
    )
    app = create_app(settings)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        reload=False,
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
