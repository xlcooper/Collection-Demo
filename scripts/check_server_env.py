#!/usr/bin/env python3
"""Collect a safe, Git-friendly snapshot of the auto DL server environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "0.1.0"
MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)
MIN_CUDA = (12, 6)


def run_command(command: list[str], timeout: int = 15) -> dict[str, Any]:
    """Run a diagnostic command without raising when the command is unavailable."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"available": False, "return_code": None, "output": "not_found"}
    except subprocess.TimeoutExpired:
        return {"available": True, "return_code": None, "output": "timeout"}
    except OSError as exc:
        return {
            "available": False,
            "return_code": None,
            "output": f"os_error:{type(exc).__name__}",
        }

    output = result.stdout.strip() or result.stderr.strip()
    return {
        "available": True,
        "return_code": result.returncode,
        "output": compact_output(output),
    }


def compact_output(value: str, max_lines: int = 8, max_chars: int = 1200) -> str:
    """Keep reports readable and prevent commands from flooding the JSON file."""
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    compact = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        compact += "\n..."
    if len(compact) > max_chars:
        compact = compact[:max_chars] + "..."
    return compact


def parse_major_minor(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def bytes_to_gib(value: int) -> float:
    return round(value / (1024**3), 2)


def add_issue(
    issues: list[dict[str, str]], severity: str, code: str, message: str
) -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def inspect_package(distribution: str, import_name: str | None = None) -> dict[str, Any]:
    module_name = import_name or distribution.replace("-", "_")
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None

    try:
        importable = importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        importable = False

    return {
        "installed": bool(version or importable),
        "version": version or ("editable_or_unknown" if importable else None),
    }


def collect_git(repo_root: Path) -> dict[str, Any]:
    commit = run_command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = run_command(["git", "-C", str(repo_root), "status", "--porcelain"])
    git_version = run_command(["git", "--version"])
    git_lfs = run_command(["git", "lfs", "version"])

    dirty_entries = 0
    if status["return_code"] == 0 and status["output"] not in {"", "not_found"}:
        dirty_entries = len(status["output"].splitlines())

    return {
        "available": git_version["available"] and git_version["return_code"] == 0,
        "version": git_version["output"],
        "commit": commit["output"] if commit["return_code"] == 0 else None,
        "worktree_clean": status["return_code"] == 0 and dirty_entries == 0,
        "dirty_entry_count": dirty_entries,
        "git_lfs": {
            "available": git_lfs["available"] and git_lfs["return_code"] == 0,
            "version": git_lfs["output"],
        },
    }


def collect_system(repo_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(repo_root)
    memory_total_bytes: int | None = None

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_total_bytes = int(line.split()[1]) * 1024
                break

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "memory_total_gib": (
            bytes_to_gib(memory_total_bytes) if memory_total_bytes is not None else None
        ),
        "repo_disk": {
            "total_gib": bytes_to_gib(disk.total),
            "free_gib": bytes_to_gib(disk.free),
        },
    }


def collect_gpu(issues: list[dict[str, str]]) -> dict[str, Any]:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, Any]] = []

    if query["return_code"] == 0:
        for index, line in enumerate(query["output"].splitlines()):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 3:
                try:
                    memory_mib: int | None = int(fields[1])
                except ValueError:
                    memory_mib = None
                devices.append(
                    {
                        "index": index,
                        "name": fields[0],
                        "memory_total_mib": memory_mib,
                        "driver_version": fields[2],
                    }
                )
    else:
        add_issue(
            issues,
            "blocked",
            "nvidia_gpu_unavailable",
            "nvidia-smi 未发现可用 NVIDIA GPU；SAM3 官方实现需要 CUDA GPU。",
        )

    nvcc = run_command(["nvcc", "--version"])
    if not nvcc["available"] or nvcc["return_code"] != 0:
        add_issue(
            issues,
            "warning",
            "nvcc_not_found",
            "未发现 nvcc；预编译运行库可能仍可用，但编译可选 CUDA 扩展会受限。",
        )

    return {
        "nvidia_smi_available": query["return_code"] == 0,
        "device_count": len(devices),
        "devices": devices,
        "nvcc": nvcc,
    }


def collect_conda(issues: list[dict[str, str]]) -> dict[str, Any]:
    conda = run_command(["conda", "--version"])
    available = conda["available"] and conda["return_code"] == 0
    raw_current_environment = os.environ.get("CONDA_DEFAULT_ENV")
    current_environment = (
        Path(raw_current_environment).name if raw_current_environment else None
    )

    if not available:
        add_issue(
            issues,
            "blocked",
            "conda_not_found",
            "未发现 Conda；项目要求使用 Conda 管理运行环境。",
        )
    elif current_environment == "base":
        add_issue(
            issues,
            "warning",
            "conda_base_active",
            "当前处于 base 环境；项目运行应使用独立环境。",
        )

    return {
        "available": available,
        "version": conda["output"],
        "current_environment": current_environment,
    }


def collect_python(issues: list[dict[str, str]]) -> dict[str, Any]:
    version_tuple = (sys.version_info.major, sys.version_info.minor)
    if version_tuple < MIN_PYTHON:
        add_issue(
            issues,
            "blocked",
            "python_too_old",
            f"当前 Python {version_tuple[0]}.{version_tuple[1]}，SAM3 要求 Python >= 3.12。",
        )

    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable_name": Path(sys.executable).name,
        "meets_sam3_minimum": version_tuple >= MIN_PYTHON,
    }


def collect_pytorch(
    packages: dict[str, dict[str, Any]], issues: list[dict[str, str]]
) -> dict[str, Any]:
    torch_package = packages["torch"]
    if not torch_package["installed"]:
        add_issue(
            issues,
            "blocked",
            "torch_not_installed",
            "未安装 PyTorch；SAM3 要求 PyTorch >= 2.7。",
        )
        return {"importable": False}

    parsed_torch = parse_major_minor(torch_package["version"])
    if parsed_torch is not None and parsed_torch < MIN_TORCH:
        add_issue(
            issues,
            "blocked",
            "torch_too_old",
            f"当前 PyTorch {torch_package['version']}，SAM3 要求 PyTorch >= 2.7。",
        )

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        cuda_parsed = parse_major_minor(cuda_version)
        if not cuda_available:
            add_issue(
                issues,
                "blocked",
                "torch_cuda_unavailable",
                "PyTorch 无法使用 CUDA。",
            )
        if cuda_parsed is not None and cuda_parsed < MIN_CUDA:
            add_issue(
                issues,
                "blocked",
                "cuda_too_old",
                f"PyTorch 编译 CUDA 为 {cuda_version}，SAM3 要求 CUDA >= 12.6。",
            )

        device_count = torch.cuda.device_count() if cuda_available else 0
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]

        cudnn_version = None
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            cudnn_version = torch.backends.cudnn.version()

        nccl_version = None
        if cuda_available and hasattr(torch.cuda, "nccl"):
            try:
                nccl_version = torch.cuda.nccl.version()
            except (AttributeError, RuntimeError):
                nccl_version = None

        return {
            "importable": True,
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "compiled_cuda_version": cuda_version,
            "device_count": device_count,
            "device_names": device_names,
            "cudnn_version": cudnn_version,
            "nccl_version": nccl_version,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive broken imports
        add_issue(
            issues,
            "blocked",
            "torch_import_failed",
            f"PyTorch 导入失败：{type(exc).__name__}。",
        )
        return {"importable": False, "error_type": type(exc).__name__}


def collect_packages(issues: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    package_map = {
        "torch": ("torch", "torch"),
        "torchvision": ("torchvision", "torchvision"),
        "sam3": ("sam3", "sam3"),
        "transformers": ("transformers", "transformers"),
        "qwen_vl_utils": ("qwen-vl-utils", "qwen_vl_utils"),
        "huggingface_hub": ("huggingface-hub", "huggingface_hub"),
        "vllm": ("vllm", "vllm"),
        "sglang": ("sglang", "sglang"),
    }
    packages = {
        key: inspect_package(distribution, import_name)
        for key, (distribution, import_name) in package_map.items()
    }

    if not packages["sam3"]["installed"]:
        add_issue(
            issues,
            "blocked",
            "sam3_not_installed",
            "未安装 SAM3 项目包。",
        )
    if not packages["transformers"]["installed"]:
        add_issue(
            issues,
            "warning",
            "transformers_not_installed",
            "未安装 Transformers，Qwen3-VL 直接推理路径当前不可用。",
        )
    if not packages["qwen_vl_utils"]["installed"]:
        add_issue(
            issues,
            "warning",
            "qwen_vl_utils_not_installed",
            "未安装 qwen-vl-utils。",
        )

    return packages


def inspect_target_path(raw_path: str | None) -> dict[str, Any]:
    if not raw_path:
        return {"configured": False}

    path = Path(raw_path).expanduser()
    existing_path = path if path.exists() else path.parent
    info: dict[str, Any] = {
        "configured": True,
        "exists": path.exists(),
        "is_directory": path.is_dir() if path.exists() else None,
        "writable": os.access(existing_path, os.W_OK) if existing_path.exists() else False,
    }
    if existing_path.exists():
        disk = shutil.disk_usage(existing_path)
        info["disk_free_gib"] = bytes_to_gib(disk.free)
    return info


def summarize(issues: list[dict[str, str]]) -> dict[str, Any]:
    counts = {
        "blocked": sum(issue["severity"] == "blocked" for issue in issues),
        "warning": sum(issue["severity"] == "warning" for issue in issues),
    }
    status = "blocked" if counts["blocked"] else "warning" if counts["warning"] else "ready"
    return {"status": status, "issue_counts": counts, "issues": issues}


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a safe environment report for Collection-Demo."
    )
    parser.add_argument(
        "--output",
        default="environment/server_env_report.json",
        help="Report path, relative to the repository root by default.",
    )
    parser.add_argument("--model-dir", help="Optional server model directory to inspect.")
    parser.add_argument("--data-dir", help="Optional server data directory to inspect.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    issues: list[dict[str, str]] = []
    packages = collect_packages(issues)
    report = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "sam3": {
                "source": "facebookresearch/sam3",
                "minimum_python": "3.12",
                "minimum_pytorch": "2.7",
                "minimum_cuda": "12.6",
            },
            "mllm": {
                "primary": "Qwen/Qwen3-VL-8B-Instruct",
                "fallback": "Qwen/Qwen3-VL-4B-Instruct",
                "alternative": "OpenGVLab/InternVL3.5-8B-HF",
            },
        },
        "git": collect_git(repo_root),
        "system": collect_system(repo_root),
        "gpu": collect_gpu(issues),
        "conda": collect_conda(issues),
        "python": collect_python(issues),
        "packages": packages,
        "pytorch": collect_pytorch(packages, issues),
        "paths": {
            "model_directory": inspect_target_path(args.model_dir),
            "data_directory": inspect_target_path(args.data_dir),
        },
        "credentials": {
            "hf_token_environment_configured": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            ),
            "note": "只记录令牌是否通过环境变量配置，不读取或保存令牌值。缓存登录状态未检查。",
        },
    }

    if not report["git"]["available"]:
        add_issue(issues, "blocked", "git_not_found", "未发现 Git。")
    if report["system"]["os"] != "Linux":
        add_issue(
            issues,
            "warning",
            "non_linux_server",
            "当前系统不是 Linux；官方 SAM3 环境通常按 Linux/CUDA 路径部署。",
        )
    if not report["credentials"]["hf_token_environment_configured"]:
        add_issue(
            issues,
            "warning",
            "hf_token_env_not_set",
            "未检测到 Hugging Face 令牌环境变量；若未使用缓存登录，将无法下载受限 checkpoint。",
        )

    report["summary"] = summarize(issues)
    write_report(report, output_path)

    relative_output = output_path
    try:
        relative_output = output_path.relative_to(repo_root)
    except ValueError:
        pass
    print(f"Environment report: {relative_output}")
    print(f"Status: {report['summary']['status']}")
    print(f"Issues: {len(issues)}")

    return 2 if report["summary"]["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
