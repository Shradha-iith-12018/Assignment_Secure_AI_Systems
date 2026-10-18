"""Record the software and execution environment used for experiments."""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def package_version(distribution: str) -> str | None:
    """Return an installed package version without spawning a subprocess."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def collect_device_info() -> dict:
    """
    Record the requested and actual execution device.

    When CPU is explicitly selected, CUDA probing is intentionally skipped.
    This avoids initializing an incompatible CUDA/cuDNN stack on systems where
    a GPU is visible but unsupported by the installed PyTorch build.
    """
    requested = os.environ.get("SSE_DEVICE", "auto").strip().lower()

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "SSE_DEVICE must be one of: auto, cpu, cuda "
            f"(got {requested!r})"
        )

    info = {
        "requested_device": requested,
        "execution_device": None,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_is_available": None,
        "gpu_name": None,
        "gpu_compute_capability": None,
        "cuda_execution_usable": None,
        "cuda_probe_skipped": False,
        "cuda_probe_error": None,
    }

    if requested == "cpu":
        info["execution_device"] = "cpu"
        info["cuda_probe_skipped"] = True
        return info

    try:
        cuda_available = torch.cuda.is_available()
        info["torch_cuda_is_available"] = cuda_available

        if not cuda_available:
            if requested == "cuda":
                raise RuntimeError(
                    "CUDA was explicitly requested, but "
                    "torch.cuda.is_available() returned False."
                )

            info["execution_device"] = "cpu"
            info["cuda_execution_usable"] = False
            return info

        # Perform a real CUDA operation.  This catches cases where CUDA is
        # detected but the installed PyTorch build does not support the GPU's
        # compute capability.
        probe = torch.ones(1, device="cuda")
        probe = probe + 1
        torch.cuda.synchronize()
        del probe

        info["execution_device"] = "cuda"
        info["cuda_execution_usable"] = True
        info["gpu_name"] = torch.cuda.get_device_name(0)

        major, minor = torch.cuda.get_device_capability(0)
        info["gpu_compute_capability"] = f"{major}.{minor}"

    except Exception as exc:
        info["cuda_execution_usable"] = False
        info["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"

        if requested == "cuda":
            raise RuntimeError(
                "CUDA was explicitly requested but could not execute "
                "successfully."
            ) from exc

        info["execution_device"] = "cpu"

    return info


def main() -> None:
    packages = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "numpy": package_version("numpy"),
        "scikit-learn": package_version("scikit-learn"),
        "matplotlib": package_version("matplotlib"),
        "adversarial-robustness-toolbox": package_version(
            "adversarial-robustness-toolbox"
        ),
        "safetensors": package_version("safetensors"),
        "bandit": package_version("bandit"),
        "semgrep": package_version("semgrep"),
        "pip-audit": package_version("pip-audit"),
    }

    environment = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "packages": packages,
        "tools": {
            # These tools are Python distributions in this project
            # environment, so their package versions are authoritative.
            "bandit": packages["bandit"],
            "semgrep": packages["semgrep"],
            "pip-audit": packages["pip-audit"],
        },
        "device": collect_device_info(),
    }

    output = RESULTS_DIR / "environment.json"
    output.write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Environment written to {output}")


if __name__ == "__main__":
    main()
