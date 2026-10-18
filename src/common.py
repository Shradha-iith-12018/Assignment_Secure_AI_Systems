"""Shared helpers for paths, validation, MNIST loading, model training, and secure artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MANIFEST = MODELS_DIR / "manifest.json"

_SAFE_NAME = re.compile(r"^[a-z0-9_]+$")
_DEVICE: torch.device | None = None


def _check_name(name: str) -> str:
    """Validate names that are later used as file names."""
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"Invalid artifact name: {name!r}")
    return name


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for repeatable experiments."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _cuda_probe() -> tuple[bool, Exception | None]:
    """Return whether this PyTorch build can actually execute a CUDA kernel."""
    if not torch.cuda.is_available():
        return False, None
    try:
        probe = torch.ones(1, device="cuda")
        probe = probe + 1
        torch.cuda.synchronize()
        del probe
        return True, None
    except Exception as exc:  # hardware/build mismatch is environment-dependent
        return False, exc


def get_device() -> torch.device:
    """Return the execution device selected through ``SSE_DEVICE``.

    ``SSE_DEVICE`` may be ``auto`` (default), ``cpu`` or ``cuda``. ``auto``
    uses CUDA only after a real CUDA operation succeeds. This is stronger
    than ``torch.cuda.is_available()`` and handles systems where a GPU is
    visible but its compute capability is unsupported by the installed
    PyTorch build.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    requested = os.environ.get("SSE_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"SSE_DEVICE must be one of 'auto', 'cpu', or 'cuda'; got {requested!r}"
        )

    if requested == "cpu":
        _DEVICE = torch.device("cpu")
        return _DEVICE

    usable, exc = _cuda_probe()
    if usable:
        _DEVICE = torch.device("cuda")
        return _DEVICE

    if requested == "cuda":
        if exc is None:
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation cannot execute on the detected GPU."
        ) from exc

    if exc is not None:
        warnings.warn(
            "CUDA is visible but unusable by this PyTorch build; falling back to CPU. "
            "Set SSE_DEVICE=cpu to suppress CUDA probing on this system.",
            RuntimeWarning,
            stacklevel=2,
        )

    _DEVICE = torch.device("cpu")
    return _DEVICE


def validate_images(x: torch.Tensor, *, name: str = "images") -> None:
    """Validate an MNIST image tensor at a trust boundary."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
        raise ValueError(f"{name} must have shape (N, 1, 28, 28); got {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise TypeError(f"{name} must contain floating-point pixel values")
    if not torch.isfinite(x).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")
    if x.numel() and (x.min().item() < 0.0 or x.max().item() > 1.0):
        raise ValueError(f"{name} pixel values must be in [0, 1]")


def validate_labels(y: torch.Tensor, *, n: int | None = None, name: str = "labels") -> None:
    """Validate MNIST labels."""
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if y.ndim != 1:
        raise ValueError(f"{name} must have shape (N,); got {tuple(y.shape)}")
    if n is not None and len(y) != n:
        raise ValueError(f"{name} length {len(y)} does not match image count {n}")
    if y.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long; got {y.dtype}")
    if y.numel() and (y.min().item() < 0 or y.max().item() > 9):
        raise ValueError(f"{name} values must be in [0, 9]")


def validate_dataset(x: torch.Tensor, y: torch.Tensor, *, name: str = "dataset") -> None:
    validate_images(x, name=f"{name} images")
    validate_labels(y, n=len(x), name=f"{name} labels")
    if len(x) == 0:
        raise ValueError(f"{name} must not be empty")


def load_mnist():
    """Return ``(x_train, y_train, x_test, y_test)`` in plain pixel space.

    Images are float32 tensors in [0, 1] with shape ``(N, 1, 28, 28)``.
    torchvision's MNIST loader verifies downloaded archive integrity.
    """

    def to_tensors(ds):
        x = ds.data.unsqueeze(1).float() / 255.0
        y = ds.targets.long()
        return x, y

    train = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)
    x_train, y_train = to_tensors(train)
    x_test, y_test = to_tensors(test)
    validate_dataset(x_train, y_train, name="MNIST training set")
    validate_dataset(x_test, y_test, name="MNIST test set")
    return x_train, y_train, x_test, y_test


class MnistCNN(nn.Module):
    """Two convolution layers and two fully connected layers for MNIST."""

    MEAN, STD = 0.1307, 0.3081

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        """Return penultimate-layer activations for activation clustering."""
        x = (x - self.MEAN) / self.STD
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(self.drop1(x), 1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.drop2(self.features(x)))


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    adversary: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    adv_ratio: float = 0.5,
    log_prefix: str = "",
):
    """Train a fresh ``MnistCNN`` and return ``(model, loss_history)``."""
    validate_dataset(x, y, name="training data")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if lr <= 0:
        raise ValueError("learning rate must be > 0")
    if not 0.0 <= adv_ratio <= 1.0:
        raise ValueError("adv_ratio must be in [0, 1]")

    device = get_device()
    model = MnistCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        total_loss = 0.0

        for i in range(0, len(x), batch_size):
            idx = perm[i : i + batch_size]
            xb = x[idx].to(device)
            yb = y[idx].to(device)

            if adversary is not None:
                k = int(round(len(xb) * adv_ratio))
                if k > 0:
                    adv_idx = torch.randperm(len(xb), device=device)[:k]
                    model.eval()
                    xb = xb.clone()
                    xb[adv_idx] = adversary(model, xb[adv_idx], yb[adv_idx])
                    model.train()

            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        history.append(total_loss / len(x))
        print(f"{log_prefix}epoch {epoch + 1}/{epochs}  train loss {history[-1]:.4f}")

    model.eval()
    return model, history


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest() -> dict[str, str]:
    """Read and validate the trusted artifact manifest."""
    if not MANIFEST.exists():
        return {}
    try:
        data = json.loads(MANIFEST.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model manifest is not valid JSON: {MANIFEST}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise RuntimeError(f"Model manifest has an invalid structure: {MANIFEST}")
    return data


def save_model(model: nn.Module, name: str) -> Path:
    """Save model weights in non-pickle ``safetensors`` format.

    Only tensor weights from ``state_dict`` are serialized. A SHA-256 digest
    is recorded separately in ``models/manifest.json`` so artifact integrity
    is verified before deserialization. The manifest is not a signature; in a
    production system it should be protected or signed independently.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _check_name(name)
    path = MODELS_DIR / f"{safe_name}.safetensors"

    state_dict = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in model.state_dict().items()
    }
    save_safetensors(state_dict, str(path))

    manifest = _read_manifest()
    # Remove the obsolete pickle-backed entry if this repository was upgraded
    # from the earlier .pt representation. The old file, if any, is never
    # loaded by this implementation.
    manifest.pop(f"{safe_name}.pt", None)
    manifest[path.name] = _sha256(path)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_model(name: str) -> MnistCNN:
    """Verify SHA-256 and load weights from non-executable ``safetensors``."""
    safe_name = _check_name(name)
    path = MODELS_DIR / f"{safe_name}.safetensors"
    if not path.exists():
        legacy = MODELS_DIR / f"{safe_name}.pt"
        if legacy.exists():
            raise RuntimeError(
                f"Legacy pickle-backed model artifact exists at {legacy}, but this version "
                "loads only .safetensors files. Retrain/regenerate the model first."
            )
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    if not MANIFEST.exists():
        raise RuntimeError(f"Trusted model manifest does not exist: {MANIFEST}")

    manifest = _read_manifest()
    expected = manifest.get(path.name)
    if expected is None:
        raise RuntimeError(f"No trusted hash for {path.name}; refusing to load it.")

    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Integrity check failed for {path}; refusing to load it.")

    # Load tensor data on CPU first. safetensors does not execute Python code
    # during deserialization; the model is moved to the selected device only
    # after the state_dict has been validated by PyTorch.
    state_dict = load_safetensors(str(path), device="cpu")
    model = MnistCNN()
    model.load_state_dict(state_dict, strict=True)
    return model.to(get_device()).eval()


def save_npz(name: str, **arrays) -> Path:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GEN_DIR / f"{_check_name(name)}.npz"
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def load_npz(name: str) -> dict[str, torch.Tensor]:
    path = GEN_DIR / f"{_check_name(name)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Generated dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: torch.from_numpy(data[k]) for k in data.files}


def save_json(name: str, obj) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{_check_name(name)}.json"
    path.write_text(json.dumps(obj, indent=2) + "\n")
    return path


def load_json(name: str):
    path = RESULTS_DIR / f"{_check_name(name)}.json"
    return json.loads(path.read_text()) if path.exists() else None
