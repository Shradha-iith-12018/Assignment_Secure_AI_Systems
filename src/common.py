"""Shared helpers: paths, MNIST loading, the CNN, the training loop and integrity-checked model save/load."""
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MANIFEST = MODELS_DIR / "manifest.json"

# Artifact names become file names, so only allow a safe character set (no path traversal).
_SAFE_NAME = re.compile(r"^[a-z0-9_]+$")


def _check_name(name: str) -> str:
    if not _SAFE_NAME.match(name):
        raise ValueError(f"Invalid artifact name: {name!r}")
    return name


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist():
    """Return (x_train, y_train, x_test, y_test).

    Images are float32 tensors in [0, 1] with shape (N, 1, 28, 28). torchvision verifies the
    MD5 of each downloaded archive before extracting it.
    """
    def to_tensors(ds):
        return ds.data.unsqueeze(1).float() / 255.0, ds.targets.long()

    train = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)
    return (*to_tensors(train), *to_tensors(test))


class MnistCNN(nn.Module):
    """Two conv layers + two fully connected layers. Normalisation happens inside the model,
    so attacks and defenses all work in plain [0, 1] pixel space."""

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
        """Penultimate-layer activations (used by the activation-clustering defense)."""
        x = (x - self.MEAN) / self.STD
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(self.drop1(x), 1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.drop2(self.features(x)))


def train_model(x, y, epochs=5, batch_size=128, lr=1e-3, adversary=None, adv_ratio=0.5, log_prefix=""):
    """Train a fresh MnistCNN and return (model, history).

    If `adversary(model, xb, yb)` is given, `adv_ratio` of every batch is replaced by adversarial
    examples crafted against the current model (adversarial training).
    """
    device = get_device()
    model = MnistCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        total_loss = 0.0
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x[idx].to(device), y[idx].to(device)
            if adversary is not None:
                k = int(len(xb) * adv_ratio)
                model.eval()
                xb = xb.clone()
                xb[:k] = adversary(model, xb[:k], yb[:k])
                model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        history.append(total_loss / len(x))
        print(f"{log_prefix}epoch {epoch + 1}/{epochs}  train loss {history[-1]:.4f}")
    model.eval()
    return model, history


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(model: nn.Module, name: str) -> Path:
    """Save weights only (no pickled code) and record their SHA-256 in the manifest."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{_check_name(name)}.pt"
    torch.save(model.state_dict(), path)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[path.name] = _sha256(path)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return path


def load_model(name: str) -> MnistCNN:
    """Load a model only if its hash matches the manifest, and with weights_only=True so a
    tampered file cannot run code through pickle."""
    path = MODELS_DIR / f"{_check_name(name)}.pt"
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    expected = manifest.get(path.name)
    if expected is None or _sha256(path) != expected:
        raise RuntimeError(f"Integrity check failed for {path}; refusing to load it.")
    model = MnistCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(get_device()).eval()


def save_npz(name: str, **arrays) -> Path:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GEN_DIR / f"{_check_name(name)}.npz"
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def load_npz(name: str) -> dict:
    with np.load(GEN_DIR / f"{_check_name(name)}.npz", allow_pickle=False) as data:
        return {k: torch.from_numpy(data[k]) for k in data.files}


def save_json(name: str, obj) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{_check_name(name)}.json"
    path.write_text(json.dumps(obj, indent=2))
    return path


def load_json(name: str):
    path = RESULTS_DIR / f"{_check_name(name)}.json"
    return json.loads(path.read_text()) if path.exists() else None
