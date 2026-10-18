"""Deliberately insecure SAST control sample. Never import or execute this file."""
import pickle
import random
import subprocess
import urllib.request

import torch


def load_model_naive(path):
    # Real fix: weights_only=True plus SHA-256 verification in src/common.py::load_model.
    return torch.load(path)


def load_model_pickle(path):
    # Real fix: application artifacts are state_dict tensors; generated NPZ data uses allow_pickle=False.
    with open(path, "rb") as f:
        return pickle.load(f)


def download_mnist(url):
    # Real fix: torchvision.datasets.MNIST(download=True) uses fixed dataset mirrors and integrity checks.
    archive = "/tmp/mnist.gz"
    urllib.request.urlretrieve(url, archive)
    subprocess.call("gunzip -f " + archive, shell=True)
    return archive


def pick_poison_indices(n, k):
    # Intentional control finding. Poison selection is reproducibility-sensitive, not secret-sensitive.
    return random.sample(range(n), k)


def check_input(x):
    # Real fix: validate_images/validate_dataset raise explicit exceptions in src/common.py.
    assert x.shape[1:] == (1, 28, 28)
    return x
