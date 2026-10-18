"""SAST control sample. NOT part of the application and never imported or run.

These are the naive versions of things the real code in src/, attacks/ and defenses/ also has to do
(load a model, fetch the dataset, pick samples to poison, check input). Scanning this file shows that
Bandit and Semgrep detect each pattern; the report maps every finding to the fix used in the real code.
"""
import pickle
import random
import subprocess
import urllib.request

import torch


def load_model_naive(path):
    # Fix in src/common.py: load_model() -> torch.load(weights_only=True) + SHA-256 manifest check
    return torch.load(path)


def load_model_pickle(path):
    # Fix: weights saved as a state_dict, never as a pickled object
    with open(path, "rb") as f:
        return pickle.load(f)


def download_mnist(url):
    # Fix: torchvision.datasets.MNIST(download=True) with built-in MD5 verification
    archive = "/tmp/mnist.gz"
    urllib.request.urlretrieve(url, archive)
    subprocess.call("gunzip -f " + archive, shell=True)
    return archive


def pick_poison_indices(n, k):
    # Fix: seeded torch.Generator; non-cryptographic randomness is acceptable here
    return random.sample(range(n), k)


def check_input(x):
    # Fix: explicit exception (_check_name in src/common.py raises ValueError)
    assert x.shape[1:] == (1, 28, 28)
    return x
