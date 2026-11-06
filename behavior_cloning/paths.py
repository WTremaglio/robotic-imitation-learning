"""Locations of the datasets, checkpoints, and evaluations."""

import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")
CHECKPOINT_DIR = os.path.join(REPO_DIR, "checkpoints")
EVALUATION_DIR = os.path.join(REPO_DIR, "evaluations")


def dataset_file(name: str) -> str:
    """Return the merged dataset file of a dataset under data/."""
    return os.path.join(DATA_DIR, name, "dataset.npz")


def checkpoint_dir(name: str) -> str:
    """Return the folder of a checkpoint under checkpoints/."""
    return os.path.join(CHECKPOINT_DIR, name)


def network_file(name: str) -> str:
    """Return the network file of a checkpoint under checkpoints/."""
    return os.path.join(CHECKPOINT_DIR, name, "network.pt")


def training_file(name: str) -> str:
    """Return the training record of a network checkpoint."""
    return os.path.join(CHECKPOINT_DIR, name, "training.json")
