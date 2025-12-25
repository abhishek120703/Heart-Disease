# src/utils.py
"""
Utility helper functions for AI-Driven Crop Disease Prediction & Management System.
Includes:
- Set random seed
- Accuracy calculation
- Confusion matrix
- Model save/load helpers
- Pretty printing utilities
"""

import os
import random
from typing import List, Dict

import numpy as np
import torch


# ------------------------------------------------------------
# Fix random seeds for reproducible training
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    Fix seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Seed set to: {seed}")


# ------------------------------------------------------------
# Accuracy calculation
# ------------------------------------------------------------
def accuracy(outputs, labels) -> float:
    """
    Computes accuracy for a batch.
    """
    preds = outputs.argmax(1)
    return (preds == labels).float().mean().item()


# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------
def confusion_matrix(
    preds: List[int],
    labels: List[int],
    num_classes: int,
) -> np.ndarray:
    """
    Returns a confusion matrix of shape (num_classes, num_classes)
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    for p, t in zip(preds, labels):
        matrix[t][p] += 1

    return matrix


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------
def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to: {path}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
def load_model(model, path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"[INFO] Model loaded from: {path}")
    return model


# ------------------------------------------------------------
# Print class distribution summary
# ------------------------------------------------------------
def print_class_distribution(dataset, classes: List[str]):
    """
    Prints total number of images in each class.
    """
    from collections import Counter

    counts = Counter()
    for _, label in dataset.samples:
        counts[classes[label]] += 1

    print("\n[INFO] Class Distribution:")
    for cls in classes:
        print(f"  {cls:40s} : {counts[cls]} images")
    print()


# ------------------------------------------------------------
# GPU memory check
# ------------------------------------------------------------
def gpu_memory():
    """
    Prints total and reserved GPU memory (if cuda available).
    """
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
        allocated = torch.cuda.memory_allocated(0) / (1024**2)

        print(f"\n[GPU MEMORY]")
        print(f"  Total     : {total:.2f} MB")
        print(f"  Reserved  : {reserved:.2f} MB")
        print(f"  Allocated : {allocated:.2f} MB")
    else:
        print("[INFO] CUDA not available. Using CPU.")


# ------------------------------------------------------------
# Timer utility
# ------------------------------------------------------------
class Timer:
    """
    Use:
        t = Timer()
        t.start()
        ... your code ...
        t.stop("training")
    """

    def __init__(self):
        import time
        self.time = time

    def start(self):
        self.t0 = self.time.time()

    def stop(self, name="process"):
        t = self.time.time() - self.t0
        print(f"[TIMER] {name} took {t:.2f} seconds.")
