# src/data.py
"""
Data utilities for AI-Driven Crop Disease Prediction & Management System.

Folder structure expected:
data/plant_village/<class_name>/*.jpg

Provides:
- CropDiseaseDataset: PyTorch Dataset
- get_transforms(): torchvision transforms
- make_dataloaders(): train/val/test dataloaders
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random, math

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


# -------------------------------------------------------
# Build class mapping
# -------------------------------------------------------
def build_class_mapping(root: str) -> Dict[str, int]:
    """
    Maps class_name → index using sorted directory names.
    """
    root_p = Path(root)
    classes = sorted([p.name for p in root_p.iterdir() if p.is_dir()])
    return {c: i for i, c in enumerate(classes)}


# -------------------------------------------------------
# Dataset Class
# -------------------------------------------------------
class CropDiseaseDataset(Dataset):
    """
    Dataset expecting root/class_name/*.jpg images.
    """

    def __init__(
        self,
        root: str,
        img_size: int = 224,
        split: str = "train",
        classes: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)

        # Collect class names
        if classes is None:
            self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        else:
            self.classes = classes

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect all images
        self.samples: List[Tuple[str, int]] = []
        for cls in self.classes:
            cls_folder = self.root / cls
            if not cls_folder.exists():
                continue
            for img in cls_folder.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img), self.class_to_idx[cls]))

        # Assign transforms
        self.transform = transform or get_transforms(img_size, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# -------------------------------------------------------
# Transforms
# -------------------------------------------------------
def get_transforms(img_size: int = 224, split: str = "train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


# -------------------------------------------------------
# Build train/val/test dataloaders
# -------------------------------------------------------
def make_dataloaders(
    root: str,
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 2,
    seed: int = 42,
):
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    classes = sorted([p.name for p in root_p.iterdir() if p.is_dir()])

    # Full dataset for splitting
    full_ds = CropDiseaseDataset(root, img_size=img_size, split="train", classes=classes)
    total = len(full_ds)

    indices = list(range(total))
    if shuffle:
        random.Random(seed).shuffle(indices)

    val_count = int(total * val_split)
    test_count = int(total * test_split)
    train_count = total - val_count - test_count

    train_idx = indices[:train_count]
    val_idx = indices[train_count:train_count + val_count]
    test_idx = indices[train_count + val_count:]

    # Helper: create subset dataset with proper transforms
    def build_subset(idxs, split_name):
        if not idxs:
            return None
        ds = CropDiseaseDataset(root, img_size=img_size, split=split_name, classes=classes)
        ds.samples = [full_ds.samples[i] for i in idxs]  # keep only required files
        return ds

    train_ds = build_subset(train_idx, "train")
    val_ds = build_subset(val_idx, "val")
    test_ds = build_subset(test_idx, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers) if val_ds else None

    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers) if test_ds else None

    return train_loader, val_loader, test_loader, classes


if __name__ == "__main__":
    print("Testing dataset loading...")

    train_loader, val_loader, _, classes = make_dataloaders(
        root="data/plant_village",
        batch_size=4,
        img_size=224
    )

    print("Classes:", classes)
    print("Train batches:", len(train_loader))

    imgs, labels = next(iter(train_loader))
    print("Batch shape:", imgs.shape, labels.shape)
