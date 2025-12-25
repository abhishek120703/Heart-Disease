# src/train.py
"""
Training script for AI-Driven Crop Disease Prediction & Management System.

Usage:
    python -m src.train --data data/plant_village --epochs 10 --batch-size 16
"""

import os
import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from src.data import make_dataloaders
from src.model import build_model, get_device


# ------------------------------------------------------------
# Train one epoch
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ------------------------------------------------------------
# Validate one epoch
# ------------------------------------------------------------
def validate(model, loader, criterion, device):
    if loader is None:
        return 0, 0

    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ------------------------------------------------------------
# Main training function
# ------------------------------------------------------------
def train_model(args):
    device = get_device()
    print(f"Using device: {device}")

    # Create checkpoints folder
    Path("checkpoints").mkdir(exist_ok=True)

    # Load data
    train_loader, val_loader, _, classes = make_dataloaders(
        root=args.data,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        test_split=0.0,
        num_workers=2
    )

    print(f"Found {len(classes)} classes.")
    print("Classes:", classes)

    # Build model
    model = build_model(
        num_classes=len(classes),
        backbone=args.backbone,
        pretrained=True
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TensorBoard logging
    writer = SummaryWriter(log_dir="runs/crop_disease")

    best_acc = 0

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("🔥 Best model updated!")

    print("\nTraining completed.")
    print(f"Best Validation Accuracy: {best_acc:.4f}")


# ------------------------------------------------------------
# Command-line Arguments
# ------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset, e.g. data/plant_village")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="resnet18, resnet50, or any timm model")

    return parser.parse_args()


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    train_model(args)
