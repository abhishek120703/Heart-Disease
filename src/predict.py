# src/predict.py
"""
Inference utilities for AI-Driven Crop Disease Prediction & Management System.

Provides:
- load_model()
- predict_image()
- predict_batch()
"""

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model, get_device


# ------------------------------------------------------------
# Default transforms for inference
# ------------------------------------------------------------
def get_inference_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ------------------------------------------------------------
# Load model from checkpoint
# ------------------------------------------------------------
def load_model(
    checkpoint_path: str,
    classes: List[str],
    backbone: str = "resnet18",
    img_size: int = 224,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    print(f"Loading model on {device}...")

    model = build_model(num_classes=len(classes), backbone=backbone, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


# ------------------------------------------------------------
# Predict on a single image
# ------------------------------------------------------------
def predict_image(
    img_path: str,
    model,
    classes: List[str],
    device,
    img_size: int = 224,
) -> Tuple[str, float]:
    """
    Returns:
        (predicted_class_name, confidence)
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    transform = get_inference_transform(img_size)

    # Load image
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dimension
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    predicted_class = classes[pred_idx.item()]
    confidence = conf.item()

    return predicted_class, confidence


# ------------------------------------------------------------
# Predict batch of images
# ------------------------------------------------------------
def predict_batch(
    image_paths: List[str],
    model,
    classes: List[str],
    device,
    img_size: int = 224,
) -> List[Tuple[str, float]]:
    results = []
    for img_path in image_paths:
        pred, conf = predict_image(
            img_path=img_path,
            model=model,
            classes=classes,
            device=device,
            img_size=img_size
        )
        results.append((pred, conf))
    return results


# ------------------------------------------------------------
# CLI Testing
# Run: python src/predict.py --image "path/to/image.jpg"
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Image Prediction Script")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--classes", type=str, default="data/plant_village")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--backbone", type=str, default="resnet18")
    args = parser.parse_args()

    # Load classes
    class_dirs = sorted([p.name for p in Path(args.classes).iterdir() if p.is_dir()])

    # Load model
    model, device = load_model(
        checkpoint_path=args.checkpoint,
        classes=class_dirs,
        backbone=args.backbone,
    )

    # Predict
    pred_class, conf = predict_image(
        img_path=args.image,
        model=model,
        classes=class_dirs,
        device=device
    )

    print("\nPrediction Result:")
    print(f" Image: {args.image}")
    print(f" Class: {pred_class}")
    print(f" Confidence: {conf:.4f}")
