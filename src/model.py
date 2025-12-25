# src/model.py
"""
Model builder for AI-Driven Crop Disease Prediction & Management System.

Supports:
- torchvision ResNet18, ResNet50
- timm EfficientNet / ConvNeXt / ViT models (optional)
"""

import torch
import torch.nn as nn
from typing import Optional


# ------------------------------------------------------------
# Build Model
# ------------------------------------------------------------
def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    pretrained: bool = True,
) -> nn.Module:
    """
    Builds a CNN model for plant disease classification.
    Supports both torchvision and timm models.

    Args:
        num_classes: number of output classes
        backbone: "resnet18" | "resnet50" | "<any timm model>"
        pretrained: load ImageNet pretrained weights

    Returns:
        PyTorch model (nn.Module)
    """
    backbone = backbone.lower()

    # Try TIMM first (if installed)
    try:
        import timm
        timm_available = True
    except Exception:
        timm_available = False

    # --------------------------------------------------------
    # If user specifies a timm-compatible backbone
    # --------------------------------------------------------
    if timm_available and backbone not in ["resnet18", "resnet50"]:
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes
        )
        return model

    # --------------------------------------------------------
    # Torchvision ResNet models
    # --------------------------------------------------------
    from torchvision import models

    if backbone == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model

    elif backbone == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model

    # --------------------------------------------------------
    # If user gives unknown backbone
    # --------------------------------------------------------
    if timm_available:
        try:
            model = timm.create_model(
                backbone, pretrained=pretrained, num_classes=num_classes
            )
            return model
        except Exception:
            pass

    raise ValueError(f"Unsupported backbone: {backbone}")


# ------------------------------------------------------------
# Utility: get device
# ------------------------------------------------------------
def get_device() -> torch.device:
    """
    Returns GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Utility: summary
# ------------------------------------------------------------
def print_model_summary(model: nn.Module, input_size=(1, 3, 224, 224)):
    """
    Prints a readable model summary.
    """
    from torchsummary import summary  # pip install torchsummary (optional)
    try:
        summary(model, input_size[1:])
    except Exception:
        print(model)
        print("Install torchsummary for a better summary.")


# ------------------------------------------------------------
# Quick test
# Run: python src/model.py
# ------------------------------------------------------------
if __name__ == "__main__":
    num_classes = 10
    model = build_model(num_classes=num_classes, backbone="resnet18", pretrained=False)
    print(model)
    print("Model OK. Total parameters:", sum(p.numel() for p in model.parameters()))
