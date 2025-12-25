# run_train.py
"""
Convenience script to run training from project root.

Usage examples:
    # default run (edit defaults below or pass CLI args)
    python run_train.py

    # override defaults
    python run_train.py --data data/plant_village --epochs 15 --batch-size 32 --backbone resnet50

This script builds an argparse.Namespace and calls src.train.train_model(args).
"""

import argparse
import sys
from pathlib import Path

# ensure src is importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

def parse_args():
    parser = argparse.ArgumentParser("Run training (wrapper)")

    # Data + training hyperparams (defaults are safe for quick runs)
    parser.add_argument("--data", type=str, default="data/plant_village", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Model backbone (resnet18/resnet50 or timm name)")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")

    return parser.parse_args()


def main():
    args = parse_args()

    # Optional: print run summary
    print("Starting training with configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # set seed for reproducibility (uses src.utils.set_seed if available)
    try:
        from utils import set_seed  # src/utils.py (importable since we inserted src in sys.path)
        set_seed(args.seed)
    except Exception:
        try:
            # sometimes named differently; just attempt safe seeding
            import random, numpy as np, torch
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            print(f"[WARN] Fallback seed set to {args.seed}")
        except Exception:
            print("[WARN] Could not set seed (utils.set_seed unavailable)")

    # Build a simple namespace object compatible with src.train.train_model
    # train_model expects an argparse.Namespace with attributes used in src/train.py
    # We'll create a Namespace with those attributes.
    import argparse as _argparse
    args_ns = _argparse.Namespace()
    # copy expected attributes from parsed args (and set defaults that train.py expects)
    args_ns.data = args.data
    args_ns.epochs = args.epochs
    args_ns.batch_size = args.batch_size
    args_ns.img_size = args.img_size
    args_ns.lr = args.lr
    args_ns.val_split = args.val_split
    args_ns.backbone = args.backbone

    # Call the training entrypoint
    try:
        from train import train_model  # imports src/train.py (module name 'train' because src is in sys.path)
    except Exception as e:
        # As a fallback try importing as package
        try:
            from src.train import train_model
        except Exception:
            raise ImportError(f"Could not import train_model from src.train: {e}")

    train_model(args_ns)


if __name__ == "__main__":
    main()
