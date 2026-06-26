"""
Training script for MiniUNet heatmap regression.

Usage:
    python src/train.py --data-dir data/raw --epochs 50 --batch-size 8

Output:
    checkpoints/best_model.pt     - model with lowest validation loss
    checkpoints/final_model.pt    - model from last epoch
    training_log.json             - per-epoch loss records
    training_loss.png             - loss curve plot
"""
import json
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import SixshotDataset
from src.model import MiniUNet
from src.train_utils import plot_losses, visualize_prediction


def parse_args():
    p = argparse.ArgumentParser(description="Train MiniUNet for Sixshot target detection")
    p.add_argument("--data-dir", type=str, default="data/raw",
                   help="Directory with frame_*.png and frame_*.json files")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--input-size", type=int, default=256,
                   help="Resize images to (input_size x input_size)")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Gaussian sigma for heatmap targets")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data for validation")
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", type=str, default=default_dev,
                   help="Training device: 'cpu' or 'cuda'")
    p.add_argument("--output-dir", type=str, default="checkpoints",
                   help="Directory for saving model checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    # -- Setup -----------------------------------------------------------
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Sixshot Training")
    print(f"  Data:    {args.data_dir}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}  |  LR: {args.lr}")
    print(f"  Size:    {args.input_size}x{args.input_size}  |  Sigma: {args.sigma}")
    print(f"  Device:  {device}")
    print("=" * 55)

    # -- Data augmentation ------------------------------------------------
    train_tf = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])

    # -- Dataset ----------------------------------------------------------
    dataset = SixshotDataset(
        data_dir=args.data_dir,
        sigma=args.sigma,
        downsample=1,
        input_size=args.input_size,
    )

    if len(dataset) == 0:
        print("[Train] ERROR: No labeled samples found.")
        print("        Run the labeler tools first, then re-run training.")
        sys.exit(1)

    # Split train / val
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"  Train samples: {train_size}  |  Val samples: {val_size}")

    # -- Model, loss, optimizer -------------------------------------------
    model = MiniUNet().to(device)
    print(f"  Parameters:   {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -- Training loop ----------------------------------------------------
    log = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs} [Train]",
                    unit="batch", leave=False)
        for imgs, hms in pbar:
            imgs, hms = imgs.to(device), hms.to(device)

            # Apply augmentation on GPU (ColorJitter works on tensors)
            imgs = train_tf(imgs)

            pred = model(imgs)
            loss = criterion(pred, hms)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0

        with torch.inference_mode():
            for imgs, hms in val_loader:
                imgs, hms = imgs.to(device), hms.to(device)
                pred = model(imgs)
                loss = criterion(pred, hms)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # ---- Logging ----
        entry = {"epoch": epoch, "train_loss": round(train_loss, 6),
                 "val_loss": round(val_loss, 6)}
        log.append(entry)

        improved = " *" if val_loss < best_val_loss else ""
        print(f"  Epoch {epoch:3d} | Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f}{improved}")

        # ---- Checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_dir / "best_model.pt")

    # -- Save final artifacts ---------------------------------------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "val_loss": val_loss,
    }, output_dir / "final_model.pt")

    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    # -- Generate plots ---------------------------------------------------
    plot_losses(str(log_path), str(output_dir / "training_loss.png"))

    # Visualize one validation sample
    val_img, val_hm = val_ds[0]
    visualize_prediction(
        model, val_img, val_hm, device=str(device),
        output_path=str(output_dir / "prediction_sample.png"),
    )

    print(f"\n  Best val loss: {best_val_loss:.6f}")
    print(f"  Outputs saved to: {output_dir.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
