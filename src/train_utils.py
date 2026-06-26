"""
Training utilities - loss plotting and prediction visualization.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_losses(log_path: str, output_path: str = "training_loss.png"):
    """Plot train vs validation loss curves from a JSON log file."""
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    val_loss = [e["val_loss"] for e in log]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, "o-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, "s-", label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[TrainUtils] Loss plot saved to {output_path}")

    # Diagnostic hints
    last_train = train_loss[-1]
    last_val = val_loss[-1]
    gap = last_val / (last_train + 1e-8) - 1.0

    if gap > 0.5:
        print("  Diagnostic: Large train/val gap - possible overfitting.")
        print("  Try: more data, stronger augmentation, or reduce epochs.")
    elif last_val > 0.01:
        print("  Diagnostic: High val loss - possible underfitting.")
        print("  Try: more epochs, lower LR, or larger model.")
    else:
        print("  Diagnostic: Loss looks healthy.")


def visualize_prediction(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    hm_gt: torch.Tensor,
    device: str = "cpu",
    output_path: str = "prediction_sample.png",
):
    """Save a side-by-side comparison: input | ground-truth heatmap | predicted heatmap."""
    model.eval()
    with torch.inference_mode():
        pred = model(img_tensor.unsqueeze(0).to(device))
    pred_hm = pred.squeeze().cpu().numpy()

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    gt = hm_gt.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Heatmap")
    axes[1].axis("off")

    axes[2].imshow(pred_hm, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Predicted Heatmap")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[TrainUtils] Prediction sample saved to {output_path}")
