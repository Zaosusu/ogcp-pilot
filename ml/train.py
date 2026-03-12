"""
OpenGuitarChordProject - 训练脚本
使用方法:
    python ml/train.py --data_dir dataset/raw --epochs 50
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders, CHORD_LABELS
from model import get_model


# ── 工具函数 ─────────────────────────────────────────────────────

def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        optimizer.zero_grad()
        out  = model(mel)
        loss = criterion(out, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = mel.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(out, label) * bs
        n += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    all_preds, all_labels = [], []

    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        out  = model(mel)
        loss = criterion(out, label)

        bs = mel.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(out, label) * bs
        n += bs

        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(label.cpu().tolist())

    return total_loss / n, total_acc / n, all_preds, all_labels


def print_confusion_summary(preds, labels, chord_labels):
    from collections import defaultdict
    correct = defaultdict(int)
    total   = defaultdict(int)
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    print("\n── 每类准确率 ──")
    for i, name in enumerate(chord_labels):
        if total[i] > 0:
            print(f"  {name:>4}: {correct[i]}/{total[i]}  "
                  f"({100*correct[i]/total[i]:.0f}%)")


def save_training_plot(log_rows, save_dir):
    """训练结束后生成并保存曲线图"""
    epochs     = [r["epoch"]      for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    val_loss   = [r["val_loss"]   for r in log_rows]
    train_acc  = [r["train_acc"]  * 100 for r in log_rows]
    val_acc    = [r["val_acc"]    * 100 for r in log_rows]

    best = max(log_rows, key=lambda r: r["val_acc"])
    best_e, best_acc = int(best["epoch"]), best["val_acc"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("OGCP Chord Recognition — Training Curves",
                 fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="#4C72B0", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="#DD8452", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Accuracy
    ax2.plot(epochs, train_acc, label="Train Acc", color="#4C72B0", linewidth=2)
    ax2.plot(epochs, val_acc,   label="Val Acc",   color="#DD8452", linewidth=2)
    ax2.axvline(best_e, color="gray", linestyle="--", alpha=0.6,
                label=f"Best @ ep{best_e}")
    ax2.annotate(f"{best_acc:.1f}%",
                 xy=(best_e, best_acc),
                 xytext=(best_e + 1, best_acc - 6),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=10, color="gray")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {plot_path}")


# ── 主训练循环 ────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_model(args.model, num_classes=14).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    log_rows = []

    print(f"\n开始训练 {args.epochs} 个 epoch ...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | "
              f"{val_loss:>8.4f} | {val_acc*100:>6.2f}% | "
              f"{lr_now:.2e} | {elapsed:>5.1f}s")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, save_dir / "best_model.pth")
            print(f"          ✅ 新最优: {val_acc*100:.2f}%  → 已保存")

    # ── 测试集评估 ──────────────────────────────────────────────
    print("\n── 加载最优模型进行测试集评估 ──")
    ckpt = torch.load(save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\n测试集 Loss: {test_loss:.4f}")
    print(f"测试集 Acc:  {test_acc*100:.2f}%")
    print_confusion_summary(test_preds, test_labels, CHORD_LABELS)

    # ── 保存 CSV ────────────────────────────────────────────────
    log_path = save_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"训练日志已保存: {log_path}")

    # ── 生成并保存训练曲线图 ─────────────────────────────────────
    save_training_plot(log_rows, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGCP 和弦识别训练")
    parser.add_argument("--data_dir",    type=str,   default="dataset/raw")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--model",       type=str,   default="small",
                        choices=["small", "large"])
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_dir",    type=str,   default="models")
    args = parser.parse_args()
    main(args)
