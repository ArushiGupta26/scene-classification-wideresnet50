"""Train and evaluate a Wide ResNet-50-2 scene classifier."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_transforms(image_size: int):
    # ImageNet normalisation matches the pretrained TorchVision weights.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def make_loaders(train_dir: Path, test_dir: Path, image_size: int,
                 batch_size: int, seed: int, num_workers: int):
    train_tf, eval_tf = make_transforms(image_size)
    train_aug = datasets.ImageFolder(str(train_dir), transform=train_tf)
    train_eval = datasets.ImageFolder(str(train_dir), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)

    if train_aug.classes != test_ds.classes:
        raise ValueError(
            f"Train/test class folders differ: {train_aug.classes} vs {test_ds.classes}"
        )
    if len(train_aug.classes) != 6:
        print(f"Warning: expected 6 classes, found {len(train_aug.classes)}: {train_aug.classes}")

    targets = np.asarray(train_aug.targets)
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=targets
    )
    train_ds = Subset(train_aug, train_idx.tolist())
    val_ds = Subset(train_eval, val_idx.tolist())

    device = choose_device()
    # Keep workers conservative on macOS and avoid CUDA-only assumptions.
    effective_workers = max(0, num_workers)
    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=effective_workers,
        pin_memory=pin_memory, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=effective_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=effective_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, train_aug.classes


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    weights = models.Wide_ResNet50_2_Weights.DEFAULT
    model = models.wide_resnet50_2(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # The replacement classification head must always be trainable.
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets, all_preds = [], []

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_targets.extend(targets.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
    }


def evaluate(model, loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * images.size(0)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "loss": float(total_loss / max(1, len(loader.dataset))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_report": classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True,
                        help="ImageFolder root containing one subfolder per class.")
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="Held-out ImageFolder root containing one subfolder per class.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train only the final classification layer.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if not args.train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {args.train_dir}")
    if not args.test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    seed_everything(args.seed)
    device = choose_device()
    print(f"Device: {device}")
    print(f"Backbone frozen: {args.freeze_backbone}")

    train_loader, val_loader, test_loader, class_names = make_loaders(
        args.train_dir, args.test_dir, args.image_size, args.batch_size,
        args.seed, args.num_workers,
    )
    model = build_model(len(class_names), args.freeze_backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                  "learning_rate_next_epoch": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, macro-F1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "architecture": "wide_resnet50_2",
                "freeze_backbone": args.freeze_backbone,
                "seed": args.seed,
            }, checkpoint_path)
            print(f"  Saved best checkpoint: {checkpoint_path}")

    # Load the best validation checkpoint before touching the held-out test set.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, class_names)
    results = {
        "architecture": "torchvision.wide_resnet50_2",
        "dataset": "Intel Image Classification",
        "classes": class_names,
        "device": str(device),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "freeze_backbone": args.freeze_backbone,
        "selection_metric": "validation macro-F1",
        "history": history,
        "best_validation_macro_f1": best_val_f1,
        "test": test_metrics,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nHeld-out test metrics")
    print(f"Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"Macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro recall:    {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {test_metrics['macro_f1']:.4f}")
    print(f"Saved metrics:   {metrics_path}")


if __name__ == "__main__":
    main()
