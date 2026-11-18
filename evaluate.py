"""Evaluate the trained model on the test set.

Reports accuracy, precision, recall and F1-score (image level: each test image
is treated as genuine or forged) and saves a confusion matrix plot.

Image-level label = the class in the image's YOLO label file.
Image-level prediction = the class of the highest-confidence detection
(if nothing is detected, the image counts as "no detection" and is treated as wrong).

Usage:
    python evaluate.py --weights runs/signature/train/weights/best.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO

CLASS_NAMES = ["genuine", "forged"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_true_label(label_path: Path):
    """Return the class id from a YOLO label file, or None if missing/empty."""
    if not label_path.exists():
        return None
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    return int(lines[0].split()[0])


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO signature verifier.")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--images", default="dataset/images/test")
    parser.add_argument("--labels", default="dataset/labels/test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    y_true, y_pred = [], []
    skipped = 0

    for image_path in sorted(Path(args.images).iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        true_cls = read_true_label(Path(args.labels) / f"{image_path.stem}.txt")
        if true_cls is None:
            skipped += 1
            continue

        result = model.predict(str(image_path), conf=args.conf, verbose=False)[0]
        if len(result.boxes) == 0:
            pred_cls = 1 - true_cls  # no detection counts as a miss
        else:
            best = result.boxes.conf.argmax().item()
            pred_cls = int(result.boxes.cls[best].item())

        y_true.append(true_cls)
        y_pred.append(pred_cls)

    if not y_true:
        raise SystemExit("No labelled test images found. Check --images and --labels.")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Test images evaluated: {len(y_true)} (skipped without label: {skipped})")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Save metrics as text
    (out_dir / "metrics.txt").write_text(
        f"images={len(y_true)}\naccuracy={accuracy:.4f}\nprecision={precision:.4f}\n"
        f"recall={recall:.4f}\nf1={f1:.4f}\n"
    )

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"Saved metrics and confusion matrix to {out_dir}/")


if __name__ == "__main__":
    main()
