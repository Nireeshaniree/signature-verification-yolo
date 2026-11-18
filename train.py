"""Train a YOLO model to detect genuine vs forged signatures.

Usage:
    python train.py
    python train.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
"""

import argparse

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO signature verifier.")
    parser.add_argument("--data", default="data.yaml", help="Dataset config file")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLO weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/signature")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=15,   # stop early if validation does not improve
        seed=42,       # reproducible runs
    )

    print(f"Done. Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
