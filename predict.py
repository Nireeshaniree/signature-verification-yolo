"""Check a single signature image and print whether it looks genuine or forged.

Usage:
    python predict.py --weights runs/signature/train/weights/best.pt --image sample.png
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Verify one signature image.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = model.predict(args.image, conf=args.conf, verbose=False)[0]

    if len(result.boxes) == 0:
        print("No signature detected. Try a clearer image or a lower --conf value.")
        return

    best = result.boxes.conf.argmax().item()
    label = result.names[int(result.boxes.cls[best].item())]
    confidence = float(result.boxes.conf[best].item())
    print(f"Prediction: {label.upper()} (confidence {confidence:.2%})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{Path(args.image).name}"
    cv2.imwrite(str(out_path), result.plot())
    print(f"Annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
