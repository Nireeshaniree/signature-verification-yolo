"""Preprocess signature images with OpenCV.

Steps: grayscale -> denoise -> adaptive threshold (optional) -> resize.
The output is written to a new folder, so original images are never modified.

Usage:
    python preprocess.py --src raw_images --dst dataset/images/train --size 640
"""

import argparse
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def preprocess_image(image, size=640, binarize=False):
    """Return a cleaned, resized 3-channel image (YOLO expects 3 channels)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)

    if binarize:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )

    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description="Preprocess signature images.")
    parser.add_argument("--src", required=True, help="Folder with original images")
    parser.add_argument("--dst", required=True, help="Folder for processed images")
    parser.add_argument("--size", type=int, default=640, help="Output square size")
    parser.add_argument("--binarize", action="store_true", help="Apply adaptive threshold")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(src.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable file: {path.name}")
            continue
        cv2.imwrite(str(dst / path.name), preprocess_image(image, args.size, args.binarize))
        count += 1

    print(f"Processed {count} images -> {dst}")


if __name__ == "__main__":
    main()
