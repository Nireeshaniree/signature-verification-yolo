# 🖋️ Signature Verification Detection: Enhancing the Image Detection

An image-based system that tells **genuine signatures from forged ones** using **YOLO object detection**. Images are cleaned with OpenCV, a YOLO model is trained to detect and classify each signature, and performance is measured with accuracy, precision, recall, F1-score and a confusion matrix.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics%20YOLO-111F68?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

## ✨ Features

- OpenCV preprocessing: grayscale, denoising, optional thresholding, resizing
- YOLO training for two classes: `genuine` and `forged`
- Evaluation with accuracy, precision, recall, F1-score and a confusion matrix
- Single-image prediction script that saves an annotated result

## 📁 Project Structure

```
signature-verification-yolo/
├── dataset/
│   ├── images/{train,val,test}/   # signature images
│   └── labels/{train,val,test}/   # YOLO-format .txt labels
├── models/                        # keep trained weights here (optional)
├── results/                       # metrics and confusion matrix
├── data.yaml                      # dataset config
├── preprocess.py                  # OpenCV image cleaning
├── train.py                       # train the YOLO model
├── evaluate.py                    # metrics + confusion matrix
├── predict.py                     # verify one image
└── requirements.txt
```

## 🛠️ Setup

```bash
git clone https://github.com/Nireeshaniree/signature-verification-yolo.git
cd signature-verification-yolo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Dataset Format

Put images in `dataset/images/{train,val,test}` and one label file per image in `dataset/labels/{train,val,test}` with the **same file name** (`sign_001.png` → `sign_001.txt`).

Each label line uses the YOLO format, with every value normalized between 0 and 1:

```
<class_id> <x_center> <y_center> <width> <height>
```

Classes: `0 = genuine`, `1 = forged`.

> The dataset is not included in this repository. Add your own signature images or use a public signature dataset, and check its license before sharing it.

## 🚀 Usage

**1. (Optional) Preprocess images**
```bash
python preprocess.py --src raw_images --dst dataset/images/train --size 640
```

**2. Train**
```bash
python train.py --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16
```
Best weights are saved to `runs/signature/train/weights/best.pt`.

**3. Evaluate**
```bash
python evaluate.py --weights runs/signature/train/weights/best.pt
```
This prints accuracy, precision, recall and F1-score, and saves `results/metrics.txt` and `results/confusion_matrix.png`.

**4. Predict on one image**
```bash
python predict.py --weights runs/signature/train/weights/best.pt --image sample.png
```

## 📊 Results

<!-- EDIT: after running evaluate.py, paste your real numbers here -->

| Metric    | Score |
|-----------|-------|
| Accuracy  | _fill in from results/metrics.txt_ |
| Precision | _fill in_ |
| Recall    | _fill in_ |
| F1-score  | _fill in_ |

<!-- EDIT: add the confusion matrix image after you generate it:
![Confusion Matrix](results/confusion_matrix.png) -->

## 🧰 Tech Stack

`Python` `Ultralytics YOLO` `OpenCV` `PyTorch` `NumPy` `Pandas` `scikit-learn` `Jupyter Notebook` `Google Colab`

## 🔮 Future Improvements

- Train on a larger and more varied set of signatures
- Add data augmentation for more reliable forgery detection
- Build a small web interface (Django or Streamlit) for uploading signatures

## 👩‍💻 Author

**Kanipakam Nireesha**: [LinkedIn](https://www.linkedin.com/in/nireesha-k-66474b257) · [GitHub](https://github.com/Nireeshaniree)
