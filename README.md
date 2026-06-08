# ShopGuard-CV 

A computer vision pipeline for detecting shoplifting behavior in surveillance footage using **MediaPipe pose estimation** and an **LSTM-based gesture classifier**.

---

## Demo

> Real-time inference running on webcam/video — skeleton overlay with live confidence bar.

```
[ SHOPLIFTING DETECTED ████████████████░░░░ 78% ]   ← red alert
[ Normal               ████████░░░░░░░░░░░░ 42% ]   ← green
```

---

## How It Works

```
Video Input (webcam / file)
        ↓
Person Detection + Pose Estimation (MediaPipe — 33 keypoints)
        ↓
Keypoint Extraction (99 values per frame: x, y, visibility × 33)
        ↓
Sliding Window (30 frames @ step=15)
        ↓
LSTM Classifier → Normal / Shoplifting
        ↓
Real-time overlay with confidence bar
```

The key insight: instead of classifying raw pixels, the model learns from **body movement patterns** — sequences of joint positions over time. This makes it lightweight, interpretable, and camera-resolution independent.

---

## Project Structure

```
ShopGuard-CV/
├── gesture-detector/
│   ├── data/
│   │   ├── raw_videos/
│   │   │   ├── Train/
│   │   │   │   ├── Shoplifting/       ← UCF-Crime shoplifting PNGs
│   │   │   │   └── NormalVideos/      ← Normal activity PNGs
│   │   │   └── Test/
│   │   │       ├── Shoplifting/
│   │   │       └── NormalVideos/
│   │   ├── keypoints/                 ← extracted .npy files (one per clip)
│   │   └── windows/                   ← sliding window arrays for training
│   ├── models/
│   │   └── best_model.pth             ← trained LSTM weights
│   └── src/
│       ├── extract_keypoints.py       ← Step 3: keypoint extraction (shoplifting)
│       ├── extract_keypoints_normal.py← Step 3: keypoint extraction (normal)
│       ├── prepare_windows.py         ← Step 4: sliding window builder
│       ├── filter_and_rebuild.py      ← Step 4b: quality filter (removes zero frames)
│       ├── train_model.py             ← Step 5: LSTM training
│       ├── train_random_forest.py     ← Step 5b: Random Forest baseline
│       └── inference.py              ← Step 6: real-time inference
├── LICENSE
└── README.md
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Pose Estimation | MediaPipe 0.10.13 |
| Object Detection | YOLOv8 (optional) |
| Deep Learning | PyTorch |
| Classical ML | scikit-learn |
| Video Processing | OpenCV |
| Data Handling | NumPy, Pandas |

---

## Setup

```bash
git clone https://github.com/SSaadAKHTAR/ShopGuard-CV
cd ShopGuard-CV

python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install opencv-python mediapipe==0.10.13 torch scikit-learn numpy joblib
```

---

## Run Inference

```bash
# Webcam (live)
python gesture-detector/src/inference.py

# Video file — edit inference.py line:
VIDEO_SOURCE = "path/to/your/video.mp4"
```

**Controls:** Press `Q` to quit.

---

## Train From Scratch

```bash
# 1. Extract keypoints from dataset
python gesture-detector/src/extract_keypoints.py
python gesture-detector/src/extract_keypoints_normal.py

# 2. Filter noisy frames and build windows
python gesture-detector/src/filter_and_rebuild.py

# 3. Train LSTM
python gesture-detector/src/train_model.py

# 4. (Optional) Train Random Forest baseline
python gesture-detector/src/train_random_forest.py
```

---

## Dataset

[UCF-Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) — real surveillance footage with labeled shoplifting and normal activity clips.

**Data split used:**

| Split | Shoplifting Clips | Normal Clips |
|---|---|---|
| Train | 34 | 34 |
| Test | 16 | 16 |

---

## Results

| Model | Val Accuracy | Shoplifting Recall | Shoplifting Precision |
|---|---|---|---|
| LSTM (2-layer, h=128) | 58.8% | 0.80 | 0.48 |
| LSTM (1-layer, h=64) | **62.8%** | **0.87** | **0.51** |
| Random Forest | 41.0% | 0.40 | 0.30 |

Best model: **1-layer LSTM, hidden=64, threshold=0.5**

---

## Key Finding — Data Quality Analysis

A major research finding from this project:

> **76.8% of UCF-Crime frames produced zero keypoints** via MediaPipe pose estimation.

```
Total frames  : 455,166
Zero frames   : 349,506  (76.8%)
Usable frames :  105,660  (23.2%)
```

This is caused by the nature of surveillance footage — distant cameras, low resolution, difficult angles — which MediaPipe's pose estimator cannot reliably handle. After filtering clips where >85% of frames were empty, only 26/63 train clips and 15/37 test clips were usable.

This dataset limitation is the primary performance bottleneck. Future work should use:
- Datasets with closer camera angles
- Higher resolution footage
- YOLOv8-Pose instead of MediaPipe for better small-person detection

---

## Architecture

```
Input: (batch, 30, 99)        ← 30 frames, 33 landmarks × 3 values
         ↓
BatchNorm1d(30)                ← normalize each timestep
         ↓
LSTM(input=99, hidden=64)      ← learn temporal patterns
         ↓
last timestep output: (batch, 64)
         ↓
Dropout(0.5)
         ↓
Linear(64 → 2)                 ← Normal / Shoplifting
         ↓
Softmax → confidence score
```

---

## Limitations & Future Work

- **Dataset quality:** UCF-Crime is filmed from far distances; MediaPipe struggles with small figures
- **Better pose model:** YOLOv8-Pose handles distant/occluded people significantly better
- **More data:** ~1000 training windows is too small for deep learning to generalize
- **Multi-person:** current pipeline handles one person per frame
- **Temporal smoothing:** add a rolling average over last N predictions to reduce flickering

---

## Author

**Syed Saad Akhtar**  
Research Assistant, Microelectronics Research Lab (MERL), UIT Karachi

- GitHub: [github.com/SSaadAKHTAR](https://github.com/SSaadAKHTAR)
- LinkedIn: [linkedin.com/in/syed-saad-akhtar-a194a8295](https://linkedin.com/in/syed-saad-akhtar-a194a8295)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
