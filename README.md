# ShopGuard-CV 🛒🔍

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
│   │   │   │   ├── Shoplifting/       ← MP4 shoplifting clips
│   │   │   │   └── NormalVideos/      ← MP4 normal activity clips
│   │   │   └── Test/
│   │   │       ├── Shoplifting/
│   │   │       └── NormalVideos/
│   │   ├── keypoints/                 ← extracted .npy files (one per clip)
│   │   └── windows/                   ← sliding window arrays for training
│   ├── models/
│   │   └── best_model.pth             ← trained LSTM weights
│   └── src/
│       ├── extract_keypoints_video.py ← Step 3: keypoint extraction from MP4
│       ├── filter_and_rebuild.py      ← Step 4: quality filter + window builder
│       ├── train_model.py             ← Step 5: LSTM training
│       ├── train_random_forest.py     ← Step 5b: Random Forest baseline
│       ├── check_zeros.py             ← Data quality diagnostic tool
│       └── inference.py               ← Step 6: real-time inference
├── LICENSE
└── README.md
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Pose Estimation | MediaPipe 0.10.13 |
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

# Video file — edit inference.py and change:
VIDEO_SOURCE = "path/to/your/video.mp4"
```

**Controls:** Press `Q` to quit.

---

## Train From Scratch

```bash
# 1. Extract keypoints from MP4 videos
python gesture-detector/src/extract_keypoints_video.py

# 2. Filter noisy frames and build sliding windows
python gesture-detector/src/filter_and_rebuild.py

# 3. Train LSTM classifier
python gesture-detector/src/train_model.py

# 4. (Optional) Train Random Forest baseline
python gesture-detector/src/train_random_forest.py
```

---

## Dataset

### Final Dataset — Kaggle Shoplifting MP4 Dataset
[Shoplifting Dataset by Jashwant Singh Yadav](https://www.kaggle.com/datasets/jashwantsinghyadav/shoplifting-dataset)

Close-range indoor surveillance footage at 640×480 resolution, 30fps. People are clearly visible and MediaPipe can detect pose keypoints reliably.

**Data split (80/20):**

| Split | Shoplifting Clips | Normal Clips |
|---|---|---|
| Train | 73 | 72 |
| Test | 19 | 18 |
| **Total** | **92** | **90** |

### Initial Dataset — UCF-Crime (Abandoned)
[UCF-Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)

This dataset was initially used but abandoned after a data quality analysis revealed it was incompatible with MediaPipe pose estimation. See the Key Finding section below.

---

## Results

| Model | Dataset | Val Accuracy | Shoplifting Recall | Shoplifting Precision |
|---|---|---|---|---|
| LSTM (1-layer, h=64) | UCF-Crime PNGs | 62.8% | 0.87 | 0.51 |
| Random Forest | UCF-Crime PNGs | 41.0% | 0.40 | 0.30 |
| **LSTM (2-layer, h=128)** | **Kaggle MP4 Dataset** | **75.2%** | **0.74** | **0.77** |

**Best model: 2-layer LSTM, hidden=128, Kaggle MP4 dataset**

The +12.4% accuracy jump came entirely from switching datasets — with no architecture changes — proving that data quality is the dominant factor in pose-based action recognition.

---

## Key Finding — Data Quality Is Everything

The most important discovery from this project came from a systematic analysis of keypoint detection quality across datasets.

### UCF-Crime Analysis (Initial Dataset)

```
Total frames  : 455,166
Zero frames   : 349,506  (76.8%)   ← MediaPipe detected NO person
Usable frames :  105,660  (23.2%)
Usable clips  : 41 / 100
```

UCF-Crime is filmed from distant ceiling-mounted cameras at low resolution. MediaPipe's pose estimator requires a reasonably visible human figure and fails almost completely on tiny, far-away subjects. Training on 76.8% empty data meant the LSTM was learning from noise.

### Kaggle MP4 Dataset Analysis (Final Dataset)

```
Total frames  :  60,702
Zero frames   :   2,035  (3.4%)    ← MediaPipe detected person reliably
Usable frames :  58,667  (96.6%)
Usable clips  : 182 / 182
```

The Kaggle dataset uses close-range indoor cameras where people fill a significant portion of the frame. MediaPipe detected keypoints in 96.6% of frames — a complete turnaround.

### Impact on Accuracy

| Metric | UCF-Crime | Kaggle MP4 | Change |
|---|---|---|---|
| Zero keyframe rate | 76.8% | 3.4% | -73.4% |
| Usable clips | 41% | 100% | +59% |
| Val Accuracy | 62.8% | **75.2%** | **+12.4%** |

> **Conclusion:** For pose-based action recognition, camera proximity matters more than dataset size. A smaller, well-captured dataset dramatically outperforms a large surveillance dataset with poor keypoint visibility.

---

## Architecture

```
Input: (batch, 30, 99)         ← 30 frames × 33 landmarks × 3 values (x, y, visibility)
         ↓
BatchNorm1d(30)                 ← normalize each timestep
         ↓
LSTM(input=99, hidden=128, layers=2, dropout=0.3)   ← learn temporal motion patterns
         ↓
last timestep output: (batch, 128)
         ↓
Dropout(0.4)
         ↓
Linear(128 → 64) → ReLU
         ↓
Linear(64 → 2)                  ← Normal / Shoplifting
         ↓
Softmax → confidence score
```

---

## What the Model Learns

The LSTM learns **temporal sequences of body joint positions** — not what a person looks like, but how they move over time. A shoplifting action involves a characteristic sequence:

```
browsing posture → reaching motion → concealment gesture → walking away
```

Each of these maps to specific patterns in the wrist, elbow, shoulder, and hip keypoints across 30 consecutive frames (~1 second of video).

---

## Limitations & Future Work

- **Multi-person scenes:** current pipeline tracks only one person per frame
- **Camera angle dependency:** model trained on close-range footage may not generalize to ceiling cameras
- **Temporal smoothing:** add a rolling average over last N predictions to reduce flickering in real-time inference
- **YOLOv8-Pose:** would improve keypoint detection at longer distances compared to MediaPipe
- **Larger dataset:** 182 clips is sufficient for a proof-of-concept; production deployment would need 1000+ clips

---

## Author

**Syed Saad Akhtar**
Research Assistant, Microelectronics Research Lab (MERL), UIT Karachi

- GitHub: [github.com/SSaadAKHTAR](https://github.com/SSaadAKHTAR)
- LinkedIn: [linkedin.com/in/syed-saad-akhtar-a194a8295](https://linkedin.com/in/syed-saad-akhtar-a194a8295)

---

## License

MIT License — see [LICENSE](LICENSE) for details.