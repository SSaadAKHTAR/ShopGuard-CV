import numpy as np
import glob
import os

KEYPOINTS_DIR = "/home/saad/Desktop/OpenCV_project/ShopGuard-CV/gesture-detector/data/keypoints"

total_frames = 0
zero_frames  = 0

for npy_path in glob.glob(os.path.join(KEYPOINTS_DIR, "*.npy")):
    kp = np.load(npy_path)          # shape: (frames, 99)
    zero_mask    = np.all(kp == 0, axis=1)
    total_frames += len(kp)
    zero_frames  += zero_mask.sum()

pct = zero_frames / total_frames * 100
print(f"Total frames : {total_frames}")
print(f"Zero frames  : {zero_frames} ({pct:.1f}%)")