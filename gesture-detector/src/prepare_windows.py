import numpy as np
import os
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYPOINTS_DIR = os.path.join(BASE_DIR, "data", "keypoints")
OUTPUT_DIR    = os.path.join(BASE_DIR, "data", "windows")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 30
STEP        = 15
MAX_WINDOWS_PER_CLIP = 50  # cap per clip to prevent huge clips dominating

def create_windows(keypoints, label, window_size=30, step=15, max_windows=50):
    X, y = [], []
    for i in range(0, len(keypoints) - window_size + 1, step):
        if len(X) >= max_windows:  # stop once limit reached
            break
        window = keypoints[i : i + window_size]
        if window.shape == (window_size, 99):
            X.append(window)
            y.append(label)
    return X, y

def get_label(filename):
    if "Normal" in os.path.basename(filename):
        return 0
    return 1

for split in ["Train", "Test"]:

    print(f"\n Processing {split} windows")

    X_all, y_all = [], []

    npy_files = glob.glob(os.path.join(KEYPOINTS_DIR, f"{split}_*.npy"))
    print(f"   Found {len(npy_files)} .npy files")

    for npy_path in npy_files:
        keypoints = np.load(npy_path)
        label     = get_label(npy_path)

        X_windows, y_windows = create_windows(
            keypoints, label,
            max_windows=MAX_WINDOWS_PER_CLIP
        )

        X_all.extend(X_windows)
        y_all.extend(y_windows)

        print(f"   {'🛒 Shoplifting' if label == 1 else '🚶 Normal    '} "
              f"{os.path.basename(npy_path)}: "
              f"{len(keypoints)} frames → {len(X_windows)} windows")

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    indices = np.random.permutation(len(X_all))
    X_all   = X_all[indices]
    y_all   = y_all[indices]

    np.save(os.path.join(OUTPUT_DIR, f"X_{split}.npy"), X_all)
    np.save(os.path.join(OUTPUT_DIR, f"y_{split}.npy"), y_all)

    shoplifting_count = int(np.sum(y_all == 1))
    normal_count      = int(np.sum(y_all == 0))

    print(f"\n    {split} summary:")
    print(f"      Total windows : {len(X_all)}")
    print(f"      Shoplifting   : {shoplifting_count} windows")
    print(f"      Normal        : {normal_count} windows")
    print(f"      Shape X       : {X_all.shape}")
    print(f"      Shape y       : {y_all.shape}")

print("\n Window preparation complete!")
print(f"   Saved to: {OUTPUT_DIR}")