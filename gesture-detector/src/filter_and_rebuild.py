import numpy as np
import glob
import os

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYPOINTS_DIR = os.path.join(BASE_DIR, "data", "keypoints")
WINDOWS_DIR   = os.path.join(BASE_DIR, "data", "windows")
os.makedirs(WINDOWS_DIR, exist_ok=True)

WINDOW_SIZE          = 30
STEP                 = 15
MAX_WINDOWS_PER_CLIP = 50
MIN_VALID_RATIO      = 0.5   # skip window if >50% frames are zeros

def is_valid_window(window):
    """Return True if at least 50% of frames have detected keypoints."""
    zero_frames = np.sum(np.all(window == 0, axis=1))
    return (zero_frames / len(window)) < 0.5

def create_clean_windows(keypoints, label):
    X, y = [], []
    for i in range(0, len(keypoints) - WINDOW_SIZE + 1, STEP):
        if len(X) >= MAX_WINDOWS_PER_CLIP:
            break
        window = keypoints[i : i + WINDOW_SIZE]
        if window.shape == (WINDOW_SIZE, 99) and is_valid_window(window):
            X.append(window)
            y.append(label)
    return X, y

def get_label(filename):
    if "Normal" in os.path.basename(filename):
        return 0
    return 1

# ── Process Train and Test ─────────────────────────────────────
for split in ["Train", "Test"]:

    print(f"\n📂 Processing {split}...")

    X_all, y_all = [], []
    skipped_clips    = 0
    kept_clips       = 0

    npy_files = glob.glob(os.path.join(KEYPOINTS_DIR, f"{split}_*.npy"))

    for npy_path in npy_files:
        keypoints = np.load(npy_path)
        label     = get_label(npy_path)

        # Check overall clip quality first
        zero_ratio = np.sum(np.all(keypoints == 0, axis=1)) / len(keypoints)

        if zero_ratio > 0.85:
            # Clip is mostly empty — skip entirely
            print(f"   ⏭️  SKIP {os.path.basename(npy_path)} "
                  f"({zero_ratio*100:.0f}% zeros)")
            skipped_clips += 1
            continue

        X_windows, y_windows = create_clean_windows(keypoints, label)

        if len(X_windows) == 0:
            print(f"   ⏭️  SKIP {os.path.basename(npy_path)} "
                  f"(no valid windows after filtering)")
            skipped_clips += 1
            continue

        X_all.extend(X_windows)
        y_all.extend(y_windows)
        kept_clips += 1

        print(f"   {'🛒' if label==1 else '🚶'} "
              f"{os.path.basename(npy_path)}: "
              f"{zero_ratio*100:.0f}% zeros → {len(X_windows)} windows")

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Shuffle
    indices = np.random.permutation(len(X_all))
    X_all   = X_all[indices]
    y_all   = y_all[indices]

    np.save(os.path.join(WINDOWS_DIR, f"X_{split}.npy"), X_all)
    np.save(os.path.join(WINDOWS_DIR, f"y_{split}.npy"), y_all)

    shoplifting = int(np.sum(y_all == 1))
    normal      = int(np.sum(y_all == 0))

    print(f"\n   ✅ {split} summary:")
    print(f"      Clips kept    : {kept_clips}")
    print(f"      Clips skipped : {skipped_clips}")
    print(f"      Total windows : {len(X_all)}")
    print(f"      Shoplifting   : {shoplifting}")
    print(f"      Normal        : {normal}")
    print(f"      Shape X       : {X_all.shape}")

print("\n✅ Clean windows saved!")