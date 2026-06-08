import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "keypoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = {
    "Train": {
        "Shoplifting":  os.path.join(DATA_DIR, "raw_videos", "Train", "Shoplifting"),
        "NormalVideos": os.path.join(DATA_DIR, "raw_videos", "Train", "NormalVideos"),
    },
    "Test": {
        "Shoplifting":  os.path.join(DATA_DIR, "raw_videos", "Test", "Shoplifting"),
        "NormalVideos": os.path.join(DATA_DIR, "raw_videos", "Test", "NormalVideos"),
    }
}

# ── Extract keypoints from a single video file ────────────────
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    with mp_pose.Pose(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                kp = np.array([
                    [lm.x, lm.y, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ]).flatten()
            else:
                kp = np.zeros(99)

            keypoints_sequence.append(kp)

    cap.release()
    return np.array(keypoints_sequence)  # shape: (num_frames, 99)

# ── Process all splits and classes ───────────────────────────
total_clips = 0
total_zeros = 0
total_frames = 0

for split_name, classes in SPLITS.items():
    print(f"\n📂 {split_name}")

    for class_name, folder_path in classes.items():
        label_str = "🛒 Shoplifting" if class_name == "Shoplifting" else "🚶 Normal"
        print(f"\n   {label_str} — {folder_path}")

        video_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        print(f"   Found {len(video_files)} videos")

        for video_file in sorted(video_files):
            video_path = os.path.join(folder_path, video_file)
            clip_name  = os.path.splitext(video_file)[0]  # remove extension

            # Safe filename — remove spaces and parentheses
            safe_name  = clip_name.replace(" ", "_").replace("(", "").replace(")", "")
            save_name  = f"{split_name}_{class_name}_{safe_name}.npy"
            save_path  = os.path.join(OUTPUT_DIR, save_name)

            keypoints  = extract_keypoints_from_video(video_path)

            # Stats
            zero_frames = np.sum(np.all(keypoints == 0, axis=1))
            zero_pct    = zero_frames / len(keypoints) * 100 if len(keypoints) > 0 else 0
            total_clips  += 1
            total_zeros  += zero_frames
            total_frames += len(keypoints)

            np.save(save_path, keypoints)

            status = "⚠️ " if zero_pct > 50 else "✅"
            print(f"      {status} {video_file}: "
                  f"{len(keypoints)} frames, "
                  f"{zero_pct:.0f}% zeros → saved")

# ── Overall quality report ────────────────────────────────────
overall_zero_pct = total_zeros / total_frames * 100 if total_frames > 0 else 0
print(f"\n{'='*55}")
print(f"✅ Extraction complete!")
print(f"   Total clips   : {total_clips}")
print(f"   Total frames  : {total_frames}")
print(f"   Zero frames   : {total_zeros} ({overall_zero_pct:.1f}%)")
print(f"   Saved to      : {OUTPUT_DIR}")
print(f"{'='*55}")

if overall_zero_pct < 30:
    print("🎉 Excellent data quality! MediaPipe detected people well.")
elif overall_zero_pct < 60:
    print("⚠️  Moderate quality. Should still train better than UCF-Crime.")
else:
    print("❌ High zero rate. Camera angles may still be problematic.")