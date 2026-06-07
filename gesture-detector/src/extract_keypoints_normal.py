import cv2
import mediapipe as mp
import numpy as np
import os
from collections import defaultdict

mp_pose = mp.solutions.pose

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

SPLITS = {
    "Train": (os.path.join(DATA_DIR, "raw_videos", "Train", "NormalVideos"), 34),
    "Test":  (os.path.join(DATA_DIR, "raw_videos", "Test",  "NormalVideos"), 16),
}

OUTPUT_DIR = os.path.join(DATA_DIR, "keypoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_clip_name(filename):
    parts = filename.rsplit("_", 1)
    return parts[0]

def get_frame_number(filename):
    name = os.path.splitext(filename)[0]
    return int(name.rsplit("_", 1)[-1])

def extract_keypoints_from_clip(image_paths):
    keypoints_sequence = []

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3
    ) as pose:

        for img_path in image_paths:
            frame = cv2.imread(img_path)

            if frame is None:
                keypoints_sequence.append(np.zeros(99))
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                kp = np.array([
                    [lm.x, lm.y, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ]).flatten()
            else:
                kp = np.zeros(99)

            keypoints_sequence.append(kp)

    return np.array(keypoints_sequence)


for split_name, (folder_path, max_clips) in SPLITS.items():

    print(f"\n Processing {split_name} Normal (using {max_clips} clips)")

    clips = defaultdict(list)
    for fname in os.listdir(folder_path):
        if fname.endswith(".png"):
            clip = get_clip_name(fname)
            clips[clip].append(fname)

    print(f"   Found {len(clips)} total clips, selecting {max_clips}")

    #  use selected_clips in the loop
    selected_clips = list(clips.items())[:max_clips]

    for clip_name, filenames in selected_clips:

        filenames_sorted = sorted(filenames, key=get_frame_number)
        full_paths = [os.path.join(folder_path, f) for f in filenames_sorted]

        print(f"    {clip_name} → {len(full_paths)} frames", end="")

        keypoints = extract_keypoints_from_clip(full_paths)

        save_name = f"{split_name}_Normal_{clip_name}.npy"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        np.save(save_path, keypoints)

        print(f" → saved {keypoints.shape}")

print("\n Normal keypoint extraction complete!")
print(f"   Saved to: {OUTPUT_DIR}")
