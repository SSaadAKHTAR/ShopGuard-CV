import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os

# ── Model definition (must match training) ────────────────────
class GestureClassifier(nn.Module):
    def __init__(self, input_size=99, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(30)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        out = self.classifier(out[:, -1, :])
        return out

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# ── Load model ────────────────────────────────────────────────
device = torch.device("cpu")
model  = GestureClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Model loaded")

# ── MediaPipe ─────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ── Config ────────────────────────────────────────────────────
WINDOW_SIZE = 30
THRESHOLD   = 0.5
frame_buffer = []

# ── Colors ────────────────────────────────────────────────────
COLOR_NORMAL     = (0, 200, 0)    # green
COLOR_SHOPLIFTING = (0, 0, 255)   # red
COLOR_COLLECTING  = (200, 200, 0) # yellow — still collecting frames

# ── Open video — change to 0 for webcam ──────────────────────
VIDEO_SOURCE = 0   # 0 = webcam, or "path/to/video.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("❌ Could not open video source")
    exit()

print("🎥 Running inference... Press Q to quit")

current_label = "Collecting frames..."
current_color = COLOR_COLLECTING
confidence    = 0.0

with mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            kp = np.array([
                [lm.x, lm.y, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]).flatten()

            # Draw skeleton on frame
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,255,255),   thickness=2)
            )
        else:
            kp = np.zeros(99)

        frame_buffer.append(kp)

        if len(frame_buffer) >= WINDOW_SIZE:
            window = np.array(frame_buffer[-WINDOW_SIZE:])  # last 30 frames
            tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                probs  = torch.softmax(output, dim=1)
                conf_shoplifting = probs[0, 1].item()
                pred   = 1 if conf_shoplifting >= THRESHOLD else 0

            confidence = conf_shoplifting

            if pred == 1:
                current_label = "Normal"
                current_color = COLOR_NORMAL
                
            else:
                current_label = "SHOPLIFTING DETECTED"
                current_color = COLOR_SHOPLIFTING

            frame_buffer.pop(0)


        # Background bar at top
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

        # Main label
        cv2.putText(frame, current_label,
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    current_color, 2, cv2.LINE_AA)

        # Confidence bar
        if len(frame_buffer) >= WINDOW_SIZE or confidence > 0:
            bar_width = int(confidence * (w - 30))
            cv2.rectangle(frame, (15, 50), (15 + bar_width, 62),
                          current_color, -1)
            cv2.rectangle(frame, (15, 50), (w - 15, 62),
                          (100, 100, 100), 1)
            cv2.putText(frame, f"{confidence*100:.0f}%",
                        (w - 60, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)

        # Frame buffer progress (while collecting initial 30 frames)
        if len(frame_buffer) < WINDOW_SIZE:
            progress = len(frame_buffer) / WINDOW_SIZE
            bar_w    = int(progress * (w - 30))
            cv2.rectangle(frame, (15, 50), (15 + bar_w, 62),
                          COLOR_COLLECTING, -1)
            cv2.putText(frame,
                        f"Buffering {len(frame_buffer)}/{WINDOW_SIZE}",
                        (15, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)

        cv2.imshow("ShopGuard-CV", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Done")