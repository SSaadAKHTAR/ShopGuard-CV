import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS_DIR = os.path.join(BASE_DIR, "data", "windows")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print(" Loading data")
X_train = np.load(os.path.join(WINDOWS_DIR, "X_Train.npy"))
y_train = np.load(os.path.join(WINDOWS_DIR, "y_Train.npy"))
X_test  = np.load(os.path.join(WINDOWS_DIR, "X_Test.npy"))
y_test  = np.load(os.path.join(WINDOWS_DIR, "y_Test.npy"))

print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test : {X_test.shape},  y_test : {y_test.shape}")

# LSTM needs (samples, 30, 99) but RF needs (samples, 2970)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat  = X_test.reshape(len(X_test),  -1)
print(f"\n   Flattened train: {X_train_flat.shape}")
print(f"   Flattened test : {X_test_flat.shape}")

n_normal      = int(np.sum(y_train == 0))
n_shoplifting = int(np.sum(y_train == 1))
print(f"\n  Train — Normal: {n_normal}, Shoplifting: {n_shoplifting}")

print("\n Training Random Forest...")

clf = RandomForestClassifier(
    n_estimators=300,       # number of trees
    max_depth=15,           # prevent overfitting
    min_samples_leaf=3,     # each leaf needs at least 3 samples
    class_weight='balanced',# handles imbalance automatically
    random_state=42,
    n_jobs=-1               # use all CPU cores
)

clf.fit(X_train_flat, y_train)
print(" Training complete!")

print("\n Evaluating on test set")
y_pred = clf.predict(X_test_flat)

print(classification_report(y_test, y_pred, target_names=["Normal", "Shoplifting"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n   Rows = Actual, Cols = Predicted")
print("   [[TN  FP]")
print("    [FN  TP]]")

#  Feature importance (top 10) 
print("\n🔍 Top 10 most important keypoint features:")
importances = clf.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]

LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]
COORDS = ["x", "y", "vis"]

for rank, idx in enumerate(top_indices):
    frame_num  = idx // 99
    kp_offset  = idx  % 99
    landmark   = LANDMARK_NAMES[kp_offset // 3]
    coord      = COORDS[kp_offset % 3]
    print(f"   {rank+1:2d}. frame={frame_num:2d} | {landmark}.{coord} "
          f"(importance={importances[idx]:.4f})")

# Save model 
model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
joblib.dump(clf, model_path)
print(f"\n Model saved to: {model_path}")