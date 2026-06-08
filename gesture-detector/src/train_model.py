import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS_DIR = os.path.join(BASE_DIR, "data", "windows")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading data")
X_train = np.load(os.path.join(WINDOWS_DIR, "X_Train.npy"))
y_train = np.load(os.path.join(WINDOWS_DIR, "y_Train.npy"))
X_test  = np.load(os.path.join(WINDOWS_DIR, "X_Test.npy"))
y_test  = np.load(os.path.join(WINDOWS_DIR, "y_Test.npy"))

print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test : {X_test.shape},  y_test : {y_test.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS_DIR = os.path.join(BASE_DIR, "data", "windows")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading data")
X_train = np.load(os.path.join(WINDOWS_DIR, "X_Train.npy"))
y_train = np.load(os.path.join(WINDOWS_DIR, "y_Train.npy"))
X_test  = np.load(os.path.join(WINDOWS_DIR, "X_Test.npy"))
y_test  = np.load(os.path.join(WINDOWS_DIR, "y_Test.npy"))

print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test : {X_test.shape},  y_test : {y_test.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS_DIR = os.path.join(BASE_DIR, "data", "windows")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading data")
X_train = np.load(os.path.join(WINDOWS_DIR, "X_Train.npy"))
y_train = np.load(os.path.join(WINDOWS_DIR, "y_Train.npy"))
X_test  = np.load(os.path.join(WINDOWS_DIR, "X_Test.npy"))
y_test  = np.load(os.path.join(WINDOWS_DIR, "y_Test.npy"))

print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test : {X_test.shape},  y_test : {y_test.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

class GestureClassifier(nn.Module):
    def __init__(self, input_size=99, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(30)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,       # reduced from 128
            num_layers=1,         # reduced from 2
            batch_first=True,
            dropout=0.0           # no dropout with single layer
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),      # heavy dropout before final layer
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        out = self.classifier(out[:, -1, :])
        return out

n_normal      = int(np.sum(y_train == 0))
n_shoplifting = int(np.sum(y_train == 1))
total         = n_normal + n_shoplifting
weight_normal      = total / (2 * n_normal)
weight_shoplifting = total / (2 * n_shoplifting)

# Boost shoplifting weight further to improve recall
weight_shoplifting *= 1.5

class_weights = torch.tensor([weight_normal, weight_shoplifting], dtype=torch.float32)
print(f"\n⚖️  Class weights: Normal={weight_normal:.2f}, Shoplifting={weight_shoplifting:.2f}")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

model     = GestureClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=7, factor=0.5
)

EPOCHS    = 80
THRESHOLD = 0.5  

print(f"\n Training for {EPOCHS} epochs (threshold={THRESHOLD})\n")

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = torch.argmax(outputs, dim=1)
        train_correct += (preds == y_batch).sum().item()
        train_total   += y_batch.size(0)

    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs  = model(X_batch)
            probs    = torch.softmax(outputs, dim=1)
            # predict shoplifting if confidence >= THRESHOLD
            preds    = (probs[:, 1] >= THRESHOLD).long()
            val_correct += (preds == y_batch).sum().item()
            val_total   += y_batch.size(0)

    train_acc = train_correct / train_total * 100
    val_acc   = val_correct   / val_total   * 100
    avg_loss  = train_loss    / len(train_loader)

    scheduler.step(val_acc)

    print(f"Epoch [{epoch:02d}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.1f}% | "
          f"Val Acc: {val_acc:.1f}%"
          + (" ← best" if val_acc > best_val_acc else ""))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))

#  Final Evaluation 
print(f"\n Best Val Accuracy: {best_val_acc:.1f}%")
print("\n Final Classification Report:")

model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth")))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs   = torch.softmax(outputs, dim=1)
        preds   = (probs[:, 1] >= THRESHOLD).cpu().numpy().astype(int)
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print(classification_report(all_labels, all_preds,
                             target_names=["Normal", "Shoplifting"]))
print(" Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\n   Rows = Actual, Cols = Predicted")
print("   [[TN  FP]")
print("    [FN  TP]]")

n_normal      = int(np.sum(y_train == 0))
n_shoplifting = int(np.sum(y_train == 1))
total         = n_normal + n_shoplifting
weight_normal      = total / (2 * n_normal)
weight_shoplifting = total / (2 * n_shoplifting)

# Boost shoplifting weight further to improve recall
weight_shoplifting *= 1.5

class_weights = torch.tensor([weight_normal, weight_shoplifting], dtype=torch.float32)
print(f"\n⚖️  Class weights: Normal={weight_normal:.2f}, Shoplifting={weight_shoplifting:.2f}")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

model     = GestureClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5
)

EPOCHS    = 80
THRESHOLD = 0.5  

print(f"\n Training for {EPOCHS} epochs (threshold={THRESHOLD})\n")

best_val_acc = 0.0

# Continue from best checkpoint if it exists
checkpoint_path = os.path.join(MODELS_DIR, "best_model.pth")
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("📂 Loaded existing best model — continuing training...")

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = torch.argmax(outputs, dim=1)
        train_correct += (preds == y_batch).sum().item()
        train_total   += y_batch.size(0)

    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs  = model(X_batch)
            probs    = torch.softmax(outputs, dim=1)
            # predict shoplifting if confidence >= THRESHOLD
            preds    = (probs[:, 1] >= THRESHOLD).long()
            val_correct += (preds == y_batch).sum().item()
            val_total   += y_batch.size(0)

    train_acc = train_correct / train_total * 100
    val_acc   = val_correct   / val_total   * 100
    avg_loss  = train_loss    / len(train_loader)

    scheduler.step(val_acc)

    print(f"Epoch [{epoch:02d}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.1f}% | "
          f"Val Acc: {val_acc:.1f}%"
          + (" ← best" if val_acc > best_val_acc else ""))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))

#  Final Evaluation 
print(f"\n Best Val Accuracy: {best_val_acc:.1f}%")
print("\n Final Classification Report:")

model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth")))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs   = torch.softmax(outputs, dim=1)
        preds   = (probs[:, 1] >= THRESHOLD).cpu().numpy().astype(int)
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print(classification_report(all_labels, all_preds,
                             target_names=["Normal", "Shoplifting"]))
print(" Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\n   Rows = Actual, Cols = Predicted")
print("   [[TN  FP]")
print("    [FN  TP]]")

n_normal      = int(np.sum(y_train == 0))
n_shoplifting = int(np.sum(y_train == 1))
total         = n_normal + n_shoplifting
weight_normal      = total / (2 * n_normal)
weight_shoplifting = total / (2 * n_shoplifting)

# Boost shoplifting weight further to improve recall
weight_shoplifting *= 1.5

class_weights = torch.tensor([weight_normal, weight_shoplifting], dtype=torch.float32)
print(f"\n⚖️  Class weights: Normal={weight_normal:.2f}, Shoplifting={weight_shoplifting:.2f}")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

model     = GestureClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=7, factor=0.5
)

EPOCHS    = 80
THRESHOLD = 0.5  

print(f"\n Training for {EPOCHS} epochs (threshold={THRESHOLD})\n")

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = torch.argmax(outputs, dim=1)
        train_correct += (preds == y_batch).sum().item()
        train_total   += y_batch.size(0)

    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs  = model(X_batch)
            probs    = torch.softmax(outputs, dim=1)
            # predict shoplifting if confidence >= THRESHOLD
            preds    = (probs[:, 1] >= THRESHOLD).long()
            val_correct += (preds == y_batch).sum().item()
            val_total   += y_batch.size(0)

    train_acc = train_correct / train_total * 100
    val_acc   = val_correct   / val_total   * 100
    avg_loss  = train_loss    / len(train_loader)

    scheduler.step(val_acc)

    print(f"Epoch [{epoch:02d}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.1f}% | "
          f"Val Acc: {val_acc:.1f}%"
          + (" ← best" if val_acc > best_val_acc else ""))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))

#  Final Evaluation 
print(f"\n Best Val Accuracy: {best_val_acc:.1f}%")
print("\n Final Classification Report:")

model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth")))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs   = torch.softmax(outputs, dim=1)
        preds   = (probs[:, 1] >= THRESHOLD).cpu().numpy().astype(int)
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print(classification_report(all_labels, all_preds,
                             target_names=["Normal", "Shoplifting"]))
print(" Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\n   Rows = Actual, Cols = Predicted")
print("   [[TN  FP]")
print("    [FN  TP]]")