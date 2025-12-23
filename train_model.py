import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import pickle
import os

# ================================
# 1. Load dataset (neutral + A-Z)
# ================================
print("="*60)
print("🚀 TRAINING LSTM MODEL (neutral + A-Z)")
print("="*60)

labels = ['neutral'] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
dfs = {}
missing = []

print("\n[1/5] Loading dataset...")
for lbl in labels:
    fname = f"{lbl}.txt"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        dfs[lbl] = df
        print(f"✅ {lbl:7s}: {len(df)} frames -> {fname}")
    else:
        missing.append(fname)

if missing:
    print("\n❌ ERROR: File berikut belum ada:")
    for f in missing:
        print("   -", f)
    print("   Kumpulkan dulu semua data gesture dengan program data collection.")
    exit()

print(f"\nTotal kelas: {len(labels)}")


# ================================
# 2. Build sequences (sliding window)
# ================================
no_of_timesteps = 20
X, y = [], []

label_to_id = {lbl: idx for idx, lbl in enumerate(labels)}
# contoh: {'neutral':0, 'A':1, 'B':2, ... 'Z':26}

print("\n[2/5] Building sequences...")

def build_seq(df, label_id):
    data = df.values   # (N, 63)
    n = len(data)
    for i in range(no_of_timesteps, n):
        X.append(data[i-no_of_timesteps:i, :])   # (20, 63)
        y.append(label_id)

for lbl, df in dfs.items():
    lbl_id = label_to_id[lbl]
    build_seq(df, lbl_id)
    print(f"   Kelas {lbl:7s} -> id {lbl_id:2d}, sequence: {len(df) - no_of_timesteps}")

X = np.array(X)
y = np.array(y)
print("\nX shape:", X.shape, "y shape:", y.shape)

if X.shape[0] == 0:
    print("❌ ERROR: Tidak ada sequence. Cek jumlah frame dan no_of_timesteps.")
    exit()


# ================================
# 3. Split train-test
# ================================
print("\n[3/5] Splitting train-test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

print("✅ Train samples:", len(X_train))
print("✅ Test  samples:", len(X_test))


# ================================
# 4. Build & train LSTM model
# ================================
print("\n[4/5] Building LSTM model...")

num_classes = len(labels)   # 27

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))  # 27 kelas

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled. Start training...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# ================================
# 5. Save model + label map
# ================================
print("\n[5/5] Saving model...")

label_map = {idx: lbl for lbl, idx in label_to_id.items()}

model_dict = {
    'model': model,
    'labels': label_map
}

with open('gesture_model_az.pkl', 'wb') as f:
    pickle.dump(model_dict, f)

print("\n✅ Gesture LSTM model saved to gesture_model_az.pkl")
print("   Kelas:", label_map)
print("="*60)
