import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle
import os

print("="*60)
print("🚀 TRAINING LSTM MODEL - GESTURE RECOGNITION")
print("="*60)

# ====== 1. Load dataset ======
print("\n[1/6] Loading dataset...")
try:
    neutral_df = pd.read_csv('neutral.txt')
    A_df       = pd.read_csv('A.txt')
    L_df       = pd.read_csv('L.txt')
    print(f"✅ neutral: {len(neutral_df)} frames")
    print(f"✅ A: {len(A_df)} frames")
    print(f"✅ L: {len(L_df)} frames")
except FileNotFoundError as e:
    print(f"❌ ERROR: File tidak ditemukan - {e}")
    print("   Jalankan hands_data_generation_new.py terlebih dahulu!")
    exit()

no_of_timesteps = 20   # sliding window
X, y = [], []

def build_seq(df, label):
    data = df.values
    n = len(data)
    for i in range(no_of_timesteps, n):
        X.append(data[i-no_of_timesteps:i, :])
        y.append(label)

print("\n[2/6] Building sequences...")
build_seq(neutral_df, 0)
build_seq(A_df, 1)
build_seq(L_df, 2)

X = np.array(X)
y = np.array(y)
print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

# ====== 2. Split train-test ======
print("\n[3/6] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)
print(f"✅ Train: {len(X_train)} samples")
print(f"✅ Test: {len(X_test)} samples")

# ====== 3. Build LSTM model (OPTIMIZED) ======
print("\n[4/6] Building LSTM model...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(32),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled")
model.summary()

# ====== 4. Train with callbacks ======
print("\n[5/6] Training model...")
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ====== 5. Evaluate ======
print("\n[6/6] Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*60}")
print(f"📊 HASIL TRAINING:")
print(f"{'='*60}")
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"{'='*60}")

# ====== 6. Save model + labels ======
print("\n💾 Saving model...")
model.save('gesture_model.h5')

label_map = {0: 'neutral', 1: 'A', 2: 'L'}
with open('gesture_labels.pkl', 'wb') as f:
    pickle.dump(label_map, f)

print(f"✅ gesture_model.h5 saved")
print(f"✅ gesture_labels.pkl saved")

print("\n" + "="*60)
print("🎉 TRAINING SELESAI!")
print("="*60)
print("📌 Langkah berikutnya:")
print("   1. Test dengan test_gesture_live.py")
print("   2. Jika akurasi bagus, jalankan weapon_access_system.py")
print("="*60)
