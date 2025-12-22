import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import pickle

# ====== 1. Load dataset (neutral, A, L) ======
neutral_df = pd.read_csv('neutral.txt')
A_df       = pd.read_csv('A.txt')
L_df       = pd.read_csv('L.txt')

no_of_timesteps = 20   # sliding window
X, y = [], []

def build_seq(df, label):
    data = df.values    # (N, 63)
    n = len(data)
    for i in range(no_of_timesteps, n):
        X.append(data[i-no_of_timesteps:i, :])  # shape (20, 63)
        y.append(label)

build_seq(neutral_df, 0)  # 0 = neutral
build_seq(A_df, 1)        # 1 = A
build_seq(L_df, 2)        # 2 = L

X = np.array(X)
y = np.array(y)
print("X shape:", X.shape, "y shape:", y.shape)

# ====== 2. Split train-test ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y
)

# ====== 3. Build LSTM model ======
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))  # 3 kelas: neutral, A, L

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ====== 4. Train ======
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ====== 5. Save model + label ======
model_dict = {
    'model': model,
    'labels': {0: 'neutral', 1: 'A', 2: 'L'}
}
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model_dict, f)

print("✅ Gesture LSTM model saved to gesture_model.pkl")
