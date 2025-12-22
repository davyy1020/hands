import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mpHands = mp.solutions.hands
mpDraw  = mp.solutions.drawing_utils

labels = ['neutral', 'A', 'L']
no_of_frames = 500  # 500 frame per gesture (cukup untuk akurasi)

def make_landmark_timestamp(results):
    """21 titik × (x,y,z) = 63 fitur"""
    cLm = []
    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        for lm in hand_lms.landmark:
            cLm.extend([lm.x, lm.y, lm.z])
    return cLm + [0] * (63 - len(cLm))

print("="*60)
print("🎯 GESTURE DATA COLLECTION - TRAINING ULANG")
print("="*60)
print("INSTRUKSI:")
print("1. NEUTRAL: Tangan rileks, telapak terbuka menghadap kamera")
print("2. A: Pose seperti di screenshot (jari telunjuk + ibu jari bentuk L terbalik)")
print("3. L: Pose L dengan ibu jari ke atas, telunjuk horizontal")
print("")
print("TIPS:")
print("- Pastikan pencahayaan bagus")
print("- Tangan harus jelas terlihat")
print("- KONSISTEN: jangan ubah posisi tangan saat recording")
print("- SPACE = mulai gesture berikutnya | Q = quit")
print("="*60 + "\n")

CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print(f"❌ CAMERA {CAMERA_INDEX} GAGAL")
    exit()

print(f"✅ Camera {CAMERA_INDEX} ready\n")

for label in labels:
    lmlist = []
    print(f"\n{'='*60}")
    print(f"📹 SEKARANG: Gesture '{label.upper()}' ({no_of_frames} frames)")
    print(f"{'='*60}")
    
    if label == 'neutral':
        print("👉 Tangan rileks, telapak terbuka menghadap kamera")
    elif label == 'A':
        print("👉 Pose A: telunjuk + ibu jari bentuk L terbalik (lihat contoh)")
    elif label == 'L':
        print("👉 Pose L: ibu jari ke atas, telunjuk ke samping")
    
    print("\n⏳ Tekan SPACE untuk mulai recording...")
    
    # Wait for SPACE
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f"Gesture: {label.upper()}", 
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, "Tekan SPACE untuk mulai", 
                    (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Gesture Data Collection", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            print(f"🎬 RECORDING dimulai untuk '{label}'...\n")
            break
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    # Start recording
    with mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        while len(lmlist) < no_of_frames:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frameRGB)

            if results.multi_hand_landmarks:
                mpDraw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mpHands.HAND_CONNECTIONS
                )
                lm = make_landmark_timestamp(results)
                lmlist.append(lm)
                
                progress = len(lmlist) / no_of_frames * 100
                
                # Progress bar
                bar_length = 400
                filled_length = int(bar_length * len(lmlist) / no_of_frames)
                cv2.rectangle(frame, (20, 80), (20 + bar_length, 110), (50, 50, 50), -1)
                cv2.rectangle(frame, (20, 80), (20 + filled_length, 110), (0, 255, 0), -1)
                
                cv2.putText(
                    frame,
                    f"{label.upper()}: {len(lmlist)}/{no_of_frames} ({progress:.1f}%)",
                    (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "TANGAN TIDAK TERDETEKSI!",
                    (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frame,
                f"Gesture: {label.upper()}",
                (20, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                "JANGAN UBAH POSISI! | Q=Quit",
                (20, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("Gesture Data Collection", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    df = pd.DataFrame(lmlist)
    df.to_csv(f"{label}.txt", index=False)
    print(f"\n✅ {label}.txt saved ({len(lmlist)} frames)")
    print(f"   Lokasi: {os.path.abspath(label + '.txt')}")
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("🎉 SEMUA DATA BERHASIL DIKUMPULKAN!")
print("="*60)
print("File yang dibuat:")
print("  - neutral.txt")
print("  - A.txt")
print("  - L.txt")
print("\n📌 Langkah berikutnya: Jalankan train_lstm.py")
print("="*60)
