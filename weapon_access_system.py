import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import time
import json
import csv
from datetime import datetime

import numpy as np
import mediapipe as mp
import face_recognition
import pickle
from tensorflow.keras.models import load_model

# ================== CONFIG ==================

CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720

ACCESS_RECORDS_FILE = "weapon_access_records.csv"
SECRET_GESTURE_SEQUENCE = ["A", "L"]

# ===== GESTURE MODEL (LSTM) =====
GESTURE_MODEL_PATH = './gesture_model.h5'
GESTURE_LABELS_PATH = './gesture_labels.pkl'
gesture_model = None
gesture_labels = {}
NO_OF_TIMESTEPS = 20

# ===== FACE RECOGNITION =====
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []
known_soldier_data = {}

# ================== MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)
mp_draw = mp.solutions.drawing_utils

# ================== HELPER FUNCTIONS ==================
def draw_text_with_outline(img, text, pos, font, scale, color, thickness):
    """Draw text with black outline for better visibility"""
    x, y = pos
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_animated_popup(frame, title, subtitle, progress=1.0, status="success"):
    """Draw modern animated popup with progress bar"""
    h, w = frame.shape[:2]
    
    box_w = 800
    box_h = 300
    box_x = (w - box_w) // 2
    box_y = (h - box_h) // 2
    
    if status == "success":
        box_color = (0, 180, 0)
        title_color = (255, 255, 255)
        border_color = (0, 255, 0)
    elif status == "detected":
        box_color = (0, 140, 255)
        title_color = (255, 255, 255)
        border_color = (0, 200, 255)
    elif status == "failed":
        box_color = (0, 0, 200)
        title_color = (255, 255, 255)
        border_color = (0, 100, 255)
    else:
        box_color = (60, 60, 60)
        title_color = (255, 255, 255)
        border_color = (150, 150, 150)
    
    # Shadow
    shadow_offset = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (box_x + shadow_offset, box_y + shadow_offset), 
                  (box_x + box_w + shadow_offset, box_y + box_h + shadow_offset),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Main box
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Border
    cv2.rectangle(frame, (box_x-2, box_y-2), (box_x + box_w+2, box_y + box_h+2), border_color, 6)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
    
    # Icon
    icon_y = box_y + 60
    icon_x = box_x + box_w // 2
    if status == "success":
        cv2.circle(frame, (icon_x, icon_y), 35, (255, 255, 255), 4)
        cv2.line(frame, (icon_x - 15, icon_y), (icon_x - 5, icon_y + 15), (255, 255, 255), 4)
        cv2.line(frame, (icon_x - 5, icon_y + 15), (icon_x + 15, icon_y - 10), (255, 255, 255), 4)
    elif status == "detected":
        cv2.circle(frame, (icon_x, icon_y), 35, (255, 255, 255), 4)
        cv2.line(frame, (icon_x, icon_y - 15), (icon_x, icon_y + 5), (255, 255, 255), 5)
        cv2.circle(frame, (icon_x, icon_y + 15), 3, (255, 255, 255), -1)
    elif status == "failed":
        cv2.circle(frame, (icon_x, icon_y), 35, (255, 255, 255), 4)
        cv2.line(frame, (icon_x - 12, icon_y - 12), (icon_x + 12, icon_y + 12), (255, 255, 255), 5)
        cv2.line(frame, (icon_x + 12, icon_y - 12), (icon_x - 12, icon_y + 12), (255, 255, 255), 5)
    
    # Title
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0]
    title_x = box_x + (box_w - title_size[0]) // 2
    title_y = box_y + 140
    cv2.putText(frame, title, (title_x, title_y), 
                cv2.FONT_HERSHEY_DUPLEX, 1.8, title_color, 3, cv2.LINE_AA)
    
    # Subtitle
    subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
    subtitle_x = box_x + (box_w - subtitle_size[0]) // 2
    subtitle_y = box_y + 190
    cv2.putText(frame, subtitle, (subtitle_x, subtitle_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Progress bar
    if progress < 1.0:
        bar_y = box_y + box_h - 40
        bar_x = box_x + 50
        bar_w = box_w - 100
        bar_h = 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        progress_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), border_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

# ================== WEAPON ACCESS SYSTEM ==================
class WeaponAccessSystem:
    def __init__(self):
        self.current_user = None
        self.gesture_sequence = []
        self.gesture_timestamps = []
        self.face_recognition_active = False
        self.face_recognition_timeout = 0
        self.face_recognition_duration = 8
        self.screenshot_dir = "access_screenshots"
        
        self.last_detected_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5
        self.max_sequence_time = 5.0
        
        self.access_granted_time = 0
        self.show_access_granted = False
        self.sequence_detected_time = 0
        self.show_sequence_alert = False
        self.sequence_failed_time = 0
        self.show_sequence_failed = False
        self.face_failed_time = 0
        self.show_face_failed = False
        self.setup_directories()

    def setup_directories(self):
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
        if not os.path.exists(ACCESS_RECORDS_FILE):
            with open(ACCESS_RECORDS_FILE, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Tanggal", "Waktu", "Nama", "NRP", "Pangkat",
                    "Kesatuan", "Gesture Sequence", "Status", "Screenshot Path"
                ])

    def record_access(self, soldier_data):
        current_datetime = datetime.now()
        date_str = current_datetime.strftime("%Y-%m-%d")
        time_str = current_datetime.strftime("%H:%M:%S")
        screenshot_path = self.take_screenshot(soldier_data['name'], date_str, time_str)

        access_data = [
            date_str,
            time_str,
            soldier_data['name'],
            soldier_data.get('nrp', 'N/A'),
            soldier_data.get('rank', 'N/A'),
            soldier_data.get('unit', 'N/A'),
            " -> ".join(SECRET_GESTURE_SEQUENCE),
            "GRANTED",
            screenshot_path
        ]
        with open(ACCESS_RECORDS_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(access_data)

        print(f"✅ Akses: {soldier_data['name']}")
        self.show_access_granted = True
        self.access_granted_time = time.time()
        return True

    def take_screenshot(self, name, date_str, time_str):
        filename = f"access_{name}_{date_str}_{time_str.replace(':', '')}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        return filepath

    def process_gesture(self, gesture, confidence, current_time):
        """REALTIME gesture processing"""
        if gesture == self.last_detected_gesture:
            return False
        
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return False
        
        if gesture and gesture != "neutral" and confidence > 0.5:
            self.gesture_sequence.append(gesture)
            self.gesture_timestamps.append(current_time)
            self.last_detected_gesture = gesture
            self.last_gesture_time = current_time
            
            if len(self.gesture_sequence) > 5:
                self.gesture_sequence.pop(0)
                self.gesture_timestamps.pop(0)
            
            print(f"🔹 {gesture} [{confidence:.2f}]")
            return True
        return False

    def check_secret_gesture_sequence(self):
        """Check sequence dengan validasi waktu"""
        if len(self.gesture_sequence) < len(SECRET_GESTURE_SEQUENCE):
            return False
        
        recent = self.gesture_sequence[-len(SECRET_GESTURE_SEQUENCE):]
        recent_times = self.gesture_timestamps[-len(SECRET_GESTURE_SEQUENCE):]
        
        if recent == SECRET_GESTURE_SEQUENCE:
            time_diff = recent_times[-1] - recent_times[0]
            
            if time_diff <= self.max_sequence_time:
                print(f"✅ Sandi OK! ({time_diff:.2f}s)")
                return True
            else:
                print(f"❌ Terlalu lambat: {time_diff:.2f}s")
                self.show_sequence_failed = True
                self.sequence_failed_time = time.time()
                self.gesture_sequence.clear()
                self.gesture_timestamps.clear()
                return False
        return False

    def activate_face_recognition(self):
        self.face_recognition_active = True
        self.face_recognition_timeout = time.time() + self.face_recognition_duration
        self.show_sequence_alert = True
        self.sequence_detected_time = time.time()
        print(f"⚠️  SANDI TERDETEKSI!\n")

    def update_face_recognition_status(self):
        if self.face_recognition_active and time.time() > self.face_recognition_timeout:
            self.face_recognition_active = False
            self.current_user = None
            # Tampilkan popup gagal
            self.show_face_failed = True
            self.face_failed_time = time.time()
            print("❌ Wajah tidak dikenali - Timeout")
    
    def reset(self):
        self.gesture_sequence.clear()
        self.gesture_timestamps.clear()
        self.last_detected_gesture = None
        self.last_gesture_time = 0

# ================== FACE RECOGNITION ==================
def load_soldier_data():
    soldier_data_file = "soldier_data.json"
    if os.path.exists(soldier_data_file):
        with open(soldier_data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_known_faces():
    global known_face_encodings, known_face_names, known_soldier_data
    known_soldier_data = load_soldier_data()
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[WARN] {KNOWN_FACES_DIR} not found")
        return
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)
    print(f"[FACE] {len(known_face_names)} loaded")

def recognize_face(frame):
    if len(known_face_encodings) == 0:
        return None, 0.0
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]
        if min_distance < 0.5:
            name = known_face_names[best_match_index]
            confidence = (1 - min_distance) * 100
            return name, confidence
    return None, 0.0

def get_soldier_data(name):
    return known_soldier_data.get(name, {
        'name': name,
        'nrp': 'N/A',
        'rank': 'N/A',
        'unit': 'N/A'
    })

# ================== GESTURE MODEL ==================
def load_gesture_model():
    global gesture_model, gesture_labels
    try:
        if os.path.exists(GESTURE_MODEL_PATH):
            gesture_model = load_model(GESTURE_MODEL_PATH)
            if os.path.exists(GESTURE_LABELS_PATH):
                with open(GESTURE_LABELS_PATH, 'rb') as f:
                    gesture_labels.clear()
                    gesture_labels.update(pickle.load(f))
            print("[GESTURE] Model OK")
            return True
        else:
            print("[ERROR] gesture_model.h5 not found")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

# ================== MAIN ==================
def main():
    if not load_gesture_model():
        return

    load_known_faces()
    access_system = WeaponAccessSystem()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Camera failed")
        return

    print("\n" + "="*70)
    print("SISTEM KEAMANAN AKSES GUDANG SENJATA - TNI")
    print("="*70)
    print("SEQUENTIAL BIOMETRIC FUSION SYSTEM")
    print("-" * 70)
    print("1. MODE SIAGA: Monitoring gesture (Face OFF)")
    print(f"2. TRIGGER: Sandi '{' → '.join(SECRET_GESTURE_SEQUENCE)}' < 5s")
    print("3. AKTIVASI: Face Recognition ON (8s)")
    print("4. VERIFIKASI: Validasi identitas")
    print("5. LOGGING: Auto-capture + CSV")
    print("="*70)
    print("Q=Quit | R=Reset | F=Fullscreen")
    print("="*70 + "\n")

    window_name = 'SISTEM KEAMANAN AKSES GUDANG SENJATA - TNI'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    current_frame = None
    gesture_seq_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame.copy()
        frame = cv2.flip(frame, 1)
        current_time = time.time()

        access_system.update_face_recognition_status()

        # === GESTURE ===
        current_gesture = None
        gesture_confidence = 0.0
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = hands.process(rgb_frame)

        if hand_res.multi_hand_landmarks:
            hlm = hand_res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

            feat = []
            for lm in hlm.landmark:
                feat.extend([lm.x, lm.y, lm.z])

            if sum(feat) != 0:
                gesture_seq_buffer.append(feat)
                if len(gesture_seq_buffer) > NO_OF_TIMESTEPS:
                    gesture_seq_buffer.pop(0)

            if len(gesture_seq_buffer) == NO_OF_TIMESTEPS:
                X = np.array(gesture_seq_buffer).reshape(1, NO_OF_TIMESTEPS, -1)
                prob = gesture_model.predict(X, verbose=0)[0]
                cls = int(np.argmax(prob))
                confidence = float(np.max(prob))
                
                if confidence > 0.5:
                    gesture = gesture_labels.get(cls, "Unknown")
                    current_gesture = gesture
                    gesture_confidence = confidence
                    access_system.process_gesture(gesture, confidence, current_time)

        # === CHECK SECRET ===
        if access_system.check_secret_gesture_sequence():
            access_system.activate_face_recognition()
            access_system.gesture_sequence.clear()
            access_system.gesture_timestamps.clear()
            gesture_seq_buffer.clear()

        # === FACE ===
        face_name = None
        face_confidence = 0.0

        if access_system.face_recognition_active:
            face_name, face_confidence = recognize_face(frame)
            if face_name and face_confidence > 60:
                access_system.current_user = face_name
                soldier_data = get_soldier_data(face_name)
                
                if access_system.record_access(soldier_data):
                    screenshot_path = access_system.take_screenshot(
                        face_name,
                        datetime.now().strftime("%Y-%m-%d"),
                        datetime.now().strftime("%H:%M:%S")
                    )
                    cv2.imwrite(screenshot_path, current_frame)
                    access_system.face_recognition_active = False

        # === DISPLAY ===
        h, w = frame.shape[:2]
        
        # POPUP: Failed (prioritas tertinggi)
        if access_system.show_sequence_failed and (current_time - access_system.sequence_failed_time) < 2.5:
            elapsed = current_time - access_system.sequence_failed_time
            progress = elapsed / 2.5
            draw_animated_popup(frame, "SANDI GAGAL!", 
                              "Terlalu lambat! Max 5 detik",
                              progress, "failed")
        elif access_system.show_sequence_failed:
            access_system.show_sequence_failed = False
        
        # POPUP: Face Failed (wajah tidak dikenali)
        elif access_system.show_face_failed and (current_time - access_system.face_failed_time) < 2.5:
            elapsed = current_time - access_system.face_failed_time
            progress = elapsed / 2.5
            draw_animated_popup(frame, "WAJAH TIDAK DIKENALI!", 
                              "Kembali ke mode scan gesture",
                              progress, "failed")
        elif access_system.show_face_failed:
            access_system.show_face_failed = False
        
        # POPUP: Detected (harus selesai dulu)
        elif access_system.show_sequence_alert and (current_time - access_system.sequence_detected_time) < 2.5:
            elapsed = current_time - access_system.sequence_detected_time
            progress = elapsed / 2.5
            draw_animated_popup(frame, "SANDI TERDETEKSI!", 
                              "Face Recognition Aktif",
                              progress, "detected")
        elif access_system.show_sequence_alert:
            access_system.show_sequence_alert = False
        
        # POPUP: Success (muncul setelah orange selesai)
        elif access_system.show_access_granted and (current_time - access_system.access_granted_time) < 3.0:
            elapsed = current_time - access_system.access_granted_time
            progress = elapsed / 3.0
            draw_animated_popup(frame, "AKSES DIIZINKAN!", 
                              f"Selamat datang, {access_system.current_user}",
                              progress, "success")
        elif access_system.show_access_granted:
            access_system.show_access_granted = False
        
        # Header
        draw_text_with_outline(frame, "SISTEM KEAMANAN AKSES SENJATA - TNI",
                    (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

        # Gesture
        if current_gesture and current_gesture != "neutral":
            gesture_text = f"Gesture: {current_gesture}"
            draw_text_with_outline(frame, gesture_text,
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Confidence bar
            bar_x, bar_y = 30, 120
            bar_w, bar_h = 300, 15
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            conf_w = int(bar_w * gesture_confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_w, bar_y + bar_h), (0, 255, 0), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
            
            conf_text = f"{gesture_confidence:.1%}"
            cv2.putText(frame, conf_text, (bar_x + bar_w + 10, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            draw_text_with_outline(frame, "Gesture: Waiting...",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

        # Face
        status_y = 180
        if access_system.face_recognition_active:
            time_left = int(access_system.face_recognition_timeout - current_time)
            draw_text_with_outline(frame, f"Face Recognition: AKTIF ({time_left}s)",
                        (30, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if face_name:
                soldier_data = get_soldier_data(face_name)
                draw_text_with_outline(frame, f"Prajurit: {face_name}",
                            (30, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                draw_text_with_outline(frame, f"NRP: {soldier_data.get('nrp', 'N/A')}",
                            (30, status_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                draw_text_with_outline(frame, "Tampilkan wajah",
                            (30, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Datetime
        draw_text_with_outline(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (w - 350, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if current_state == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord('r'):
            access_system.reset()
            gesture_seq_buffer.clear()
            print("🔄 Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()