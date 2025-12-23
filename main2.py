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
from collections import deque

# ================== CONFIG ==================
CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720

ACCESS_RECORDS_FILE = "weapon_access_records.csv"
SOLDIER_DATA_FILE = "soldier_data.json"
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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ================== BUTTON CLASS ==================
class Button:
    def __init__(self, x, y, w, h, text, color=(100, 100, 100), text_color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover_color = tuple(min(c + 40, 255) for c in color)
        self.is_hovered = False
        
    def draw(self, frame):
        color = self.hover_color if self.is_hovered else self.color
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 2)
        
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)
    
    def is_clicked(self, mouse_x, mouse_y):
        return self.x <= mouse_x <= self.x + self.w and self.y <= mouse_y <= self.y + self.h
    
    def check_hover(self, mouse_x, mouse_y):
        self.is_hovered = self.is_clicked(mouse_x, mouse_y)

# ================== HELPER FUNCTIONS ==================
def draw_text_with_outline(img, text, pos, font, scale, color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_animated_popup(frame, title, subtitle, progress=1.0, status="success", soldier_info=None):
    """Draw modern animated popup with progress bar and optional soldier info"""
    h, w = frame.shape[:2]
    
    # Adjust box size if showing detailed info
    if soldier_info and status == "success":
        box_w = 900
        box_h = 450
    else:
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
    
    overlay = frame.copy()
    shadow_offset = 10
    cv2.rectangle(overlay, 
                  (box_x + shadow_offset, box_y + shadow_offset), 
                  (box_x + box_w + shadow_offset, box_y + box_h + shadow_offset),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    cv2.rectangle(frame, (box_x-2, box_y-2), (box_x + box_w+2, box_y + box_h+2), border_color, 6)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
    
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
    
    # ===== SOLDIER DETAILED INFO =====
    if soldier_info and status == "success":
        info_y_start = box_y + 230
        line_height = 35
        label_x = box_x + 80
        value_x = box_x + 280
        
        # Draw separator line
        cv2.line(frame, (box_x + 50, info_y_start - 10), 
                (box_x + box_w - 50, info_y_start - 10), (255, 255, 255), 2)
        
        # Nama Lengkap
        cv2.putText(frame, "Nama", (label_x, info_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, f": {soldier_info['name']}", (value_x, info_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # NRP
        cv2.putText(frame, "NRP", (label_x, info_y_start + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, f": {soldier_info['nrp']}", (value_x, info_y_start + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Pangkat
        cv2.putText(frame, "Pangkat", (label_x, info_y_start + line_height * 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, f": {soldier_info['rank']}", (value_x, info_y_start + line_height * 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Kesatuan
        cv2.putText(frame, "Kesatuan", (label_x, info_y_start + line_height * 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, f": {soldier_info['unit']}", (value_x, info_y_start + line_height * 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Tanggal & Waktu
        cv2.line(frame, (box_x + 50, info_y_start + line_height * 3 + 20), 
                (box_x + box_w - 50, info_y_start + line_height * 3 + 20), (255, 255, 255), 1)
        
        datetime_y = info_y_start + line_height * 4 + 10
        datetime_text = soldier_info['datetime']
        datetime_size = cv2.getTextSize(datetime_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
        datetime_x = box_x + (box_w - datetime_size[0]) // 2
        cv2.putText(frame, datetime_text, (datetime_x, datetime_y), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
    
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

# ================== REGISTRATION SYSTEM ==================
class RegistrationSystem:
    def __init__(self):
        self.is_registering = False
        self.capture_countdown = 0
        self.captured_frame = None
        self.registration_data = {'full_name': '', 'nrp': '', 'rank': '', 'unit': ''}
        self.fields = ['full_name', 'nrp', 'rank', 'unit']
        self.field_index = 0
        self.show_registration_success = False
        self.registration_success_time = 0
        self.show_registration_failed = False
        self.registration_failed_time = 0
        self.failed_message = ""
        self.photo_preview = None
        self.buttons = {}
        self.keyboard_buttons = []
        
    def start_registration(self):
        self.is_registering = True
        self.capture_countdown = 0
        self.captured_frame = None
        self.photo_preview = None
        self.registration_data = {'full_name': '', 'nrp': '', 'rank': '', 'unit': ''}
        self.field_index = 0
        self.setup_ui()
        print("\n" + "="*50)
        print("MODE PENDAFTARAN ANGGOTA BARU")
        print("="*50)
    
    def setup_ui(self):
        self.buttons['up'] = Button(20, 500, 60, 50, "UP", (70, 70, 70))
        self.buttons['down'] = Button(90, 500, 60, 50, "DN", (70, 70, 70))
        self.buttons['delete'] = Button(160, 500, 80, 50, "DEL", (150, 50, 50))
        self.buttons['capture'] = Button(20, 560, 220, 50, "AMBIL FOTO", (0, 120, 200))
        self.buttons['submit'] = Button(20, 620, 220, 50, "KIRIM", (0, 150, 0))
        self.buttons['cancel'] = Button(20, 680, 220, 50, "BATAL", (150, 50, 50))
        
        keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ' ']
        ]
        
        self.keyboard_buttons = []
        start_x = 260
        start_y = 500
        key_w = 40
        key_h = 40
        spacing = 5
        
        for row_idx, row in enumerate(keys):
            row_buttons = []
            x_offset = start_x + (row_idx * 20)
            for col_idx, key in enumerate(row):
                w = 100 if key == ' ' else key_w
                btn = Button(
                    x_offset + col_idx * (key_w + spacing),
                    start_y + row_idx * (key_h + spacing),
                    w, key_h,
                    'SPACE' if key == ' ' else key,
                    (60, 60, 60)
                )
                row_buttons.append(btn)
            self.keyboard_buttons.append(row_buttons)
    
    @property
    def current_field(self):
        return self.fields[self.field_index]
    
    def handle_button_click(self, x, y):
        if self.buttons['up'].is_clicked(x, y):
            if self.field_index > 0:
                self.field_index -= 1
        elif self.buttons['down'].is_clicked(x, y):
            if self.field_index < len(self.fields) - 1:
                self.field_index += 1
        elif self.buttons['delete'].is_clicked(x, y):
            if self.registration_data[self.current_field]:
                self.registration_data[self.current_field] = self.registration_data[self.current_field][:-1]
        elif self.buttons['capture'].is_clicked(x, y):
            if self.photo_preview is None:
                self.capture_countdown = 3
                print("📸 Countdown dimulai...")
        elif self.buttons['submit'].is_clicked(x, y):
            if self.photo_preview is not None:
                return 'submit'
        elif self.buttons['cancel'].is_clicked(x, y):
            self.is_registering = False
            return 'cancel'
        
        for row in self.keyboard_buttons:
            for btn in row:
                if btn.is_clicked(x, y):
                    char = ' ' if btn.text == 'SPACE' else btn.text
                    self.registration_data[self.current_field] += char
        
        return None
    
    def update_hover(self, x, y):
        for btn in self.buttons.values():
            btn.check_hover(x, y)
        for row in self.keyboard_buttons:
            for btn in row:
                btn.check_hover(x, y)
    
    def save_registration(self):
        full_name = self.registration_data['full_name'].strip()
        nrp = self.registration_data['nrp'].strip()
        rank = self.registration_data['rank'].strip()
        unit = self.registration_data['unit'].strip()
        
        if not full_name:
            self.failed_message = "Nama Lengkap harus diisi!"
            self.show_registration_failed = True
            self.registration_failed_time = time.time()
            print(f"❌ {self.failed_message}")
            return False
        
        if self.captured_frame is None:
            self.failed_message = "Foto belum diambil!"
            self.show_registration_failed = True
            self.registration_failed_time = time.time()
            print(f"❌ {self.failed_message}")
            return False
        
        name_id = full_name.lower().replace(" ", "_")
        
        if os.path.exists(SOLDIER_DATA_FILE):
            with open(SOLDIER_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        if name_id in data:
            self.failed_message = f"Nama '{name_id}' sudah terdaftar!"
            self.show_registration_failed = True
            self.registration_failed_time = time.time()
            print(f"❌ {self.failed_message}")
            return False
        
        data[name_id] = {
            'name': full_name,
            'nrp': nrp or 'N/A',
            'rank': rank or 'N/A',
            'unit': unit or 'N/A'
        }
        
        with open(SOLDIER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        person_dir = os.path.join(KNOWN_FACES_DIR, name_id)
        os.makedirs(person_dir, exist_ok=True)
        photo_path = os.path.join(person_dir, f"{name_id}.jpg")
        cv2.imwrite(photo_path, self.captured_frame)
        
        print(f"✅ Pendaftaran berhasil: {full_name}")
        print(f"📁 ID: {name_id}")
        print(f"📁 Foto disimpan: {photo_path}")
        
        self.show_registration_success = True
        self.registration_success_time = time.time()
        return True

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
        self.gesture_cooldown = 0.15
        self.max_sequence_time = 5.0
        
        self.gesture_vote_window = deque(maxlen=3)
        
        self.access_granted_time = 0
        self.show_access_granted = False
        self.sequence_detected_time = 0
        self.show_sequence_alert = False
        self.sequence_failed_time = 0
        self.show_sequence_failed = False
        self.face_failed_time = 0
        self.show_face_failed = False
        
        # Store soldier info for detailed popup
        self.granted_soldier_info = None
        
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
        """REALTIME: Voting system untuk stabilitas tanpa delay"""
        self.gesture_vote_window.append(gesture)
        
        if gesture == self.last_detected_gesture:
            return False
        
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return False
        
        if len(self.gesture_vote_window) >= 2:
            vote_count = self.gesture_vote_window.count(gesture)
            if vote_count < 2:
                return False
        
        if gesture and gesture != "neutral" and confidence > 0.70:
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
            self.show_face_failed = True
            self.face_failed_time = time.time()
            print("❌ Wajah tidak dikenali - Timeout")
    
    def reset(self):
        self.gesture_sequence.clear()
        self.gesture_timestamps.clear()
        self.last_detected_gesture = None
        self.last_gesture_time = 0
        self.gesture_vote_window.clear()

# ================== FACE RECOGNITION ==================
def load_soldier_data():
    if os.path.exists(SOLDIER_DATA_FILE):
        with open(SOLDIER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_known_faces():
    global known_face_encodings, known_face_names, known_soldier_data
    known_soldier_data = load_soldier_data()
    known_face_encodings.clear()
    known_face_names.clear()
    
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

# ================== MOUSE CALLBACK ==================
mouse_x, mouse_y = 0, 0
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

# ================== MAIN ==================
def main():
    global mouse_clicked
    
    if not load_gesture_model():
        return

    load_known_faces()
    access_system = WeaponAccessSystem()
    registration_system = RegistrationSystem()

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
    print("Klik tombol untuk memilih mode")
    print("="*70 + "\n")

    window_name = 'SISTEM KEAMANAN AKSES GUDANG SENJATA - TNI'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    mode_btn_absen = Button(FRAME_W // 2 - 250, FRAME_H // 2 - 50, 200, 100, "MODE ABSEN", (0, 120, 0))
    mode_btn_daftar = Button(FRAME_W // 2 + 50, FRAME_H // 2 - 50, 200, 100, "MODE DAFTAR", (0, 120, 200))

    current_frame = None
    gesture_seq_buffer = deque(maxlen=NO_OF_TIMESTEPS)
    mode = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame.copy()
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        h, w = frame.shape[:2]

        # ===== MODE SELECTION =====
        if mode is None:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            draw_text_with_outline(frame, "PILIH MODE OPERASI",
                        (w//2 - 250, h//2 - 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
            
            mode_btn_absen.draw(frame)
            mode_btn_daftar.draw(frame)
            mode_btn_absen.check_hover(mouse_x, mouse_y)
            mode_btn_daftar.check_hover(mouse_x, mouse_y)
            
            if mouse_clicked:
                if mode_btn_absen.is_clicked(mouse_x, mouse_y):
                    mode = "attendance"
                    print("\n📋 Mode: ABSEN")
                elif mode_btn_daftar.is_clicked(mouse_x, mouse_y):
                    mode = "registration"
                    registration_system.start_registration()
                    print("\n📝 Mode: PENDAFTARAN")
                mouse_clicked = False
            
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # ===== MODE REGISTRATION =====
        if registration_system.is_registering:
            form_width = int(w * 0.2)
            camera_width = w - form_width
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (form_width, h), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
            cv2.line(frame, (form_width, 0), (form_width, h), (0, 255, 255), 3)
            
            cv2.putText(frame, "PENDAFTARAN", (20, 40), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
            
            field_labels = {
                'full_name': 'Nama Lengkap',
                'nrp': 'NRP',
                'rank': 'Pangkat',
                'unit': 'Kesatuan'
            }
            
            y_start = 80
            for idx, field in enumerate(registration_system.fields):
                y = y_start + (idx * 90)
                is_active = (registration_system.field_index == idx)
                color = (0, 255, 0) if is_active else (200, 200, 200)
                
                cv2.putText(frame, field_labels[field], (20, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                box_x = 20
                box_y = y + 5
                box_w = form_width - 40
                box_h = 30
                
                if is_active:
                    cv2.rectangle(frame, (box_x-2, box_y-2), (box_x + box_w+2, box_y + box_h+2), (0, 255, 255), 2)
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 1)
                
                text = registration_system.registration_data[field]
                if len(text) > 15:
                    text = "..." + text[-12:]
                cv2.putText(frame, text, (box_x + 5, box_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            registration_system.update_hover(mouse_x, mouse_y)
            for btn in registration_system.buttons.values():
                btn.draw(frame)
            
            for row in registration_system.keyboard_buttons:
                for btn in row:
                    btn.draw(frame)
            
            if registration_system.photo_preview is not None:
                preview = registration_system.photo_preview.copy()
                preview = cv2.flip(preview, 1)
                preview_h, preview_w = preview.shape[:2]
                scale = min(camera_width / preview_w, h / preview_h) * 0.9
                new_w = int(preview_w * scale)
                new_h = int(preview_h * scale)
                preview_resized = cv2.resize(preview, (new_w, new_h))
                
                y_offset = (h - new_h) // 2
                x_offset = form_width + (camera_width - new_w) // 2
                
                frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = preview_resized
                cv2.rectangle(frame, (x_offset-3, y_offset-3), 
                            (x_offset+new_w+3, y_offset+new_h+3), (0, 255, 0), 3)
                
                cv2.putText(frame, "Foto berhasil diambil!", 
                           (form_width + 50, 50), 
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            
            if registration_system.capture_countdown > 0:
                countdown_text = str(registration_system.capture_countdown)
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_DUPLEX, 6, 6)[0]
                text_x = form_width + (camera_width - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                
                cv2.circle(frame, (text_x + text_size[0]//2, text_y - text_size[1]//2), 
                          150, (0, 255, 255), 8)
                cv2.putText(frame, countdown_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_DUPLEX, 6, (0, 255, 255), 6)
                
                time.sleep(1)
                registration_system.capture_countdown -= 1
                
                if registration_system.capture_countdown == 0:
                    registration_system.captured_frame = current_frame.copy()
                    registration_system.photo_preview = current_frame.copy()
                    print("✅ Foto berhasil diambil!")
            
            if mouse_clicked:
                result = registration_system.handle_button_click(mouse_x, mouse_y)
                if result == 'submit':
                    if not registration_system.show_registration_success and \
                       not registration_system.show_registration_failed:
                        registration_system.save_registration()
                elif result == 'cancel':
                    mode = None
                    print("❌ Pendaftaran dibatalkan")
                mouse_clicked = False
            
            if registration_system.show_registration_success and \
               (current_time - registration_system.registration_success_time) < 5.0:
                elapsed = current_time - registration_system.registration_success_time
                progress = elapsed / 5.0
                draw_animated_popup(frame, "PENDAFTARAN BERHASIL!", 
                                  "Data tersimpan di database",
                                  progress, "success")
            elif registration_system.show_registration_success:
                registration_system.show_registration_success = False
                registration_system.is_registering = False
                load_known_faces()
                mode = None
            
            if registration_system.show_registration_failed and \
               (current_time - registration_system.registration_failed_time) < 4.0:
                elapsed = current_time - registration_system.registration_failed_time
                progress = elapsed / 4.0
                draw_animated_popup(frame, "PENDAFTARAN GAGAL!", 
                                  registration_system.failed_message,
                                  progress, "failed")
            elif registration_system.show_registration_failed:
                registration_system.show_registration_failed = False
            
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue
        
        # ===== MODE ATTENDANCE =====
        access_system.update_face_recognition_status()

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

            if len(gesture_seq_buffer) == NO_OF_TIMESTEPS:
                X = np.array(gesture_seq_buffer).reshape(1, NO_OF_TIMESTEPS, -1)
                prob = gesture_model.predict(X, verbose=0)[0]
                cls = int(np.argmax(prob))
                confidence = float(np.max(prob))
                
                if confidence > 0.65:
                    gesture = gesture_labels.get(cls, "Unknown")
                    current_gesture = gesture
                    gesture_confidence = confidence
                    access_system.process_gesture(gesture, confidence, current_time)

        if access_system.check_secret_gesture_sequence():
            access_system.activate_face_recognition()
            access_system.gesture_sequence.clear()
            access_system.gesture_timestamps.clear()
            gesture_seq_buffer.clear()

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
                    
                    # Store soldier info for detailed popup
                    access_system.granted_soldier_info = {
                        'name': soldier_data['name'],
                        'nrp': soldier_data.get('nrp', 'N/A'),
                        'rank': soldier_data.get('rank', 'N/A'),
                        'unit': soldier_data.get('unit', 'N/A'),
                        'datetime': datetime.now().strftime("%d %B %Y, %H:%M:%S WIB")
                    }
                    
                    access_system.face_recognition_active = False

        # Display popups
        if access_system.show_sequence_failed and (current_time - access_system.sequence_failed_time) < 2.5:
            elapsed = current_time - access_system.sequence_failed_time
            progress = elapsed / 2.5
            draw_animated_popup(frame, "SANDI GAGAL!", 
                              "Terlalu lambat! Max 5 detik",
                              progress, "failed")
        elif access_system.show_sequence_failed:
            access_system.show_sequence_failed = False
        
        elif access_system.show_face_failed and (current_time - access_system.face_failed_time) < 2.5:
            elapsed = current_time - access_system.face_failed_time
            progress = elapsed / 2.5
            draw_animated_popup(frame, "WAJAH TIDAK DIKENALI!", 
                              "Kembali ke mode scan gesture",
                              progress, "failed")
        elif access_system.show_face_failed:
            access_system.show_face_failed = False
        
        elif access_system.show_sequence_alert and (current_time - access_system.sequence_detected_time) < 2.5:
            elapsed = current_time - access_system.sequence_detected_time
            progress = elapsed / 2.5
            draw_animated_popup(frame, "SANDI TERDETEKSI!", 
                              "Face Recognition Aktif",
                              progress, "detected")
        elif access_system.show_sequence_alert:
            access_system.show_sequence_alert = False
        
        elif access_system.show_access_granted and (current_time - access_system.access_granted_time) < 4.0:
            elapsed = current_time - access_system.access_granted_time
            progress = elapsed / 4.0
            draw_animated_popup(frame, "AKSES DIIZINKAN!", 
                              f"Selamat datang, {access_system.granted_soldier_info['name']}",
                              progress, "success", 
                              soldier_info=access_system.granted_soldier_info)
        elif access_system.show_access_granted:
            access_system.show_access_granted = False
        
        # UI
        draw_text_with_outline(frame, "MODE: ABSEN",
                    (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        
        back_btn = Button(w - 220, 20, 200, 60, "KEMBALI", (100, 100, 100))
        back_btn.check_hover(mouse_x, mouse_y)
        back_btn.draw(frame)
        
        if mouse_clicked and back_btn.is_clicked(mouse_x, mouse_y):
            mode = None
            access_system.reset()
            gesture_seq_buffer.clear()
            print("\n🔙 Kembali ke menu utama")
            mouse_clicked = False

        if current_gesture and current_gesture != "neutral":
            gesture_text = f"Gesture: {current_gesture}"
            draw_text_with_outline(frame, gesture_text,
                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            bar_x, bar_y = 30, 130
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
                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

        status_y = 200
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

        draw_text_with_outline(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (w - 350, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            access_system.reset()
            gesture_seq_buffer.clear()
            print("🔄 Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
