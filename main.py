# =========================================================================
#  Be More Agent — BMO Unified Controller 🤖
#  A Local, Offline-First AI Agent & Face Tracker for Raspberry Pi 4
# =========================================================================

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import os
import subprocess
import random
import re
import sys
import select
import traceback
import atexit
import datetime
import warnings
import wave
import queue

# Suppress harmless library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

# Core audio & numerical dependencies
import sounddevice as sd
import numpy as np
import scipy.signal

# --- AI ENGINES ---
try:
    import openwakeword
    from openwakeword.model import Model
    OWW_AVAILABLE = True
except ImportError:
    OWW_AVAILABLE = False
    print("[INIT WARNING] openwakeword not installed — Wake Word detection will be disabled.")

import ollama

# --- SPEECH TO TEXT ---
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[INIT WARNING] faster-whisper not installed — STT transcription will be disabled.")

# --- WEB SEARCH ---
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("[INIT WARNING] duckduckgo-search not installed — Web search action will be disabled.")

# --- COMPUTER VISION ---
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[INIT WARNING] opencv-python not installed — Camera & Face Detection disabled.")

# --- RASPBERRY PI HARDWARE ---
# Ensure system apt packages (libcamera / picamera2 / lgpio) are visible even inside a virtualenv
_SYS_DIST_PACKAGES = "/usr/lib/python3/dist-packages"
if os.path.isdir(_SYS_DIST_PACKAGES) and _SYS_DIST_PACKAGES not in sys.path:
    sys.path.append(_SYS_DIST_PACKAGES)

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except Exception as _picam_err:
    PICAMERA_AVAILABLE = False
    print(f"[INIT INFO] picamera2 unavailable ({_picam_err}) — Will fall back to OpenCV VideoCapture.")

try:
    from gpiozero import Servo
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[INIT INFO] gpiozero not installed — Will use Mock Servos.")

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.3

# HARDWARE SETTINGS
INPUT_DEVICE_NAME = None

DEFAULT_CONFIG = {
    "text_model": "gemma3:1b",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "chat_memory": True,
    "system_prompt_extras": "",
    "input_device": None,
    "input_sample_rate": None,
    "servo_gpio1": 12,
    "servo_gpio2": 13,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 15
}

# LLM SETTINGS
# num_thread=2 keeps 2 cores free on the Pi 4 for audio, wake-word, and camera pipelines.
# num_ctx=512 prevents memory swap thrashing.
OLLAMA_OPTIONS = {
    'keep_alive': '-1',
    'num_thread': 2,
    'num_ctx': 512,
    'temperature': 0.7,
    'top_k': 40,
    'top_p': 0.9
}

def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"Config Error: {e}. Using defaults.")
    return config

CURRENT_CONFIG = load_config()
TEXT_MODEL = CURRENT_CONFIG["text_model"]

# USB microphone keywords for auto-detection
USB_MIC_KEYWORDS = ["usb", "pnp", "uac", "microphone", "mic"]

def resolve_input_device(config):
    requested = config.get("input_device")
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[AUDIO] Device query failed: {e}", flush=True)
        return None

    # Log available audio input devices
    print("[AUDIO] Available devices:", flush=True)
    for idx, dev in enumerate(devices):
        tag = "IN" if dev.get("max_input_channels", 0) > 0 else "  "
        print(f"  [{tag}] {idx}: {dev.get('name')}", flush=True)

    if requested not in (None, "", "default"):
        if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
            index = int(requested)
            if 0 <= index < len(devices):
                return index
            print(f"[AUDIO] Configured device index {index} not found.", flush=True)
            return None

        requested_lower = str(requested).lower()
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0 and requested_lower in dev.get("name", "").lower():
                return idx
        print(f"[AUDIO] Configured device name '{requested}' not found.", flush=True)
        return None

    # Auto-detect: Prefer USB microphone devices
    for idx, dev in enumerate(devices):
        name = dev.get("name", "").lower()
        if dev.get("max_input_channels", 0) > 0:
            if any(kw in name for kw in USB_MIC_KEYWORDS):
                print(f"[AUDIO] Auto-detected USB mic: [{idx}] {dev.get('name')}", flush=True)
                return idx

    # Fallback to first available input device
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            print(f"[AUDIO] Fallback input device: [{idx}] {dev.get('name')}", flush=True)
            return idx

    print("[AUDIO] No input device found!", flush=True)
    return None

INPUT_DEVICE_NAME = resolve_input_device(CURRENT_CONFIG)
if INPUT_DEVICE_NAME is not None:
    try:
        device_info = sd.query_devices(INPUT_DEVICE_NAME)
        print(f"[AUDIO] Using input device: {device_info.get('name', INPUT_DEVICE_NAME)}", flush=True)
    except Exception:
        print(f"[AUDIO] Using input device index: {INPUT_DEVICE_NAME}", flush=True)

def choose_input_samplerate(device, preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)
    try:
        device_info = sd.query_devices(device)
        if "default_samplerate" in device_info:
            candidates.append(int(device_info["default_samplerate"]))
    except Exception as e:
        print(f"[AUDIO DEBUG] Query failed: {e}", flush=True)

    candidates.extend([44100, 48000, 32000, 16000])
    seen = set()
    for rate in candidates:
        if not rate or rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue
    return int(candidates[0]) if candidates else 44100

class BotStates:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    WARMUP = "warmup"

# --- SYSTEM PROMPT ---
BASE_SYSTEM_PROMPT = """You are a helpful robot assistant running on a Raspberry Pi.
Personality: Cute, helpful, robot.
Style: Short sentences. Enthusiastic. Plain text only - NO emojis, NO special symbols.

INSTRUCTIONS:
- For standard conversation, reply with plain text only. Do NOT add any prefix like "Normal Text:" or "You:".
- If the user asks for a physical action (time, search, photo/what you see), output a single JSON object. Do NOT wrap it in markdown or prefix it with text.

### EXAMPLES ###

User: What time is it?
You: {"action": "get_time", "value": "now"}

User: What do you see?
You: {"action": "describe_view", "value": "environment"}

User: Look at this.
You: {"action": "describe_view", "value": "target"}

User: Hello!
You: Hi! I am ready to help!

User: Search for news about robots.
You: {"action": "search_web", "value": "robots news"}

### END EXAMPLES ###
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")

# Sound Directories
greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"
error_sounds_dir = "sounds/error_sounds"

# =========================================================================
# 2. HARDWARE INTERFACING & FALLBACKS
# =========================================================================

class DummyServo:
    """Mock Servo used for testing or when running on non-Pi platforms."""
    def __init__(self, gpio_pin):
        self.gpio_pin = gpio_pin
        self._value = None
        print(f"[MOCK MOTOR] Mock Servo initialized on GPIO {gpio_pin}", flush=True)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

# =========================================================================
# 3. MOTION & SERVO CONTROLLER (Queue-Driven, Dedicated Thread)
# =========================================================================

class MotorController:
    """
    Manages continuous rotation tracking servos using a thread-safe queue.
    Prevents command queuing lag by draining old commands in the queue.
    """
    def __init__(self, gpio1=12, gpio2=13, min_pwm=0.001, max_pwm=0.002):
        self.gpio1 = gpio1
        self.gpio2 = gpio2
        self.min_pwm = min_pwm
        self.max_pwm = max_pwm
        self.servo1 = None
        self.servo2 = None
        self.queue = queue.Queue(maxsize=10)
        self.thread = None
        self.running = False
        self.init_hardware()

    def init_hardware(self):
        if GPIO_AVAILABLE:
            try:
                # gpiozero automatically resolves using lgpio on newer Raspberry Pi OS
                self.servo1 = Servo(self.gpio1, min_pulse_width=self.min_pwm, max_pulse_width=self.max_pwm)
                self.servo2 = Servo(self.gpio2, min_pulse_width=self.min_pwm, max_pulse_width=self.max_pwm)
                print(f"[MOTOR] Hardware servos initialized on GPIO {self.gpio1} and {self.gpio2}.", flush=True)
            except Exception as e:
                print(f"[MOTOR WARNING] Failed to initialize hardware servos: {e}. Falling back to mocks.", flush=True)
                self.servo1 = DummyServo(self.gpio1)
                self.servo2 = DummyServo(self.gpio2)
        else:
            self.servo1 = DummyServo(self.gpio1)
            self.servo2 = DummyServo(self.gpio2)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="MotorControllerThread")
        self.thread.start()
        print("[MOTOR] Motor worker thread started.", flush=True)

    def stop(self):
        self.running = False
        if self.thread:
            try:
                self.queue.put(('exit',), timeout=0.1)
                self.thread.join(timeout=1.0)
            except:
                pass
        self.stop_robot()

    def stop_robot(self):
        try:
            if self.servo1 and (self.servo1.value is not None or self.servo1.value != 0):
                self.servo1.value = None
            if self.servo2 and (self.servo2.value is not None or self.servo2.value != 0):
                self.servo2.value = None
        except Exception as e:
            print(f"[MOTOR ERROR] Failed to stop servos: {e}", flush=True)

    def set_move(self, error_x):
        try:
            self.queue.put_nowait(('track', error_x))
        except queue.Full:
            # Drop command if queue is full to avoid latency build-up
            pass

    def trigger_stop(self):
        try:
            self.queue.put_nowait(('stop',))
        except queue.Full:
            pass

    def _worker(self):
        while self.running:
            try:
                # Get the command, timeout allows checks of self.running
                cmd = self.queue.get(timeout=0.1)
                
                # Dynamic optimization: Drain the queue to only execute the latest command
                while not self.queue.empty():
                    try:
                        cmd = self.queue.get_nowait()
                    except queue.Empty:
                        break

                action = cmd[0]
                if action == 'exit':
                    break
                elif action == 'stop':
                    self.stop_robot()
                elif action == 'track':
                    error_x = cmd[1]
                    self.track_move(error_x)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[MOTOR ERROR] Worker error: {e}", flush=True)

    def track_move(self, error_x):
        """
        Calculates motor movement based on target offsets.
        Preserves original proportional movement math.
        """
        deadzone = 0.25         # Ignore small errors to prevent servo jitter
        base_track_speed = 0.6  # Maximum turning speed cap
        
        if abs(error_x) > deadzone:
            speed = base_track_speed * abs(error_x)
            speed = max(0.15, min(speed, base_track_speed))
            
            if error_x > 0:
                # Target is to the RIGHT -> Turn Right
                self.servo1.value = speed
                self.servo2.value = speed
            else:
                # Target is to the LEFT -> Turn Left
                self.servo1.value = -speed
                self.servo2.value = -speed
        else:
            self.stop_robot()

# =========================================================================
# 3.5 OBJECT DETECTION PIPELINE (Lightweight MobileNet-SSD)
# =========================================================================

class ObjectDetector:
    def __init__(self):
        self.model_dir = "models"
        self.proto_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
        self.model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
        
        self.proto_path = os.path.join(self.model_dir, "deploy.prototxt")
        self.model_path = os.path.join(self.model_dir, "mobilenet_iter_73000.caffemodel")
        
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.net = None
        self._initialized = False

    def initialize(self):
        if self._initialized: return True
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            import urllib.request
            if not os.path.exists(self.proto_path):
                print(f"[VISION] Downloading prototxt: {self.proto_url}...", flush=True)
                urllib.request.urlretrieve(self.proto_url, self.proto_path)
            if not os.path.exists(self.model_path):
                print(f"[VISION] Downloading caffemodel weights: {self.model_url}...", flush=True)
                urllib.request.urlretrieve(self.model_url, self.model_path)
            if CV2_AVAILABLE:
                self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self._initialized = True
                print("[VISION] MobileNet-SSD Object Detector initialized successfully.", flush=True)
                return True
        except Exception as e:
            print(f"[VISION ERROR] Failed to initialize detector: {e}", flush=True)
        return False

    def detect_objects(self, frame, confidence_threshold=0.3):
        if not self.initialize() or frame is None or self.net is None: return []
        try:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            detected_classes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    class_id = int(detections[0, 0, i, 1])
                    if 0 < class_id < len(self.classes):
                        detected_classes.append(self.classes[class_id])
            return sorted(list(set(detected_classes)))
        except Exception as e:
            print(f"[VISION ERROR] Object detection execution failed: {e}", flush=True)
            return []

# =========================================================================
# 4. CAMERA PIPELINE (Unified Frame Capture & Tracking, Dedicated Thread)
# =========================================================================

class CameraPipeline:
    """
    Manages camera device streams (Picamera2 or OpenCV fallback) in a separate thread.
    Downscales frames for low-CPU face detection and feeds offsets to MotorController.
    """
    def __init__(self, motor_controller, width=640, height=480, fps=15):
        self.motor_controller = motor_controller
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.thread = None
        
        self.face_detection_enabled = True
        self.preview_enabled = False
        self.preview_callback = None  # Receives RGB image when preview is active
        self.latest_frame = None
        
        self.cap = None
        self.picam = None
        
        # Cascade classifier for face detection
        self.face_cascade = None
        self.init_cascade()

    def init_cascade(self):
        if not CV2_AVAILABLE:
            return
        cascade_paths = [
            'haarcascade_frontalface_default.xml',
            os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        ]
        for path in cascade_paths:
            if os.path.exists(path):
                try:
                    self.face_cascade = cv2.CascadeClassifier(path)
                    print(f"[CAMERA] Loaded cascade classifier from: {path}", flush=True)
                    break
                except Exception as e:
                    print(f"[CAMERA ERROR] Error loading cascade from {path}: {e}", flush=True)
        if not self.face_cascade:
            print("[CAMERA WARNING] No face cascade classifier loaded. Face tracking will be disabled.", flush=True)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="CameraPipelineThread")
        self.thread.start()
        print("[CAMERA] Camera pipeline thread started.", flush=True)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self._cleanup_camera()

    def _get_v4l2_capture_indices(self):
        """Returns candidate /dev/videoN indices, filtering out Raspberry Pi ISP/codec metadata nodes."""
        v4l_dir = "/sys/class/video4linux"
        skip_keywords = ("pispbe", "rpi-hevc", "bcm2835-isp", "bcm2835-codec")
        preferred = []
        fallback = []
        if os.path.isdir(v4l_dir):
            for entry in sorted(os.listdir(v4l_dir)):
                if not entry.startswith("video") or not entry[5:].isdigit():
                    continue
                idx = int(entry[5:])
                name_file = os.path.join(v4l_dir, entry, "name")
                dev_name = ""
                if os.path.exists(name_file):
                    try:
                        with open(name_file, "r") as f:
                            dev_name = f.read().strip().lower()
                    except Exception:
                        pass
                if any(kw in dev_name for kw in skip_keywords):
                    continue
                if "rp1-cfe" in dev_name:
                    fallback.append(idx)
                else:
                    preferred.append(idx)
            candidates = preferred + fallback
            if candidates:
                return candidates
        return [0, 1, 2, 4, 6, 8]

    def _init_camera(self):
        # 1. Try Picamera2 first (Native Raspberry Pi CSI camera)
        if PICAMERA_AVAILABLE:
            try:
                cam_info = Picamera2.global_camera_info()
                if cam_info:
                    self.picam = Picamera2()
                    config = self.picam.create_preview_configuration(
                        main={"size": (1640, 1232), "format": "BGR888"}
                    )
                    self.picam.configure(config)
                    self.picam.start()
                    print(f"[CAMERA] Initialized Picamera2 (1640x1232 binned at {self.fps} FPS).", flush=True)
                    return True
                else:
                    print("[CAMERA INFO] Picamera2 found 0 CSI cameras connected. Falling back to OpenCV.", flush=True)
            except Exception as e:
                print(f"[CAMERA INFO] Picamera2 startup failed: {e}. Falling back to OpenCV.", flush=True)
                self.picam = None

        # 2. Try OpenCV (USB / V4L2 cameras) using explicit V4L2 backend to avoid obsensor errors
        if CV2_AVAILABLE:
            backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
            for idx in self._get_v4l2_capture_indices():
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        # Perform a test read to ensure this is a valid video stream node
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.cap = cap
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            print(f"[CAMERA] Initialized OpenCV VideoCapture on /dev/video{idx} ({self.width}x{self.height}).", flush=True)
                            return True
                        else:
                            cap.release()
                except Exception as ex:
                    print(f"[CAMERA INFO] Failed opening camera index {idx}: {ex}", flush=True)
        
        print("[CAMERA INFO] No active CSI or USB camera detected — continuing without camera.", flush=True)
        return False

    def _cleanup_camera(self):
        if self.picam:
            try:
                self.picam.stop()
                self.picam.close()
            except:
                pass
            self.picam = None
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        print("[CAMERA] Camera hardware released.", flush=True)

    def _run_loop(self):
        if not self._init_camera():
            print("[CAMERA ERROR] Camera pipeline thread exiting: init failed.", flush=True)
            self.running = False
            return

        frame_delay = 1.0 / self.fps
        last_face_time = 0
        face_lost_delay = 0.5  # Settle time before stopping motors when losing target

        while self.running:
            start_time = time.time()
            frame = None
            
            try:
                if self.picam:
                    frame_array = self.picam.capture_array()
                    # Convert RGB (Picamera2 standard) to BGR for OpenCV processing
                    frame_raw = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    frame_raw = cv2.flip(frame_raw, -1)  # Flip if mounted upside down
                    frame = cv2.resize(frame_raw, (self.width, self.height))
                elif self.cap:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        print("[CAMERA WARNING] Failed to read frame from OpenCV capture device.", flush=True)
                        time.sleep(0.1)
                        continue
            except Exception as e:
                print(f"[CAMERA ERROR] Frame capture failed: {e}", flush=True)
                time.sleep(0.2)
                continue

            if frame is None:
                time.sleep(0.01)
                continue

            self.latest_frame = frame

            # Process frame
            face_detected = False
            error_x = 0.0

            if self.face_detection_enabled and self.face_cascade is not None:
                # Downscale 4x to minimize CPU calculations
                scale_factor = 4
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray, (0, 0), fx=(1.0 / scale_factor), fy=(1.0 / scale_factor))
                
                faces = self.face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5)
                
                if len(faces) > 0:
                    # Target the first face found to avoid target jittering
                    (x, y, w, h) = faces[0]
                    # Scale bounding box back up for display
                    x, y, w, h = x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor
                    face_center_x = x + (w // 2)
                    face_detected = True
                    last_face_time = time.time()
                    
                    # Normalize deviation to range [-1.0, 1.0]
                    error_x = (face_center_x - (self.width // 2)) / (self.width // 2)
                    self.motor_controller.set_move(error_x)
                    
                    if self.preview_enabled:
                        # Blue bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # Red center point tracking dot
                        cv2.circle(frame, (face_center_x, y + (h // 2)), 5, (0, 0, 255), -1)

            if not face_detected:
                # If no faces are seen, wait for the lost delay before cutting servo speeds
                if time.time() - last_face_time > face_lost_delay:
                    self.motor_controller.trigger_stop()

            # Render overlay if Picture-in-Picture display is toggled on in GUI
            if self.preview_enabled and self.preview_callback:
                # Green center crosshair
                cv2.line(frame, ((self.width // 2) - 20, self.height // 2), ((self.width // 2) + 20, self.height // 2), (0, 255, 0), 2)
                cv2.line(frame, (self.width // 2, (self.height // 2) - 20), (self.width // 2, (self.height // 2) + 20), (0, 255, 0), 2)
                
                # Convert BGR to RGB for PIL integration
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.preview_callback(frame_rgb)

            # Sleep to match targeted framerate
            elapsed = time.time() - start_time
            sleep_time = max(0.005, frame_delay - elapsed)
            time.sleep(sleep_time)

# =========================================================================
# 5. BOT GUI & INTEGRATED EVENT LOOP
# =========================================================================

class BotGUI:
    BG_WIDTH, BG_HEIGHT = 800, 480
    CAM_W, CAM_H = 240, 180  # PIP overlay dimensions

    def __init__(self, master):
        self.master = master
        master.title("BMO Social Assistant")
        master.attributes('-fullscreen', True)
        master.bind('<Escape>', self.exit_fullscreen)
        
        # User Interaction Bindings
        master.bind('<Return>', self.handle_ptt_toggle)
        master.bind('<space>', self.handle_speaking_interrupt)
        atexit.register(self.safe_exit)
        
        # Core State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0
        self.animations = {}
        self.current_frame_index = 0
        
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()
        
        self.last_ptt_time = 0
        self.ptt_event = threading.Event()
        self.recording_active = threading.Event()
        self.interrupted = threading.Event()
        
        self.tts_queue = []
        self.tts_queue_lock = threading.Lock()
        self.tts_thread = None
        self.tts_active = threading.Event()
        self.idle_event = threading.Event()
        self.auto_listen = True
        self.current_audio_process = None
        self.exiting = False

        # --- SUB-THREAD INITIALIZATION ---
        print("[INIT] Starting Motor Controller...", flush=True)
        self.motor_controller = MotorController(
            gpio1=CURRENT_CONFIG.get("servo_gpio1", 12),
            gpio2=CURRENT_CONFIG.get("servo_gpio2", 13)
        )
        self.motor_controller.start()

        print("[INIT] Starting Camera Pipeline...", flush=True)
        self.camera_pipeline = CameraPipeline(
            motor_controller=self.motor_controller,
            width=CURRENT_CONFIG.get("camera_width", 640),
            height=CURRENT_CONFIG.get("camera_height", 480),
            fps=CURRENT_CONFIG.get("camera_fps", 15)
        )
        self.camera_pipeline.start()

        # --- WAKE WORD INITIALIZATION ---
        print("[INIT] Loading Wake Word...", flush=True)
        self.oww_model = None
        if OWW_AVAILABLE and os.path.exists(WAKE_WORD_MODEL):
            try:
                self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")
                print("[INIT] Wake Word Loaded (ONNX).", flush=True)
            except TypeError:
                try:
                    self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                    print("[INIT] Wake Word Loaded (Legacy API).", flush=True)
                except Exception as e:
                    print(f"[CRITICAL] Failed to load wake word model: {e}")
            except Exception as e:
                print(f"[CRITICAL] Failed to load wake word model: {e}")
        else:
            print(f"[CRITICAL] Wake word model file not found or package unavailable.")

        # --- SPEECH TO TEXT INITIALIZATION ---
        print("[INIT] Loading Whisper model...", flush=True)
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
                print("[INIT] Whisper ready.", flush=True)
            except Exception as e:
                print(f"[CRITICAL] Whisper failed to load: {e}", flush=True)
        else:
            print("[CRITICAL] Whisper STT unavailable.")

        # --- AUDIO OUTPUT SETUP ---
        self._output_device = self._find_output_device()
        self._piper_process = None
        self._init_piper()

        # GUI Setup
        self.background_label = tk.Label(master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind('<Button-1>', self.toggle_hud_visibility)
        
        self.response_text = tk.Text(master, height=6, width=60, wrap=tk.WORD, 
                                     state=tk.DISABLED, bg="#ffffff", fg="#000000", font=('Arial', 12))
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(master, textvariable=self.status_var, background="#2e2e2e", foreground="white")
        
        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)
        self.wake_button = ttk.Button(master, text="🎙 Wake Up", command=self.handle_wake_button)
        self.idle_button = ttk.Button(master, text="⏸ Idle", command=self.handle_idle_button)
        self.cam_button  = ttk.Button(master, text="📷 Cam", command=self.toggle_camera_feed)

        # Camera Picture-In-Picture Overlay
        self.cam_label = tk.Label(master, bg="black", relief="flat", bd=2)
        self._cam_photo = None  # PhotoImage garbage collection prevention reference

        self.load_animations()
        self.update_animation()
        
        # Start conversational agent engine loop in its own thread
        threading.Thread(target=self.safe_main_execution, daemon=True, name="AgentLoopThread").start()

    # --- UI EVENT HANDLERS ---

    def toggle_camera_feed(self):
        if self.camera_pipeline.preview_enabled:
            self.camera_pipeline.preview_enabled = False
            self.camera_pipeline.preview_callback = None
            self.cam_label.place_forget()
            self.cam_button.config(text="📷 Cam")
            print("[CAM] Preview overlay disabled", flush=True)
        else:
            self.camera_pipeline.preview_callback = self.handle_camera_frame
            self.camera_pipeline.preview_enabled = True
            
            # Place the Picture-in-Picture window at bottom-right, above the status HUD
            self.cam_label.place(
                x=self.BG_WIDTH  - self.CAM_W - 8,
                y=self.BG_HEIGHT - self.CAM_H - 36,
                width=self.CAM_W,
                height=self.CAM_H
            )
            self.cam_button.config(text="📷 Cam ●")
            print("[CAM] Preview overlay enabled", flush=True)

    def handle_camera_frame(self, frame_rgb):
        """Processes and schedules camera preview frame updates in the GUI thread."""
        try:
            img = Image.fromarray(frame_rgb).resize((self.CAM_W, self.CAM_H), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            
            def _update(p=photo):
                if self.camera_pipeline.preview_enabled:
                    self._cam_photo = p  # Keep alive
                    self.cam_label.config(image=p)
            self.master.after(0, _update)
        except Exception as e:
            print(f"[CAM ERROR] Preview rendering failed: {e}", flush=True)

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
                self.wake_button.place_forget()
                self.idle_button.place_forget()
                self.cam_button.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
                self.wake_button.place(x=110, y=10)
                self.idle_button.place(x=210, y=10)
                self.cam_button.place(x=310, y=10)
        except tk.TclError:
            pass

    def handle_ptt_toggle(self, event=None):
        current_time = time.time()
        if current_time - self.last_ptt_time < 0.5:
            return
        self.last_ptt_time = current_time

        if self.recording_active.is_set():
            print("[PTT] Toggle OFF", flush=True)
            self.recording_active.clear()
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                print("[PTT] Toggle ON", flush=True)
                self.recording_active.set()
                self.ptt_event.set()

    def handle_wake_button(self):
        print("[WAKE BTN] Manual wake triggered", flush=True)
        self.idle_event.clear()
        self.recording_active.set()
        self.ptt_event.set()
        
        def _auto_stop():
            time.sleep(5)
            self.recording_active.clear()
        threading.Thread(target=_auto_stop, daemon=True).start()

    def handle_idle_button(self):
        print("[IDLE BTN] Forcing idle", flush=True)
        self.idle_event.set()
        self.interrupted.set()
        self.recording_active.clear()
        with self.tts_queue_lock:
            self.tts_queue.clear()
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
            except:
                pass
        self.set_state(BotStates.IDLE, "Idle.")

    def handle_speaking_interrupt(self, event=None):
        if self.current_state in (BotStates.SPEAKING, BotStates.THINKING):
            self.interrupted.set()
            self.thinking_sound_active.clear()
            with self.tts_queue_lock:
                self.tts_queue.clear()
            if self.current_audio_process:
                try:
                    self.current_audio_process.terminate()
                except:
                    pass
            self.set_state(BotStates.IDLE, "Interrupted.")

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.safe_exit()

    def safe_exit(self):
        if self.exiting:
            return
        self.exiting = True
        print("\n--- SHUTDOWN SEQUENCE STARTED ---", flush=True)
        
        # Stop sub-threads
        if self.camera_pipeline:
            self.camera_pipeline.stop()
        if self.motor_controller:
            self.motor_controller.stop()

        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except:
                pass

        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear()
        
        self.save_chat_history()
        
        # Shutdown Ollama connection keepalive and free resources
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=0)
        except:
            pass
        try:
            sd.stop()
        except:
            pass

        try:
            self.master.quit()
        except Exception:
            pass

    # --- ANIMATION ENGINE ---

    def load_animations(self):
        base_path = "faces"
        states = ["idle", "listening", "thinking", "speaking", "error", "warmup"]
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
                for f in files:
                    try:
                        img = Image.open(os.path.join(folder, f)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                        self.animations[state].append(ImageTk.PhotoImage(img))
                    except Exception as e:
                        print(f"[GUI ERROR] Failed loading image {f}: {e}", flush=True)
            
            # Draw standard fallback canvases if folders are missing/empty
            if not self.animations[state]:
                if state in self.animations.get("idle", []):
                    self.animations[state] = self.animations["idle"]
                else:
                    color_mapping = {
                        "warmup": "#FFFF00", # Yellow
                        "listening": "#00FF00", # Green
                        "thinking": "#FFA500", # Orange
                        "speaking": "#800080", # Purple
                        "error": "#FF0000" # Red
                    }
                    fill = color_mapping.get(state, "#0000FF") # Default Blue
                    blank = Image.new('RGB', (self.BG_WIDTH, self.BG_HEIGHT), color=fill)
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(500, self.update_animation)
            return

        if self.current_state == BotStates.SPEAKING:
            if len(frames) > 1:
                self.current_frame_index = random.randint(1, len(frames) - 1)
            else:
                self.current_frame_index = 0
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)

        try:
            self.background_label.config(image=frames[self.current_frame_index])
        except tk.TclError:
            return  # GUI closing

        speed = 50 if self.current_state == BotStates.SPEAKING else 500
        self.master.after(speed, self.update_animation)

    def set_state(self, state, msg=""):
        def _update():
            if msg:
                print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg:
                self.status_var.set(msg)
        self.master.after(0, _update)

    def append_to_text(self, text, newline=True):
        def _update():
            try:
                self.response_text.config(state=tk.NORMAL)
                if newline:
                    self.response_text.insert(tk.END, text + "\n")
                else:
                    self.response_text.insert(tk.END, text)
                self.response_text.see(tk.END)
                self.response_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass
        self.master.after(0, _update)

    def _stream_to_text(self, chunk):
        def update_text_stream():
            try:
                self.response_text.config(state=tk.NORMAL)
                self.response_text.insert(tk.END, chunk)
                self.response_text.see(tk.END)
                self.response_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass
        self.master.after(0, update_text_stream)

    # =========================================================================
    # 6. ACTION ROUTER / TOOL EXECUTOR
    # =========================================================================

    def extract_json_from_text(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return None
        except:
            return None

    def execute_action_and_get_result(self, action_data):
        raw_action = action_data.get("action", "").lower().strip()
        value = action_data.get("value") or action_data.get("query")
        
        VALID_TOOLS = {"get_time", "search_web", "describe_view"}
        ALIASES = {
            "google": "search_web", "browser": "search_web", "news": "search_web",
            "search_news": "search_web", "check_time": "get_time",
            "look": "describe_view", "see": "describe_view", "describe_scene": "describe_view",
            "describe_image": "describe_view", "get_view": "describe_view"
        }

        action = ALIASES.get(raw_action, raw_action)
        print(f"ACTION TRIGGERED: {raw_action} -> {action}", flush=True)

        if action not in VALID_TOOLS:
            if value and isinstance(value, str) and len(value.split()) > 1:
                return f"CHAT_FALLBACK::{value}"
            return "INVALID_ACTION"

        if action == "get_time":
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."
            
        elif action == "describe_view":
            if self.camera_pipeline and self.camera_pipeline.latest_frame is not None:
                detector = getattr(self, "object_detector", None)
                if not detector:
                    detector = ObjectDetector()
                    self.object_detector = detector
                
                self.set_state(BotStates.THINKING, "Looking around...")
                detected = detector.detect_objects(self.camera_pipeline.latest_frame)
                if detected:
                    return f"I see: {', '.join(detected)}."
                else:
                    return "I don't see any recognizable objects."
            else:
                return "My camera is currently inactive or warming up."
        
        elif action == "search_web":
            if not DDGS_AVAILABLE:
                return "SEARCH_ERROR"
            print(f"Searching DuckDuckGo for: '{value}'...", flush=True)
            try:
                with DDGS() as ddgs:
                    results = []
                    # Try news search first
                    try:
                        results = list(ddgs.news(value, region='us-en', max_results=1))
                        if results:
                            print(f"[SEARCH] Found News: {results[0].get('title')}", flush=True)
                    except Exception as e:
                        print(f"[SEARCH DEBUG] News query failed: {e}", flush=True)
                    
                    # Fallback to general text search
                    if not results:
                        try:
                            results = list(ddgs.text(value, region='us-en', max_results=1))
                            if results:
                                print(f"[SEARCH] Found Text: {results[0].get('title')}", flush=True)
                        except Exception as e:
                            print(f"[SEARCH DEBUG] Text query failed: {e}", flush=True)

                    if results:
                        r = results[0]
                        title = r.get('title', 'No Title')
                        body = r.get('body', r.get('snippet', 'No Body'))
                        return f"SEARCH RESULTS for '{value}':\nTitle: {title}\nSnippet: {body[:300]}"
                    else:
                        print("[SEARCH] Returned 0 results.", flush=True)
                        return "SEARCH_EMPTY"
            except Exception as e:
                print(f"[SEARCH ERROR] Connection/API error: {e}", flush=True)
                return "SEARCH_ERROR"
        
        return None

    # =========================================================================
    # 7. CONVERSATIONAL AGENT EXECUTION
    # =========================================================================

    def safe_main_execution(self):
        try:
            self.warm_up_logic()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True, name="TTSWorkerThread")
            self.tts_thread.start()
            
            while True:
                trigger_source = self.detect_wake_word_or_ptt()
                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Resetting...")
                    continue

                self.set_state(BotStates.LISTENING, "I'm listening!")
                
                audio_file = None
                if trigger_source == "PTT":
                    audio_file = self.record_voice_ptt()
                else:
                    audio_file = self.record_voice_adaptive()
                
                if not audio_file:
                    self.set_state(BotStates.IDLE, "Heard nothing.")
                    continue
                
                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue
                
                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self.idle_event.clear()
                self.chat_and_respond(user_text)

                # Restart listening trigger unless idle commanded
                if not self.idle_event.is_set():
                    print("[AUTO] Listening active again...", flush=True)
                    self.recording_active.set()
                    self.ptt_event.set()
                    def _auto_stop():
                        time.sleep(5)
                        self.recording_active.clear()
                    threading.Thread(target=_auto_stop, daemon=True).start()
                    
        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Warming up brains...")
        try:
            # Tell Ollama to load model and pin to memory (keep_alive = -1)
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=-1)
        except Exception as e:
            print(f"Failed loading LLM model {TEXT_MODEL}: {e}", flush=True)
        
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        print("[INIT] Models loaded & active.", flush=True)

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Waiting...")
        self.ptt_event.clear()
        
        if self.oww_model:
            self.oww_model.reset()

        if self.oww_model is None:
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"

        OWW_SAMPLE_RATE = 16000
        OWW_CHUNK = 1280

        input_rate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))
        use_resampling = (input_rate != OWW_SAMPLE_RATE)
        BLOCK_SIZE = 512

        stream_args = {
            "samplerate": input_rate,
            "channels": 1,
            "dtype": 'int16',
            "blocksize": BLOCK_SIZE,
            "device": INPUT_DEVICE_NAME,
            "latency": "low",
        }

        print(f"[AUDIO] Wake word stream: {input_rate}Hz resample={use_resampling}", flush=True)

        try:
            self._listen_loop(stream_args, input_rate, OWW_SAMPLE_RATE, OWW_CHUNK, use_resampling)
        except StopIteration as si:
            return str(si)
        except Exception as e:
            print(f"[AUDIO] Stream error: {e}. Falling back to PTT.", flush=True)
            self.ptt_event.wait()
            return "PTT"

        return "WAKE"

    def _listen_loop(self, stream_args, input_rate, oww_rate, oww_chunk, use_resampling):
        needed_input_samples = int(oww_chunk * (input_rate / oww_rate)) if use_resampling else oww_chunk
        accumulator = np.array([], dtype=np.int16)
        overflow_count = 0
        MAX_OVERFLOWS = 30

        with sd.InputStream(**stream_args) as stream:
            while True:
                # Polling interrupts
                if self.ptt_event.is_set():
                    self.ptt_event.clear()
                    raise StopIteration("PTT")
                
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    sys.stdin.readline()
                    raise StopIteration("CLI")

                try:
                    data, overflow = stream.read(stream_args["blocksize"])
                except Exception as e:
                    raise RuntimeError(f"Stream read failure: {e}")

                if overflow:
                    overflow_count += 1
                    if overflow_count > MAX_OVERFLOWS:
                        raise RuntimeError(f"Stream buffer overflows exceeded limit ({overflow_count})")
                    continue
                else:
                    overflow_count = max(0, overflow_count - 1)

                chunk = np.frombuffer(data, dtype=np.int16)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()
                accumulator = np.concatenate((accumulator, chunk))

                if len(accumulator) < needed_input_samples:
                    continue

                window = accumulator[:needed_input_samples]
                accumulator = accumulator[needed_input_samples:]

                if use_resampling:
                    from math import gcd
                    g = gcd(input_rate, oww_rate)
                    up, down = oww_rate // g, input_rate // g
                    window = scipy.signal.resample_poly(window, up, down).astype(np.int16)
                    if len(window) > oww_chunk:
                        window = window[:oww_chunk]
                    elif len(window) < oww_chunk:
                        window = np.pad(window, (0, oww_chunk - len(window)))

                current_max = int(np.max(np.abs(window)))
                bar_length = 30
                filled = int((current_max / 32768) * bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                level_label = "LOUD" if current_max > 10000 else ("MED " if current_max > 2000 else "LOW ")
                oww_score_str = ""

                if current_max > 200:
                    self.oww_model.predict(window)
                    for mdl in self.oww_model.prediction_buffer.keys():
                        score = list(self.oww_model.prediction_buffer[mdl])[-1]
                        if score > 0.005:
                            oww_score_str = f" | 🎯 Wake: {score:.3f}"
                        if score > WAKE_WORD_THRESHOLD:
                            print(f"\n[WAKE] Wake Word Triggered! Score={score:.2f}", flush=True)
                            self.oww_model.reset()
                            return

                print(f"\r[MIC] {level_label} |{bar}| {current_max:5d}{oww_score_str}   ", end="", flush=True)

    def record_voice_adaptive(self, filename="input.wav"):
        print("Recording (Adaptive)...", flush=True)
        time.sleep(0.5)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        silence_threshold = 0.006
        silence_duration = 1.5
        max_record_time = 30.0
        buffer = []
        silent_chunks = 0
        chunk_duration = 0.05
        chunk_size = int(samplerate * chunk_duration)
        
        num_silent_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)
        recorded_chunks = 0
        silence_started = False

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())
            recorded_chunks += 1
            if recorded_chunks < 5:
                return
            if volume_norm < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= num_silent_chunks:
                    silence_started = True
            else:
                silent_chunks = 0

        try:
            sd.stop()
            time.sleep(0.2)
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, 
                                device=INPUT_DEVICE_NAME, blocksize=chunk_size):
                while not silence_started and recorded_chunks < max_chunks:
                    sd.sleep(int(chunk_duration * 1000))
        except Exception as e:
            print(f"[AUDIO ERROR] Adaptive recording failed: {e}", flush=True)
            return None
        
        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        print("Recording (PTT)...", flush=True)
        time.sleep(0.5)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))
        buffer = []
        
        def callback(indata, frames, time_info, status):
            buffer.append(indata.copy())
        
        try:
            sd.stop()
            time.sleep(0.2)
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=INPUT_DEVICE_NAME):
                while self.recording_active.is_set():
                    sd.sleep(50)
        except Exception as e:
            print(f"[AUDIO ERROR] PTT recording failed: {e}", flush=True)
            return None
            
        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer:
            return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        audio_data = (audio_data * 32767).astype(np.int16)

        WHISPER_RATE = 16000
        if samplerate != WHISPER_RATE:
            from math import gcd
            g = gcd(int(samplerate), WHISPER_RATE)
            up, down = WHISPER_RATE // g, int(samplerate) // g
            audio_data = scipy.signal.resample_poly(audio_data, up, down).astype(np.int16)
            print(f"[AUDIO] Resampled {samplerate}Hz -> {WHISPER_RATE}Hz for Whisper", flush=True)
            samplerate = WHISPER_RATE

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        
        print(f"[AUDIO] Saved {len(audio_data)/samplerate:.1f}s audio sample to {filename}", flush=True)
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    def transcribe_audio(self, filename):
        print("Transcribing...", flush=True)
        if self.whisper_model is None:
            print("[STT ERROR] Whisper model not loaded.", flush=True)
            return ""
        try:
            if isinstance(filename, np.ndarray):
                audio_float32 = np.ascontiguousarray(filename.squeeze(), dtype=np.float32)
            else:
                with wave.open(filename, "rb") as wf:
                    sr = wf.getframerate()
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    raw_bytes = wf.readframes(wf.getnframes())

                if not raw_bytes:
                    print("[STT] Heard: (empty audio)", flush=True)
                    return ""

                if sampwidth == 2:
                    audio_float32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    audio_float32 = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

                if nchannels > 1:
                    audio_float32 = audio_float32.reshape(-1, nchannels).mean(axis=1)

                if sr != 16000:
                    from math import gcd
                    g = gcd(int(sr), 16000)
                    audio_float32 = scipy.signal.resample_poly(
                        audio_float32, 16000 // g, int(sr) // g
                    ).astype(np.float32)

                audio_float32 = np.ascontiguousarray(audio_float32, dtype=np.float32)

            if audio_float32.size == 0:
                print("[STT] Heard: (empty audio)", flush=True)
                return ""

            segments, info = self.whisper_model.transcribe(
                audio_float32,
                language="en",
                beam_size=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.0,
                log_prob_threshold=-0.8,
            )
            
            good_segments = [
                seg.text for seg in segments
                if seg.no_speech_prob < 0.6
            ]
            transcription = " ".join(good_segments).strip()
            if not transcription:
                print("[STT] Heard: (nothing confident)", flush=True)
                return ""
            print(f"[STT] Heard: '{transcription}'", flush=True)
            return transcription
        except Exception as e:
            print(f"[STT ERROR] Whisper decoding failed: {e}", flush=True)
            return ""

    def chat_and_respond(self, text):
        if "forget everything" in text.lower() or "reset memory" in text.lower():
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.save_chat_history()
            with self.tts_queue_lock:
                self.tts_queue.append("Okay. Memory wiped.")
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        self.set_state(BotStates.THINKING, "Thinking...")
        
        user_msg = {"role": "user", "content": text}
        messages = self.permanent_memory + self.session_memory + [user_msg]
        self.session_memory.append(user_msg)
        
        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True, name="ThinkingSoundThread").start()
        
        full_response_buffer = ""
        chunk_buffer = ""

        try:
            stream = ollama.chat(model=TEXT_MODEL, messages=messages, stream=True, options=OLLAMA_OPTIONS)
            is_action_mode = False
            bot_label_written = False

            for chunk in stream:
                if self.interrupted.is_set():
                    break
                raw = chunk['message']['content']
                content = raw.encode('ascii', 'ignore').decode('ascii')
                full_response_buffer += content

                # Determine action mode based on the first non-whitespace character in the stream
                stripped = full_response_buffer.lstrip()
                if stripped and stripped.startswith('{'):
                    is_action_mode = True
                    self.thinking_sound_active.clear()
                    continue

                if is_action_mode:
                    continue

                self.thinking_sound_active.clear()

                if not bot_label_written:
                    self.append_to_text("BOT: ", newline=False)
                    bot_label_written = True

                self._stream_to_text(content)
                chunk_buffer += content

                # Sentence parser to stream to TTS workers early
                if re.search(r'[.!?\n]', chunk_buffer):
                    text_sentence = chunk_buffer.encode('ascii', 'ignore').decode('ascii').strip()
                    if text_sentence and re.search(r'[a-zA-Z0-9]', text_sentence):
                        with self.tts_queue_lock:
                            self.tts_queue.append(text_sentence)
                    chunk_buffer = ""

            if chunk_buffer.strip():
                text_sentence = chunk_buffer.encode('ascii', 'ignore').decode('ascii').strip()
                if text_sentence and re.search(r'[a-zA-Z0-9]', text_sentence):
                    with self.tts_queue_lock:
                        self.tts_queue.append(text_sentence)
                chunk_buffer = ""

            if is_action_mode:
                action_data = self.extract_json_from_text(full_response_buffer)
                if action_data:
                    tool_result = self.execute_action_and_get_result(action_data)

                    if tool_result and tool_result.startswith("CHAT_FALLBACK::"):
                        chat_text = tool_result.split("::", 1)[1]
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(chat_text, newline=True)
                        with self.tts_queue_lock:
                            self.tts_queue.append(chat_text)
                        self.session_memory.append({"role": "assistant", "content": chat_text})
                        self.save_chat_history()
                        self.wait_for_tts()
                        self.set_state(BotStates.IDLE, "Ready")
                        return

                    if tool_result == "INVALID_ACTION":
                        fallback_text = "I am not sure how to do that."
                    elif tool_result == "SEARCH_EMPTY":
                        fallback_text = "I searched, but I couldn't find any news about that."
                    elif tool_result == "SEARCH_ERROR":
                        fallback_text = "I cannot reach the internet right now."
                    else:
                        fallback_text = None

                    if fallback_text:
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(fallback_text, newline=True)
                        with self.tts_queue_lock:
                            self.tts_queue.append(fallback_text)
                    elif tool_result:
                        summary_prompt = [
                            {"role": "system", "content": "Summarize this result in one short sentence."},
                            {"role": "user", "content": f"RESULT: {tool_result}\nUser Question: {text}"}
                        ]
                        self.set_state(BotStates.THINKING, "Reading...")
                        self.thinking_sound_active.set()
                        
                        final_resp = ollama.chat(model=TEXT_MODEL, messages=summary_prompt, stream=False, options=OLLAMA_OPTIONS)
                        final_text = final_resp['message']['content']
                        
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(final_text, newline=True)
                        with self.tts_queue_lock:
                            self.tts_queue.append(final_text)
                        self.session_memory.append({"role": "assistant", "content": final_text})
            else:
                self.append_to_text("")
                self.session_memory.append({"role": "assistant", "content": full_response_buffer})
            
            self.save_chat_history()
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")
                
        except Exception as e:
            print(f"LLM Chat Error: {e}")
            self.set_state(BotStates.ERROR, "Brain Freeze!")

    def wait_for_tts(self):
        while True:
            if self.interrupted.is_set():
                break
            with self.tts_queue_lock:
                empty = len(self.tts_queue) == 0
            if empty and not self.tts_active.is_set():
                break
            time.sleep(0.05)

    def _tts_worker(self):
        while True:
            with self.tts_queue_lock:
                if self.tts_queue:
                    full_text = self.tts_queue.pop(0)
                else:
                    full_text = None
            if full_text:
                self.tts_active.set()
                self.speak(full_text)
                self.tts_active.clear()
            else:
                time.sleep(0.03)

    def _init_piper(self):
        voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_GB-semaine-medium.onnx")
        if not os.path.exists("./piper/piper"):
            print("[PIPER ERROR] ./piper/piper binary not found!", flush=True)
            return
        if not os.path.exists(voice_model):
            print(f"[PIPER ERROR] Voice model not found: {voice_model}", flush=True)
            return
        try:
            if self._piper_process and self._piper_process.poll() is None:
                self._piper_process.terminate()
            
            self._piper_process = subprocess.Popen(
                ["./piper/piper", "--model", voice_model, "--output-raw", "--json-input"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            print("[INIT] Persistent Piper TTS subprocess started.", flush=True)

            PIPER_RATE = 22050
            use_native_rate = False
            native_rate = 48000
            try:
                dev_info = sd.query_devices(self._output_device)
                native_rate = int(dev_info['default_samplerate'])
            except:
                pass
            try:
                sd.check_output_settings(device=self._output_device, samplerate=PIPER_RATE, channels=1)
            except:
                use_native_rate = True
            
            self._piper_rate = PIPER_RATE
            self._piper_native_rate = native_rate
            self._piper_use_native_rate = use_native_rate
        except Exception as e:
            print(f"[PIPER ERROR] Initialization failed: {e}", flush=True)
            self._piper_process = None

    def _find_output_device(self):
        """Scans and finds best output hardware. Prefers USB connections."""
        USB_OUTPUT_KEYWORDS = ["uac", "usb"]
        FALLBACK_KEYWORDS = ["headphones", "bcm2835", "headphone", "speaker", "analog"]
        try:
            devices = sd.query_devices()
            print("[AUDIO] Scanning output devices:", flush=True)
            for idx, dev in enumerate(devices):
                if dev.get("max_output_channels", 0) > 0:
                    print(f"  [OUT] {idx}: {dev.get('name')}", flush=True)

            for idx, dev in enumerate(devices):
                name = dev.get("name", "").lower()
                if dev.get("max_output_channels", 0) > 0:
                    if any(kw in name for kw in USB_OUTPUT_KEYWORDS):
                        print(f"[AUDIO] Selected output (USB): [{idx}] {dev.get('name')}", flush=True)
                        return idx

            for idx, dev in enumerate(devices):
                name = dev.get("name", "").lower()
                if dev.get("max_output_channels", 0) > 0:
                    if any(kw in name for kw in FALLBACK_KEYWORDS):
                        print(f"[AUDIO] Selected output (analog): [{idx}] {dev.get('name')}", flush=True)
                        return idx

            for idx, dev in enumerate(devices):
                name = dev.get("name", "").lower()
                if dev.get("max_output_channels", 0) > 0 and "hdmi" not in name and "sysdefault" not in name and "default" not in name:
                    print(f"[AUDIO] Selected output (fallback): [{idx}] {dev.get('name')}", flush=True)
                    return idx
        except Exception as e:
            print(f"[AUDIO] Output device scan failed: {e}", flush=True)
        return None

    def speak(self, text):
        import json as _json
        text = text.encode('ascii', 'ignore').decode('ascii')
        clean = re.sub(r'[^a-zA-Z0-9\s,.!?:\'\-]', '', text).strip()
        if not clean:
            return

        print(f"[PIPER] Synthesis text: '{clean}'", flush=True)

        if self._piper_process is None or self._piper_process.poll() is not None:
            print("[PIPER] Piper process died, restarting...", flush=True)
            self._init_piper()
        if self._piper_process is None:
            return

        try:
            msg = _json.dumps({"text": clean}) + "\n"
            self._piper_process.stdin.write(msg.encode())
            self._piper_process.stdin.flush()

            playback_rate = self._piper_native_rate if self._piper_use_native_rate else self._piper_rate

            with sd.RawOutputStream(samplerate=playback_rate, channels=1, dtype='int16',
                                    device=self._output_device, latency='high', blocksize=2048) as stream:
                audio_started = False
                pre_audio_deadline = time.time() + 3.0
                post_audio_gap = 0
                MAX_GAP_TICKS = 25  # 25 * 20ms = 500ms silence flags end of phrase

                while True:
                    if self.interrupted.is_set():
                        break

                    if not audio_started and time.time() > pre_audio_deadline:
                        print("[PIPER] Synthesis timeout — dropping sentence.", flush=True)
                        break

                    rlist, _, _ = select.select([self._piper_process.stdout], [], [], 0.02)
                    if not rlist:
                        if audio_started:
                            post_audio_gap += 1
                            if post_audio_gap >= MAX_GAP_TICKS:
                                break
                        continue

                    post_audio_gap = 0
                    # Use read1 to read whatever is available in the buffer immediately without blocking,
                    # which prevents stuttering or delayed audio processing in the streaming pipeline.
                    data = self._piper_process.stdout.read1(4096)
                    if not data:
                        if audio_started:
                            break
                        continue

                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if len(audio_chunk) > 0:
                        if not audio_started:
                            audio_started = True
                            self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.current_volume = np.max(np.abs(audio_chunk))
                        
                        if self._piper_use_native_rate:
                            # Downsample polyphase filtering
                            from math import gcd
                            g = gcd(self._piper_native_rate, self._piper_rate)
                            up, down = self._piper_native_rate // g, self._piper_rate // g
                            audio_chunk = scipy.signal.resample_poly(audio_chunk, up, down).astype(np.int16)
                        stream.write(audio_chunk.tobytes())

            self.current_volume = 0
        except Exception as e:
            print(f"[PIPER ERROR] Synthesis failure: {e}", flush=True)
            self._piper_process = None

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set():
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound:
                self.play_sound(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set():
                    return
                time.sleep(0.1)

    def get_random_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            return os.path.join(directory, random.choice(files)) if files else None
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except:
                native_rate = 48000

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except:
                playback_rate = native_rate
                num_samples = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate)
            sd.wait()
        except:
            pass

    # --- MEMORY STACK LOAD/SAVE ---

    def load_chat_history(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    history = json.load(f)
                    if history and len(history) > 0:
                        history[0] = {"role": "system", "content": SYSTEM_PROMPT}
                        return history
            except:
                pass
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def save_chat_history(self):
        full = self.permanent_memory + self.session_memory
        conv = full[1:]
        if len(conv) > 10:
            conv = conv[-10:]
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump([full[0]] + conv, f, indent=4)
        except Exception as e:
            print(f"[MEM ERROR] Save chat history failed: {e}", flush=True)

if __name__ == "__main__":
    print("--- SOCIAL ROBOT BMO SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
