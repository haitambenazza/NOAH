# =========================================================================
#  Be More Agent — BMO Unified Controller 🤖
#  A Local, Offline-First AI Agent & Face Tracker for Raspberry Pi 4
# =========================================================================

import os
import sys

# Ensure system apt packages (PIL/ImageTk, tkinter, libcamera, picamera2, lgpio)
# are visible even inside a virtualenv created without --system-site-packages.
_SYS_DIST_PACKAGES = "/usr/lib/python3/dist-packages"
if os.path.isdir(_SYS_DIST_PACKAGES) and _SYS_DIST_PACKAGES not in sys.path:
    sys.path.append(_SYS_DIST_PACKAGES)

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import subprocess
import random
import re
import select
import traceback
import atexit
import datetime
import warnings
import wave
import queue
from collections import deque

# Tune CPU thread pools for Pi 5 (now that LLM runs on Hailo-10H NPU)
# and cap Hugging Face Hub timeouts so Whisper doesn't stall on Wi-Fi/DNS.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

# Suppress harmless library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ddgs")

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
    print("[INIT INFO] openwakeword not installed — using Whisper wake word detection.")

import requests

# --- SPEECH TO TEXT ---
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[INIT WARNING] faster-whisper not installed — STT transcription will be disabled.")

# --- WEB SEARCH ---
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False
        print("[INIT WARNING] ddgs/duckduckgo-search not installed — Web search action will be disabled.")

# --- COMPUTER VISION ---
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[INIT WARNING] opencv-python not installed — Camera & Face Detection disabled.")

# --- RASPBERRY PI HARDWARE ---
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except Exception as _picam_err:
    PICAMERA_AVAILABLE = False
    print(f"[INIT INFO] picamera2 unavailable ({_picam_err}) — Will fall back to OpenCV VideoCapture.")

try:
    from libcamera import Transform as LibcameraTransform
except Exception:
    LibcameraTransform = None

try:
    from gpiozero import Servo
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[INIT INFO] gpiozero not installed — Will use Mock Servos.")

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_TTY = bool(getattr(sys.stdout, "isatty", lambda: False)())


def _resolve_path(path):
    """Resolves relative paths against the script directory rather than cwd."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


CONFIG_FILE = _resolve_path("config.json")
MEMORY_FILE = _resolve_path("memory.json")
WAKE_WORD_MODEL = _resolve_path("wakeword.onnx")
WAKE_WORD_THRESHOLD = 0.3
DEFAULT_WAKE_WORDS = ["hey noah", "wake up", "get up"]

# HARDWARE SETTINGS
INPUT_DEVICE_NAME = None

DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:8000",
    "text_model": "qwen2.5-instruct:1.5b",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "wake_word_engine": "whisper",
    "wake_words": list(DEFAULT_WAKE_WORDS),
    "chat_memory": True,
    "system_prompt_extras": "",
    "input_device": None,
    "input_sample_rate": None,
    "silence_duration": 0.9,
    "no_speech_timeout": 4.0,
    "max_record_time": 10.0,
    "servo_gpio1": 12,
    "servo_gpio2": 13,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 15,
    "camera_flip": True,
    "use_yunet": True,
    "debug_save_wav": False,
    "knowledge_dir": "knowledge",
    "knowledge_top_k": 3,
    "max_history_turns": 6,
}

# LLM SETTINGS (Hailo-10H NPU via hailo-ollama)
OLLAMA_OPTIONS = {
    'temperature': 0.35,
    'top_k': 20,
    'top_p': 0.85,
    'repeat_penalty': 1.15,
}
OLLAMA_FACTUAL_OPTIONS = {
    'temperature': 0.1,
    'top_k': 10,
    'top_p': 0.8,
    'repeat_penalty': 1.15,
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

def _highpass_filter(samples_float32, samplerate, cutoff_hz=150.0):
    """Removes low-frequency fan/USB rumble (<150 Hz) before VAD and Whisper."""
    if len(samples_float32) < 16 or samplerate <= cutoff_hz * 2:
        return samples_float32
    try:
        sos = scipy.signal.butter(2, cutoff_hz, btype="highpass", fs=samplerate, output="sos")
        return scipy.signal.sosfiltfilt(sos, samples_float32).astype(np.float32)
    except Exception:
        return samples_float32

def _normalize_wake_words(wake_words):
    """Returns a cleaned list of lowercase punctuation-stripped wake phrases."""
    if not isinstance(wake_words, (list, tuple, set)):
        return list(DEFAULT_WAKE_WORDS)
    cleaned = []
    seen = set()
    for item in wake_words:
        if not isinstance(item, str):
            continue
        stripped = re.sub(r"[^a-z0-9'\s]", " ", item.lower())
        norm = " ".join(stripped.split())
        if norm and norm not in seen:
            cleaned.append(norm)
            seen.add(norm)
    return cleaned if cleaned else list(DEFAULT_WAKE_WORDS)

def _format_wake_prompt(wake_words):
    """Formats wake phrases into a Whisper prompt hint."""
    phrases = _normalize_wake_words(wake_words)
    formatted = []
    for phrase in phrases:
        words = phrase.split()
        titled = []
        for idx, word in enumerate(words):
            if word == "noah":
                titled.append("Noah")
            elif word == "bmo":
                titled.append("BMO")
            elif idx == 0:
                titled.append(word.capitalize())
            else:
                titled.append(word)
        formatted.append(" ".join(titled))
    return ". ".join(formatted) + "."

def _build_wake_patterns(wake_words):
    """Builds compiled regex patterns (and safe phonetic aliases) for wake phrases."""
    patterns = []
    for phrase in _normalize_wake_words(wake_words):
        words = phrase.split()
        exact_pat = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"
        patterns.append((phrase, re.compile(exact_pat, re.IGNORECASE)))
        if phrase == "hey noah":
            for alias_pat in (
                r"\bhey\s+noa+h?\b",
                r"\bhey\s+noer\b",
                r"\bhey\s+know\s+a\b",
            ):
                patterns.append((phrase, re.compile(alias_pat, re.IGNORECASE)))
    return patterns

def _match_wake_phrase(text, wake_patterns):
    """Matches a wake phrase and extracts any trailing spoken command."""
    if not text or not wake_patterns:
        return (None, "")
    normalized = " ".join(re.sub(r"[^a-z0-9'\s]", " ", text.lower()).split())
    if not normalized:
        return (None, "")

    best_phrase = None
    best_match = None
    for phrase, pattern in wake_patterns:
        match = pattern.search(normalized)
        if match is None:
            continue
        if (
            best_match is None
            or match.start() < best_match.start()
            or (match.start() == best_match.start() and match.end() > best_match.end())
        ):
            best_phrase = phrase
            best_match = match

    if best_match is None:
        return (None, "")

    trailing = normalized[best_match.end():].strip()
    trailing = re.sub(r"^(?:noah|noa)\b[\s,]*", "", trailing).strip()
    filler_words = {
        "noah", "noa", "bmo", "please", "now",
        "already", "buddy", "robot", "um", "uh", "so",
    }
    if not trailing or trailing in filler_words or (" " not in trailing and len(trailing) < 4):
        trailing = ""
    return (best_phrase, trailing)

class HailoOllamaClient:
    """Minimal REST client compatible with hailo-ollama (Oat++ v1.4.0) and standard Ollama."""

    def __init__(self, host="http://localhost:8000", timeout=(10, 120)):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._supports_options = True
        self.session = requests.Session()

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass

    def _post_json(self, endpoint, payload, stream=False, timeout=None):
        url = f"{self.host}{endpoint}"
        req_timeout = self.timeout if timeout is None else timeout
        resp = self.session.post(url, json=payload, stream=stream, timeout=req_timeout)
        if resp.status_code in (400, 422) and "options" in payload:
            resp.close()
            print("[LLM] Server rejected 'options' payload; retrying with minimal schema.", flush=True)
            minimal_payload = {k: v for k, v in payload.items() if k != "options"}
            resp = self.session.post(url, json=minimal_payload, stream=stream, timeout=req_timeout)
            if resp.status_code < 400:
                self._supports_options = False

        if resp.status_code >= 400:
            err_body = resp.text[:400]
            resp.close()
            raise RuntimeError(f"HTTP {resp.status_code} from {endpoint}: {err_body}")
        return resp

    @staticmethod
    def _extract_model_names(data):
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("models") or data.get("data") or []
        else:
            return []
        names = []
        for item in items:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
            elif isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
        return names

    def list_installed_models(self):
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=5)
            try:
                if resp.status_code >= 400:
                    return []
                return self._extract_model_names(resp.json())
            finally:
                resp.close()
        except Exception:
            return []

    def list_available_models(self):
        try:
            resp = self.session.get(f"{self.host}/hailo/v1/list", timeout=5)
            try:
                if resp.status_code >= 400:
                    return []
                return self._extract_model_names(resp.json())
            finally:
                resp.close()
        except Exception:
            return []

    def pull(self, model):
        resp = self.session.post(
            f"{self.host}/api/pull",
            json={"model": str(model), "stream": False},
            timeout=(10, 600),
        )
        try:
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} from /api/pull: {resp.text[:400]}")
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"LLM server error: {data['error']}")
            return data
        finally:
            resp.close()

    def chat(self, model, messages, stream=False, options=None):
        clean_messages = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
            if m.get("content") is not None and str(m.get("content", "")).strip() != ""
        ]
        max_msgs = max(2, int(CURRENT_CONFIG.get("max_history_turns", 6)) * 2)
        if len(clean_messages) > (max_msgs + 1) and clean_messages[0]["role"] == "system":
            clean_messages = [clean_messages[0]] + clean_messages[-max_msgs:]
        elif len(clean_messages) > max_msgs and (not clean_messages or clean_messages[0]["role"] != "system"):
            clean_messages = clean_messages[-max_msgs:]

        # Keep prompt safely inside Hailo-10H's compiled 2048-token context window
        max_prompt_chars = 5600
        start_idx = 1 if (clean_messages and clean_messages[0]["role"] == "system") else 0
        while len(clean_messages) - start_idx > 2 and sum(len(m["content"]) for m in clean_messages) > max_prompt_chars:
            clean_messages.pop(start_idx)
            if len(clean_messages) - start_idx > 2:
                clean_messages.pop(start_idx)

        payload = {
            "model": str(model),
            "messages": clean_messages,
            "stream": bool(stream),
        }
        if options and self._supports_options:
            payload["options"] = options

        resp = self._post_json("/api/chat", payload, stream=bool(stream))
        if stream:
            def _iter_chunks():
                try:
                    for line in resp.iter_lines(chunk_size=512, decode_unicode=True):
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="ignore")
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                            if not line or line == "[DONE]":
                                continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as je:
                            print(f"[LLM WARNING] Non-JSON stream line ({je}): {line[:120]!r}", flush=True)
                            continue
                        if isinstance(obj, dict) and obj.get("error"):
                            raise RuntimeError(f"LLM server error: {obj['error']}")
                        yield obj
                finally:
                    resp.close()

            return _iter_chunks()

        try:
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"LLM server error: {data['error']}")
            return data
        finally:
            resp.close()

    def generate(self, model, prompt="", keep_alive=None, stream=False, options=None, timeout=None):
        payload = {
            "model": str(model),
            "prompt": str(prompt),
            "stream": bool(stream),
        }
        if keep_alive is not None:
            payload["keep_alive"] = str(keep_alive)
        if options and self._supports_options:
            payload["options"] = options

        resp = self._post_json("/api/generate", payload, stream=False, timeout=timeout)
        try:
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"LLM server error: {data['error']}")
            return data
        finally:
            resp.close()

CURRENT_CONFIG = load_config()
TEXT_MODEL = CURRENT_CONFIG["text_model"]
OLLAMA_HOST = CURRENT_CONFIG.get("ollama_host", "http://localhost:8000")
OLLAMA_CLIENT = HailoOllamaClient(host=OLLAMA_HOST)
print(f"[LLM] Configured Ollama client: host={OLLAMA_HOST} model={TEXT_MODEL}", flush=True)

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

# --- DYNAMIC KNOWLEDGE & PROMPT LOADER ---
FALLBACK_SYSTEM_PROMPT = "You are a helpful voice assistant. Reply in short plain-text sentences."


class KnowledgeBase:
    """Hot-reloading Markdown knowledge loader: pinned core.md + BM25 section retrieval."""

    _STOPWORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
        "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
        "is", "it", "its", "of", "on", "or", "our", "she", "that", "the",
        "their", "them", "they", "this", "to", "was", "we", "were", "what",
        "when", "where", "which", "with", "you", "your",
    })
    _SELF_TRIGGERS = frozenset({"i", "me", "my", "mine", "myself", "who"})
    _SELF_EXPANSION = ("owner", "profile", "projects", "interests", "preferences")

    def __init__(self, knowledge_dir, core_filename="core.md", max_core_chars=1400, max_context_chars=1100):
        self.knowledge_dir = knowledge_dir
        self.core_filename = core_filename
        self.max_core_chars = max_core_chars
        self.max_context_chars = max_context_chars
        self._lock = threading.Lock()
        self._mtimes = {}
        self._core_text = ""
        self._chunks = []

    @staticmethod
    def _strip_comments(text):
        return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()

    @classmethod
    def _tokenize(cls, text):
        tokens = []
        for w in re.findall(r"[a-z0-9]+", str(text).lower()):
            if w in cls._STOPWORDS:
                continue
            if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
                w = w[:-1]
            tokens.append(w)
        return tokens

    @staticmethod
    def _split_markdown_sections(file_stem, text):
        sections = []
        current_heading = file_stem
        buf = []
        for line in text.splitlines():
            heading_m = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
            if heading_m:
                body = "\n".join(buf).strip()
                if body:
                    sections.append((current_heading, body))
                title = heading_m.group(2).strip()
                current_heading = title if title.lower() == file_stem.lower() else f"{file_stem} > {title}"
                buf = []
            else:
                buf.append(line)
        body = "\n".join(buf).strip()
        if body:
            sections.append((current_heading, body))
        return sections

    def _scan_files(self):
        if not self.knowledge_dir or not os.path.isdir(self.knowledge_dir):
            return {}
        found = {}
        for root, _, files in os.walk(self.knowledge_dir):
            for fname in sorted(files):
                if fname.lower().endswith(".md"):
                    fpath = os.path.join(root, fname)
                    try:
                        found[fpath] = os.path.getmtime(fpath)
                    except OSError:
                        pass
        return found

    def _reload_if_needed(self):
        current_mtimes = self._scan_files()
        if current_mtimes == self._mtimes:
            return
        core_text = ""
        chunks = []
        core_lower = self.core_filename.lower()
        for fpath in sorted(current_mtimes):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = self._strip_comments(f.read())
            except OSError:
                continue
            if not raw:
                continue
            rel = os.path.relpath(fpath, self.knowledge_dir)
            if rel.replace("\\", "/").lower() == core_lower:
                core_text = raw[:self.max_core_chars].strip()
                continue
            stem = os.path.splitext(os.path.basename(fpath))[0].replace("_", " ").replace("-", " ").title()
            for heading, body in self._split_markdown_sections(stem, raw):
                h_tokens = self._tokenize(heading)
                b_tokens = self._tokenize(body)
                all_tokens = h_tokens + b_tokens
                if all_tokens:
                    chunks.append({
                        "title": heading,
                        "body": body,
                        "title_tokens": set(h_tokens),
                        "tokens": all_tokens,
                    })
        self._mtimes = current_mtimes
        self._core_text = core_text
        self._chunks = chunks
        print(f"[KNOWLEDGE] Loaded core={len(core_text)} chars, sections={len(chunks)} from {self.knowledge_dir}", flush=True)

    def get_core_prompt(self):
        with self._lock:
            self._reload_if_needed()
            return self._core_text

    def retrieve(self, query, top_k=3):
        if not query or top_k <= 0:
            return []
        with self._lock:
            self._reload_if_needed()
            chunks = list(self._chunks)
        if not chunks:
            return []

        import math
        q_raw = re.findall(r"[a-z0-9]+", str(query).lower())
        q_tokens = self._tokenize(query)
        if any(tok in self._SELF_TRIGGERS for tok in q_raw):
            q_tokens.extend(self._SELF_EXPANSION)
        if not q_tokens:
            return []

        n_docs = len(chunks)
        avg_len = sum(len(c["tokens"]) for c in chunks) / max(1, n_docs)
        df = {}
        for c in chunks:
            for tok in set(c["tokens"]):
                df[tok] = df.get(tok, 0) + 1

        k1, b = 1.5, 0.75
        scored = []
        for idx, c in enumerate(chunks):
            tf_map = {}
            for tok in c["tokens"]:
                tf_map[tok] = tf_map.get(tok, 0) + 1
            doc_len = len(c["tokens"])
            score = 0.0
            for q in set(q_tokens):
                tf = tf_map.get(q, 0)
                if tf > 0:
                    idf = math.log(1.0 + (n_docs - df[q] + 0.5) / (df[q] + 0.5))
                    denom = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_len)))
                    score += idf * ((tf * (k1 + 1.0)) / denom)
                if q in c["title_tokens"]:
                    score += 2.5
            if score > 0.0:
                scored.append((score, -idx, c))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = []
        used_chars = 0
        for _, _, c in scored[:top_k]:
            entry = f"[{c['title']}]\n{c['body']}"
            if used_chars + len(entry) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining > 80:
                    selected.append(entry[:remaining].rsplit(" ", 1)[0] + "...")
                break
            selected.append(entry)
            used_chars += len(entry) + 2
        return selected

    def build_system_prompt(self, query="", top_k=3, extras=""):
        core = self.get_core_prompt() or FALLBACK_SYSTEM_PROMPT
        parts = [core]
        if extras and str(extras).strip():
            parts.append(str(extras).strip())
        parts.append(f"Today's date: {datetime.date.today().isoformat()}.")
        snippets = self.retrieve(query, top_k=top_k)
        if snippets:
            parts.append("# Relevant Knowledge\n" + "\n\n".join(snippets))
        return "\n\n".join(parts)


KNOWLEDGE_BASE = KnowledgeBase(_resolve_path(CURRENT_CONFIG.get("knowledge_dir", "knowledge")))

# Sound Directories
greeting_sounds_dir = _resolve_path("sounds/greeting_sounds")
ack_sounds_dir = _resolve_path("sounds/ack_sounds")
thinking_sounds_dir = _resolve_path("sounds/thinking_sounds")
error_sounds_dir = _resolve_path("sounds/error_sounds")

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
        self._stopped = True
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
            except Exception:
                pass
        self.stop_robot()

    def stop_robot(self):
        try:
            s1_none = (self.servo1 is None or self.servo1.value is None)
            s2_none = (self.servo2 is None or self.servo2.value is None)
            if self._stopped and s1_none and s2_none:
                return
            if self.servo1 and self.servo1.value is not None:
                self.servo1.value = None
            if self.servo2 and self.servo2.value is not None:
                self.servo2.value = None
            self._stopped = True
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

                # Drain stale motion commands while never dropping an exit command
                if cmd[0] != 'exit':
                    while not self.queue.empty():
                        try:
                            next_cmd = self.queue.get_nowait()
                        except queue.Empty:
                            break
                        cmd = next_cmd
                        if cmd[0] == 'exit':
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
            self._stopped = False
        else:
            self.stop_robot()

# =========================================================================
# 3.5 OBJECT DETECTION PIPELINE (Lightweight MobileNet-SSD)
# =========================================================================

class ObjectDetector:
    def __init__(self):
        self.model_dir = _resolve_path("models")
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
            for url, path, label in (
                (self.proto_url, self.proto_path, "prototxt"),
                (self.model_url, self.model_path, "caffemodel weights"),
            ):
                if not os.path.exists(path):
                    print(f"[VISION] Downloading {label}: {url}...", flush=True)
                    r = requests.get(url, timeout=20)
                    r.raise_for_status()
                    tmp_path = path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        f.write(r.content)
                    os.replace(tmp_path, path)
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
    YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    def __init__(self, motor_controller, width=640, height=480, fps=15, camera_flip=True, use_yunet=True):
        self.motor_controller = motor_controller
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.camera_flip = bool(camera_flip)
        self.use_yunet = bool(use_yunet)
        self.running = False
        self.thread = None

        self.face_detection_enabled = True
        self.preview_enabled = False
        self.preview_callback = None  # Receives RGB image when preview is active
        self.latest_frame = None
        self._frame_lock = threading.Lock()

        self.cap = None
        self.picam = None
        self._picam_isp_ready = False
        self._picam_needs_sw_flip = False

        # Face detectors (YuNet preferred when available, Haar Cascade fallback)
        self.yunet = None
        self.face_cascade = None
        self.init_cascade()

    def get_latest_frame(self):
        with self._frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def init_cascade(self):
        if not CV2_AVAILABLE:
            return
        if self.use_yunet and hasattr(cv2, "FaceDetectorYN"):
            yunet_path = _resolve_path("models/face_detection_yunet_2023mar.onnx")
            if not os.path.exists(yunet_path):
                try:
                    os.makedirs(os.path.dirname(yunet_path), exist_ok=True)
                    print("[CAMERA] Downloading YuNet face detector...", flush=True)
                    r = requests.get(self.YUNET_URL, timeout=(3, 6))
                    r.raise_for_status()
                    tmp_path = yunet_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        f.write(r.content)
                    os.replace(tmp_path, yunet_path)
                except Exception as e:
                    print(f"[CAMERA INFO] YuNet unavailable offline ({e}) — using Haar cascade.", flush=True)
            if os.path.exists(yunet_path):
                try:
                    self.yunet = cv2.FaceDetectorYN.create(
                        yunet_path,
                        "",
                        (max(1, self.width // 2), max(1, self.height // 2)),
                        0.6,
                        0.3,
                        5000,
                    )
                    print(f"[CAMERA] Loaded YuNet face detector from: {yunet_path}", flush=True)
                except Exception as e:
                    print(f"[CAMERA WARNING] Failed loading YuNet ({e}) — falling back to Haar.", flush=True)
                    self.yunet = None

        cascade_paths = [
            _resolve_path('haarcascade_frontalface_default.xml'),
            _resolve_path('tools/haarcascade_frontalface_default.xml'),
            os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
        ]
        for path in cascade_paths:
            if os.path.exists(path):
                try:
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        self.face_cascade = cascade
                        print(f"[CAMERA] Loaded cascade classifier from: {path}", flush=True)
                        break
                except Exception as e:
                    print(f"[CAMERA ERROR] Error loading cascade from {path}: {e}", flush=True)
        if self.yunet is None and self.face_cascade is None:
            print("[CAMERA WARNING] No face detector loaded. Face tracking will be disabled.", flush=True)

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
                    use_hw_flip = bool(self.camera_flip and LibcameraTransform is not None)
                    transform = LibcameraTransform(hflip=1, vflip=1) if use_hw_flip else None
                    isp_candidates = [
                        {"main": {"size": (self.width, self.height), "format": "RGB888"}, "raw": {"size": (1640, 1232)}},
                        {"main": {"size": (self.width, self.height), "format": "RGB888"}},
                    ]
                    configured = False
                    for cfg_kwargs in isp_candidates:
                        try:
                            if transform is not None:
                                cfg_kwargs["transform"] = transform
                            config = self.picam.create_preview_configuration(**cfg_kwargs)
                            self.picam.configure(config)
                            self.picam.start()
                            self._picam_isp_ready = True
                            self._picam_needs_sw_flip = bool(self.camera_flip and not use_hw_flip)
                            configured = True
                            print(
                                f"[CAMERA] Initialized Picamera2 ISP ({self.width}x{self.height} BGR at {self.fps} FPS, hw_flip={use_hw_flip}).",
                                flush=True,
                            )
                            break
                        except Exception:
                            continue
                    if not configured:
                        config = self.picam.create_preview_configuration(
                            main={"size": (1640, 1232), "format": "BGR888"}
                        )
                        self.picam.configure(config)
                        self.picam.start()
                        self._picam_isp_ready = False
                        self._picam_needs_sw_flip = self.camera_flip
                        print(f"[CAMERA] Initialized Picamera2 legacy mode (1640x1232 at {self.fps} FPS).", flush=True)
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
            except Exception:
                pass
            self.picam = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print("[CAMERA] Camera hardware released.", flush=True)

    def _detect_faces(self, frame):
        """Detects faces using YuNet (preferred) or Haar Cascade fallback."""
        faces = []
        if self.yunet is not None:
            scale = 2
            small_w = max(1, frame.shape[1] // scale)
            small_h = max(1, frame.shape[0] // scale)
            small_bgr = cv2.resize(frame, (small_w, small_h))
            self.yunet.setInputSize((small_w, small_h))
            _, raw_faces = self.yunet.detect(small_bgr)
            if raw_faces is not None:
                for row in raw_faces:
                    x, y, w, h = (int(row[0] * scale), int(row[1] * scale), int(row[2] * scale), int(row[3] * scale))
                    if w > 0 and h > 0:
                        faces.append((x, y, w, h))
            return faces

        if self.face_cascade is not None:
            scale = 4
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=(1.0 / scale), fy=(1.0 / scale))
            detected = self.face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in detected:
                faces.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale)))
        return faces

    def _run_loop(self):
        if not self._init_camera():
            print("[CAMERA ERROR] Camera pipeline thread exiting: init failed.", flush=True)
            self.running = False
            return

        frame_delay = 1.0 / self.fps
        last_face_time = 0.0
        face_lost_delay = 0.5  # Settle time before stopping motors when losing target
        motors_moving = False

        while self.running:
            start_time = time.time()
            frame = None

            try:
                if self.picam:
                    frame_array = self.picam.capture_array()
                    if self._picam_isp_ready:
                        frame = frame_array
                        if self._picam_needs_sw_flip:
                            frame = cv2.flip(frame, -1)
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))
                    else:
                        frame_raw = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                        if self._picam_needs_sw_flip:
                            frame_raw = cv2.flip(frame_raw, -1)
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

            # Keep latest_frame unmutated for downstream object detection
            with self._frame_lock:
                self.latest_frame = frame

            best_face = None
            if self.face_detection_enabled and (self.yunet is not None or self.face_cascade is not None):
                faces = self._detect_faces(frame)
                if faces:
                    best_face = max(faces, key=lambda r: r[2] * r[3])
                    x, y, w, h = best_face
                    face_center_x = x + (w // 2)
                    last_face_time = time.time()

                    frame_half_w = max(1, frame.shape[1] // 2)
                    error_x = (face_center_x - frame_half_w) / frame_half_w
                    self.motor_controller.set_move(error_x)
                    motors_moving = True

            if best_face is None and motors_moving:
                if time.time() - last_face_time > face_lost_delay:
                    self.motor_controller.trigger_stop()
                    motors_moving = False

            # Render overlay onto a separate copy if Picture-in-Picture preview is enabled
            if self.preview_enabled and self.preview_callback:
                preview_frame = frame.copy()
                h_px, w_px = preview_frame.shape[:2]
                if best_face is not None:
                    fx, fy, fw, fh = best_face
                    cv2.rectangle(preview_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                    cv2.circle(preview_frame, (fx + (fw // 2), fy + (fh // 2)), 5, (0, 0, 255), -1)
                cv2.line(preview_frame, ((w_px // 2) - 20, h_px // 2), ((w_px // 2) + 20, h_px // 2), (0, 255, 0), 2)
                cv2.line(preview_frame, (w_px // 2, (h_px // 2) - 20), (w_px // 2, (h_px // 2) + 20), (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
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
        master.protocol("WM_DELETE_WINDOW", self.safe_exit)
        atexit.register(self.safe_exit)
        
        # Core State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0
        self.animations = {}
        self.current_frame_index = 0
        
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.chat_memory_enabled = bool(CURRENT_CONFIG.get("chat_memory", True))
        self.thinking_sound_active = threading.Event()
        self._thinking_thread = None

        self.last_ptt_time = 0
        self.last_wake_time = 0
        self._auto_stop_timer = None
        self._auto_stop_lock = threading.Lock()
        self._record_gen = 0
        self.ptt_auto_stop_duration = None
        self.ptt_event = threading.Event()
        self.recording_active = threading.Event()
        self.interrupted = threading.Event()
        self.stop_event = threading.Event()

        self.tts_queue = deque()
        self.tts_queue_lock = threading.Lock()
        self.tts_cond = threading.Condition(self.tts_queue_lock)
        self._audio_out_lock = threading.Lock()
        self._sfx_abort = threading.Event()
        self.tts_thread = None
        self.agent_thread = None
        self.tts_active = threading.Event()
        self.idle_event = threading.Event()
        self.auto_listen = True
        self._auto_followup = False
        self._last_assistant_reply_norm = ""
        self._last_user_text_norm = ""
        self._ambient_noise_floor = 0.008
        self._cam_preview_pending = False
        self.motor_controller = None
        self.camera_pipeline = None
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
            fps=CURRENT_CONFIG.get("camera_fps", 15),
            camera_flip=CURRENT_CONFIG.get("camera_flip", True),
            use_yunet=CURRENT_CONFIG.get("use_yunet", True),
        )
        self.camera_pipeline.start()

        # --- WAKE WORD INITIALIZATION ---
        print("[INIT] Loading Wake Word...", flush=True)
        self.wake_word_engine = str(CURRENT_CONFIG.get("wake_word_engine", "whisper")).strip().lower()
        self.wake_words = _normalize_wake_words(CURRENT_CONFIG.get("wake_words", DEFAULT_WAKE_WORDS))
        self._wake_patterns = _build_wake_patterns(self.wake_words)
        self._pending_wake_command = ""
        self.oww_model = None
        if self.wake_word_engine == "openwakeword":
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
                print("[CRITICAL] Wake word model file not found or package unavailable.")
        else:
            print(f"[INIT] Wake Word Engine: Whisper phrases={self.wake_words}", flush=True)

        # --- SPEECH TO TEXT INITIALIZATION ---
        print("[INIT] Loading Whisper model...", flush=True)
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = WhisperModel(
                    "tiny.en",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                    local_files_only=True,
                )
                print("[INIT] Whisper ready (local cache).", flush=True)
            except Exception as local_err:
                print(f"[INIT] Whisper local cache miss ({local_err}); downloading...", flush=True)
                try:
                    self.whisper_model = WhisperModel(
                        "tiny.en",
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=4,
                    )
                    print("[INIT] Whisper ready (downloaded).", flush=True)
                except Exception as e:
                    print(f"[CRITICAL] Whisper failed to load: {e}", flush=True)
        else:
            print("[CRITICAL] Whisper STT unavailable.")

        # --- AUDIO INPUT & OUTPUT SETUP ---
        self._input_samplerate = choose_input_samplerate(
            INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate")
        )
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
        self.agent_thread = threading.Thread(target=self.safe_main_execution, daemon=True, name="AgentLoopThread")
        self.agent_thread.start()

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
        """Resizes preview frame on camera thread and builds PhotoImage on Tk main thread."""
        if self._cam_preview_pending or self.exiting or not self.camera_pipeline.preview_enabled:
            return
        try:
            img = Image.fromarray(frame_rgb).resize((self.CAM_W, self.CAM_H), Image.BILINEAR)
            self._cam_preview_pending = True

            def _update(pil_img=img):
                self._cam_preview_pending = False
                if self.exiting or not self.camera_pipeline.preview_enabled:
                    return
                try:
                    photo = ImageTk.PhotoImage(pil_img)
                    self._cam_photo = photo  # Keep alive
                    self.cam_label.config(image=photo)
                except tk.TclError:
                    pass

            self.master.after(0, _update)
        except Exception as e:
            self._cam_preview_pending = False
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

    def _cancel_auto_stop_timer(self):
        with self._auto_stop_lock:
            self._record_gen += 1
            if self._auto_stop_timer is not None:
                self._auto_stop_timer.cancel()
                self._auto_stop_timer = None

    def _start_auto_stop_timer(self, duration=5.0):
        with self._auto_stop_lock:
            if self._auto_stop_timer is not None:
                self._auto_stop_timer.cancel()
            self._record_gen += 1
            gen = self._record_gen

            def _stop_if_current():
                with self._auto_stop_lock:
                    if self._record_gen == gen:
                        self.recording_active.clear()
                        self._auto_stop_timer = None

            timer = threading.Timer(duration, _stop_if_current)
            timer.daemon = True
            self._auto_stop_timer = timer
            timer.start()

    def handle_ptt_toggle(self, event=None):
        current_time = time.time()
        if current_time - self.last_ptt_time < 0.5:
            return
        self.last_ptt_time = current_time

        if self.recording_active.is_set():
            print("[PTT] Toggle OFF", flush=True)
            self._cancel_auto_stop_timer()
            self.recording_active.clear()
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                print("[PTT] Toggle ON", flush=True)
                self._cancel_auto_stop_timer()
                self.ptt_auto_stop_duration = None
                self.idle_event.clear()
                self.recording_active.set()
                self.ptt_event.set()

    def handle_wake_button(self):
        now = time.time()
        if now - self.last_wake_time < 0.5:
            return
        self.last_wake_time = now

        if self.current_state == BotStates.LISTENING and self.recording_active.is_set():
            print("[WAKE BTN] Already listening — ignoring duplicate trigger.", flush=True)
            return

        print("[WAKE BTN] Manual wake triggered", flush=True)
        self._cancel_auto_stop_timer()
        self.idle_event.clear()
        self.interrupted.clear()
        self.ptt_auto_stop_duration = 5.0
        self.recording_active.set()
        self.ptt_event.set()

    def handle_idle_button(self):
        print("[IDLE BTN] Forcing idle", flush=True)
        self._auto_followup = False
        self._cancel_auto_stop_timer()
        self.idle_event.set()
        self.interrupted.set()
        self._sfx_abort.set()
        self.thinking_sound_active.clear()
        self.recording_active.clear()
        self.ptt_event.clear()
        self._clear_tts_queue()
        self.set_state(BotStates.IDLE, "Idle.")

    def handle_speaking_interrupt(self, event=None):
        if self.current_state in (BotStates.SPEAKING, BotStates.THINKING):
            self.interrupted.set()
            self._sfx_abort.set()
            self.thinking_sound_active.clear()
            self._clear_tts_queue()
            self.set_state(BotStates.IDLE, "Interrupted.")

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.safe_exit()

    def safe_exit(self):
        if self.exiting:
            return
        self.exiting = True
        self.stop_event.set()
        self.interrupted.set()
        self.idle_event.set()
        self._sfx_abort.set()
        self._cancel_auto_stop_timer()
        print("\n--- SHUTDOWN SEQUENCE STARTED ---", flush=True)

        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self._clear_tts_queue()
        with self.tts_cond:
            self.tts_active.clear()
            self.tts_cond.notify_all()
        self.ptt_event.set()  # Unblock any thread waiting on ptt_event

        # Wait for audio/agent worker threads to exit their streams BEFORE tearing down PortAudio
        current = threading.current_thread()
        for t, wait_s in (
            (getattr(self, "agent_thread", None), 2.0),
            (getattr(self, "tts_thread", None), 1.5),
            (getattr(self, "_thinking_thread", None), 1.0),
        ):
            if t and t.is_alive() and t is not current:
                t.join(timeout=wait_s)

        self._kill_piper()

        # Stop sub-threads
        if self.camera_pipeline:
            self.camera_pipeline.stop()
        if self.motor_controller:
            self.motor_controller.stop()

        self.save_chat_history()

        # Shutdown Ollama connection keepalive with a short timeout
        try:
            OLLAMA_CLIENT.generate(model=TEXT_MODEL, prompt="", keep_alive=0, timeout=(1, 2))
        except Exception:
            pass
        OLLAMA_CLIENT.close()
        try:
            sd.stop()
        except Exception:
            pass

        try:
            self.master.quit()
            self.master.destroy()
        except Exception:
            pass

    # --- ANIMATION ENGINE ---

    def load_animations(self):
        base_path = _resolve_path("faces")
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
                if self.animations.get("idle"):
                    self.animations[state] = self.animations["idle"]
                else:
                    color_mapping = {
                        "warmup": "#FFFF00",
                        "listening": "#00FF00",
                        "thinking": "#FFA500",
                        "speaking": "#800080",
                        "error": "#FF0000"
                    }
                    fill = color_mapping.get(state, "#0000FF")
                    blank = Image.new('RGB', (self.BG_WIDTH, self.BG_HEIGHT), color=fill)
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(500, self.update_animation)
            return

        if self.current_state == BotStates.SPEAKING:
            if len(frames) <= 1 or self.current_volume < 350:
                self.current_frame_index = 0
            elif len(frames) == 2:
                self.current_frame_index = 1 if self.current_frame_index == 0 else 0
            else:
                self.current_frame_index = (self.current_frame_index % (len(frames) - 1)) + 1
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)

        try:
            self.background_label.config(image=frames[self.current_frame_index])
        except tk.TclError:
            return  # GUI closing

        speed = 50 if self.current_state == BotStates.SPEAKING else 500
        self.master.after(speed, self.update_animation)

    def set_state(self, state, msg=""):
        if self.exiting:
            return
        def _update():
            if self.exiting:
                return
            if msg:
                print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
                if state == BotStates.ERROR:
                    self.play_sound_async(self.get_random_sound(error_sounds_dir), interruptible=False)
            if msg:
                self.status_var.set(msg)
        try:
            self.master.after(0, _update)
        except (tk.TclError, RuntimeError):
            pass

    def _trim_response_text(self, max_lines=200):
        line_count = int(self.response_text.index("end-1c").split(".")[0])
        if line_count > max_lines:
            self.response_text.delete("1.0", f"{line_count - max_lines}.0")

    def append_to_text(self, text, newline=True):
        if self.exiting:
            return
        def _update():
            try:
                self.response_text.config(state=tk.NORMAL)
                if newline:
                    self.response_text.insert(tk.END, text + "\n")
                else:
                    self.response_text.insert(tk.END, text)
                self._trim_response_text()
                self.response_text.see(tk.END)
                self.response_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass
        try:
            self.master.after(0, _update)
        except (tk.TclError, RuntimeError):
            pass

    def _stream_to_text(self, chunk):
        if self.exiting:
            return
        def update_text_stream():
            try:
                self.response_text.config(state=tk.NORMAL)
                self.response_text.insert(tk.END, chunk)
                self._trim_response_text()
                self.response_text.see(tk.END)
                self.response_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass
        try:
            self.master.after(0, update_text_stream)
        except (tk.TclError, RuntimeError):
            pass

    # =========================================================================
    # 6. ACTION ROUTER / TOOL EXECUTOR
    # =========================================================================

    @staticmethod
    def extract_json_from_text(text):
        if not text:
            return None
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text, idx)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _filter_and_rank_search_results(query, raw_results, max_keep=3):
        """Filters out spam/non-ASCII noise and ranks snippets by query keyword overlap."""
        if not raw_results:
            return []
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "what", "when",
            "where", "which", "who", "how", "are", "was", "were", "have", "has",
            "about", "into", "news", "latest", "today", "current", "search",
        }
        query_terms = [
            w for w in re.findall(r"[a-z0-9]+", str(query).lower())
            if len(w) >= 3 and w not in stopwords
        ]
        ascii_query = all(ord(c) < 128 for c in str(query))

        scored = []
        for idx, r in enumerate(raw_results):
            if not isinstance(r, dict):
                continue
            title = " ".join(str(r.get("title") or "").split())
            body = " ".join(str(r.get("body") or r.get("snippet") or "").split())
            date_str = " ".join(str(r.get("date") or "").split())
            combined = f"{title} {body}".strip()
            if len(body) < 25:
                continue
            if ascii_query and combined:
                non_ascii = sum(1 for ch in combined if ord(ch) > 127)
                if (non_ascii / len(combined)) > 0.15:
                    continue

            title_low = title.lower()
            body_low = body.lower()
            overlap = 0
            for term in query_terms:
                if term in title_low:
                    overlap += 2
                elif term in body_low:
                    overlap += 1
            scored.append((overlap, -idx, {"title": title or "No Title", "body": body[:280], "date": date_str}))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:max_keep]]

    def execute_action_and_get_result(self, action_data):
        raw_action = str(action_data.get("action", "")).lower().strip()
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
            return f"CHAT_FALLBACK::The current time is {now}."

        elif action == "describe_view":
            latest_frame = self.camera_pipeline.get_latest_frame() if self.camera_pipeline else None
            if latest_frame is not None:
                detector = getattr(self, "object_detector", None)
                if not detector:
                    detector = ObjectDetector()
                    self.object_detector = detector

                self.set_state(BotStates.THINKING, "Looking around...")
                detected = detector.detect_objects(latest_frame)
                if detected:
                    return f"CHAT_FALLBACK::I see: {', '.join(detected)}."
                else:
                    return "CHAT_FALLBACK::I don't see any recognizable objects."
            else:
                return "CHAT_FALLBACK::My camera is currently inactive or warming up."

        elif action == "search_web":
            query = str(value).strip() if value is not None else ""
            if not query:
                return "INVALID_ACTION"
            if not DDGS_AVAILABLE:
                return "SEARCH_ERROR"
            print(f"Searching DuckDuckGo for: '{query}'...", flush=True)
            wants_news = (
                raw_action in ("news", "search_news")
                or any(w in query.lower() for w in ("news", "headline", "headlines", "latest", "breaking"))
            )
            search_order = ("news", "text") if wants_news else ("text", "news")
            try:
                ddgs = DDGS()
                try:
                    clean_results = []
                    for mode in search_order:
                        try:
                            fn = ddgs.news if mode == "news" else ddgs.text
                            try:
                                kwargs = {"region": "us-en", "safesearch": "moderate", "max_results": 6}
                                if mode == "text":
                                    kwargs["backend"] = "auto"
                                raw_results = list(fn(query, **kwargs))
                            except TypeError:
                                raw_results = list(fn(query, region="us-en", max_results=6))
                            clean_results = self._filter_and_rank_search_results(query, raw_results, max_keep=3)
                            if clean_results:
                                print(f"[SEARCH] Kept {len(clean_results)} {mode} results (top: {clean_results[0]['title']})", flush=True)
                                break
                        except Exception as e:
                            print(f"[SEARCH DEBUG] {mode.title()} query failed: {e}", flush=True)
                finally:
                    if hasattr(ddgs, "close"):
                        try:
                            ddgs.close()
                        except Exception:
                            pass
                    elif hasattr(ddgs, "__exit__"):
                        try:
                            ddgs.__exit__(None, None, None)
                        except Exception:
                            pass

                if clean_results:
                    formatted = []
                    for i, r in enumerate(clean_results, 1):
                        date_part = f" ({r['date']})" if r.get("date") else ""
                        formatted.append(f"[{i}] Title: {r['title']}{date_part}\nSnippet: {r['body']}")
                    ctx = f"SEARCH RESULTS for '{query}':\n" + "\n".join(formatted)
                    print(f"[SEARCH] Context:\n{ctx}", flush=True)
                    return ctx
                print("[SEARCH] Returned 0 clean results.", flush=True)
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
            
            while not self.exiting and not self.stop_event.is_set():
                user_text = None
                if self._auto_followup:
                    self._auto_followup = False
                    self._pending_wake_command = ""
                    if self.idle_event.is_set() or self.interrupted.is_set():
                        continue
                    if self.ptt_event.is_set():
                        self.ptt_event.clear()
                        self.idle_event.clear()
                        self.set_state(BotStates.LISTENING, "I'm listening!")
                        audio_file = self.record_voice_ptt()
                    else:
                        self.idle_event.clear()
                        self.set_state(BotStates.LISTENING, "Listening for follow-up...")
                        audio_file = self.record_voice_adaptive()
                else:
                    trigger_source = self.detect_wake_word_or_ptt()
                    if trigger_source == "EXIT" or self.exiting or self.stop_event.is_set():
                        self._pending_wake_command = ""
                        break
                    if self.interrupted.is_set():
                        self._pending_wake_command = ""
                        self.interrupted.clear()
                        self.set_state(BotStates.IDLE, "Resetting...")
                        continue

                    pending_cmd = self._pending_wake_command
                    self._pending_wake_command = ""
                    self.idle_event.clear()

                    if trigger_source == "WAKE" and pending_cmd:
                        user_text = pending_cmd
                        audio_file = None
                    else:
                        self.set_state(BotStates.LISTENING, "I'm listening!")
                        audio_file = (
                            self.record_voice_ptt()
                            if trigger_source == "PTT"
                            else self.record_voice_adaptive()
                        )

                if user_text is None:
                    if self.exiting or self.stop_event.is_set():
                        break
                    has_audio = (
                        audio_file is not None
                        and (not isinstance(audio_file, np.ndarray) or audio_file.size > 0)
                    )
                    if not has_audio:
                        self.set_state(BotStates.IDLE, "Heard nothing.")
                        continue

                    user_text = self.transcribe_audio(audio_file)
                    if self.exiting or self.stop_event.is_set():
                        break
                    if not user_text:
                        self.set_state(BotStates.IDLE, "Transcription empty.")
                        continue
                
                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self.idle_event.clear()
                replied = self.chat_and_respond(user_text)

                if self.exiting or self.stop_event.is_set():
                    break

                # Restart adaptive follow-up listening unless idle commanded or reply failed
                if replied and self.auto_listen and not self.idle_event.is_set() and not self.exiting and not self.stop_event.is_set():
                    print("[AUTO] Listening for follow-up (Adaptive)...", flush=True)
                    self._auto_followup = True
                    
        except Exception as e:
            if not self.exiting and not self.stop_event.is_set():
                traceback.print_exc()
                self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    def warm_up_logic(self):
        global TEXT_MODEL
        self.set_state(BotStates.WARMUP, "Warming up brains...")

        installed = OLLAMA_CLIENT.list_installed_models()
        if installed and TEXT_MODEL not in installed:
            base_prefix = TEXT_MODEL.split(":")[0]
            chosen = next((m for m in installed if m.startswith(base_prefix)), installed[0])
            print(f"[LLM] Configured model '{TEXT_MODEL}' not installed; using '{chosen}' from /api/tags ({installed})", flush=True)
            TEXT_MODEL = chosen
        elif not installed:
            available = OLLAMA_CLIENT.list_available_models()
            if TEXT_MODEL in available:
                target = TEXT_MODEL
            else:
                target = next((m for m in available if "qwen2.5" in m), available[0] if available else TEXT_MODEL)
            print(f"[LLM] No models installed in hailo-ollama — pulling '{target}'...", flush=True)
            try:
                OLLAMA_CLIENT.pull(target)
                TEXT_MODEL = target
                print(f"[LLM] Pulled '{target}' successfully.", flush=True)
            except Exception as pull_err:
                print(f"[LLM WARNING] Failed pulling '{target}': {pull_err}", flush=True)

        try:
            # Tell Ollama/hailo-ollama to load model into NPU memory
            OLLAMA_CLIENT.generate(model=TEXT_MODEL, prompt="", keep_alive=-1)
        except Exception as gen_err:
            try:
                OLLAMA_CLIENT.chat(
                    model=TEXT_MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                    stream=False,
                    options=OLLAMA_OPTIONS,
                )
            except Exception as e:
                print(f"Failed loading LLM model {TEXT_MODEL}: {gen_err} / {e}", flush=True)
        
        self.play_sound(self.get_random_sound(greeting_sounds_dir), interruptible=False)
        print("[INIT] Models loaded & active.", flush=True)

    def _wait_for_ptt_fallback(self):
        self.ptt_event.wait()
        if self.exiting or self.stop_event.is_set():
            return "EXIT"
        self.ptt_event.clear()
        return "PTT"

    def _check_listen_interrupts(self, check_cli=True):
        if self.exiting or self.stop_event.is_set():
            raise StopIteration("EXIT")
        if self.ptt_event.is_set():
            self.ptt_event.clear()
            print("\n[AUDIO] Wake word stream interrupted by PTT.", flush=True)
            raise StopIteration("PTT")
        if check_cli:
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    sys.stdin.readline()
                    raise StopIteration("CLI")
            except StopIteration:
                raise
            except Exception:
                pass

    @staticmethod
    def _make_audio_queue_callback(audio_q):
        def _audio_cb(indata, frames, time_info, status):
            if status and getattr(status, "input_overflow", False):
                try:
                    audio_q.put_nowait((None, True))
                except queue.Full:
                    pass
                return
            try:
                audio_q.put_nowait((indata.copy(), False))
            except queue.Full:
                try:
                    audio_q.get_nowait()
                    audio_q.put_nowait((indata.copy(), False))
                except (queue.Empty, queue.Full):
                    pass
        return _audio_cb

    @staticmethod
    def _drain_audio_queue(audio_q):
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def _print_wake_meter(samples, in_speech, last_meter_time):
        if not IS_TTY:
            return last_meter_time
        now_t = time.time()
        if now_t - last_meter_time < 0.08:
            return last_meter_time
        current_max = int(np.max(np.abs(samples))) if samples.size > 0 else 0
        bar_length = 30
        filled = min(bar_length, int((current_max / 32768) * bar_length))
        bar = "█" * filled + "░" * (bar_length - filled)
        level_label = "LOUD" if current_max > 10000 else ("MED " if current_max > 2000 else "LOW ")
        state_tag = " | 🎙️ Speech..." if in_speech else ""
        print(f"\r[MIC] {level_label} |{bar}| {current_max:5d}{state_tag}   ", end="", flush=True)
        return now_t

    def detect_wake_word_or_ptt(self):
        self._pending_wake_command = ""
        if self.exiting or self.stop_event.is_set():
            return "EXIT"
        if self.ptt_event.is_set():
            self.ptt_event.clear()
            return "PTT"

        self.set_state(BotStates.IDLE, "Waiting...")

        use_oww = (getattr(self, "wake_word_engine", "whisper") == "openwakeword")
        if use_oww:
            if self.oww_model:
                self.oww_model.reset()
            if self.oww_model is None:
                return self._wait_for_ptt_fallback()
        elif self.whisper_model is None:
            print("[AUDIO WARNING] Whisper unavailable for wake word — waiting for PTT.", flush=True)
            return self._wait_for_ptt_fallback()

        OWW_SAMPLE_RATE = 16000
        OWW_CHUNK = 1280

        input_rate = self._input_samplerate
        BLOCK_SIZE = 1024

        stream_args = {
            "samplerate": input_rate,
            "channels": 1,
            "dtype": 'int16',
            "blocksize": BLOCK_SIZE,
            "device": INPUT_DEVICE_NAME,
            "latency": "high",
        }

        try:
            if use_oww:
                use_resampling = (input_rate != OWW_SAMPLE_RATE)
                print(f"[AUDIO] Wake word stream: {input_rate}Hz resample={use_resampling}", flush=True)
                self._listen_loop(stream_args, input_rate, OWW_SAMPLE_RATE, OWW_CHUNK, use_resampling)
            else:
                print(f"[AUDIO] Wake word stream (whisper): {input_rate}Hz phrases={self.wake_words}", flush=True)
                self._listen_loop_whisper(stream_args, input_rate)
        except StopIteration as si:
            return str(si)
        except Exception as e:
            if self.exiting or self.stop_event.is_set():
                return "EXIT"
            print(f"[AUDIO] Stream error: {e}. Falling back to PTT.", flush=True)
            return self._wait_for_ptt_fallback()

        if self.exiting or self.stop_event.is_set():
            return "EXIT"
        return "WAKE"

    def _close_stream(self, stream):
        """Aborts and closes a sounddevice stream without blocking in Pa_StopStream."""
        if stream is None:
            return
        try:
            stream.abort(ignore_errors=True)
        except TypeError:
            try:
                stream.abort()
            except Exception:
                pass
        except Exception:
            pass
        try:
            stream.close(ignore_errors=True)
        except TypeError:
            try:
                stream.close()
            except Exception:
                pass
        except Exception:
            pass

    def _transcribe_wake_audio(self, audio_float32):
        """Runs fast Whisper decoding on a candidate wake-word speech burst."""
        if self.whisper_model is None or audio_float32 is None or audio_float32.size < 3200:
            return ""
        try:
            segments, _ = self.whisper_model.transcribe(
                audio_float32,
                language="en",
                beam_size=1,
                temperature=0.0,
                initial_prompt=_format_wake_prompt(self.wake_words),
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.45,
                    min_speech_duration_ms=150,
                    min_silence_duration_ms=250,
                    speech_pad_ms=150,
                ),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.0,
                log_prob_threshold=-1.0,
            )
            kept_segs = [
                seg for seg in segments
                if getattr(seg, "no_speech_prob", 0.0) < 0.65
                and getattr(seg, "avg_logprob", 0.0) > -1.05
            ]
            return " ".join(seg.text for seg in kept_segs).strip()
        except Exception as e:
            print(f"\n[WAKE STT ERROR] Whisper wake decode failed: {e}", flush=True)
            return ""

    def _listen_loop_whisper(self, stream_args, input_rate):
        """Listens continuously and detects configured wake phrases via Whisper."""
        blocksize = int(stream_args.get("blocksize", 1024))
        overflow_count = 0
        MAX_OVERFLOWS = 30
        audio_q = queue.Queue(maxsize=128)

        calib_chunks_needed = max(3, int(round(0.3 * input_rate / blocksize)))
        preroll_max_chunks = max(4, int(round(0.45 * input_rate / blocksize)))
        min_speech_chunks = max(2, int(round(0.12 * input_rate / blocksize)))
        SILENCE_END_SEC = 0.55
        MAX_BURST_SEC = 4.0

        calib_rms = []
        preroll = deque(maxlen=preroll_max_chunks)
        noise_floor = 0.008

        def _compute_thresholds(floor):
            return max(0.015, floor * 1.9 + 0.006), max(0.010, floor * 1.35 + 0.004)

        speech_threshold, silence_threshold = _compute_thresholds(noise_floor)
        in_speech = False
        speech_chunks = []
        burst_rms_vals = []
        speech_active_chunks = 0
        burst_samples = 0
        silence_samples = 0
        last_meter_time = 0.0

        try:
            sos = scipy.signal.butter(2, 150.0, btype="highpass", fs=input_rate, output="sos")
            zi = scipy.signal.sosfilt_zi(sos) * 0.0
        except Exception:
            sos = None
            zi = None

        cb_args = dict(stream_args)
        cb_args["callback"] = self._make_audio_queue_callback(audio_q)
        stream = sd.InputStream(**cb_args)
        stream.start()
        try:
            while not self.exiting and not self.stop_event.is_set():
                self._check_listen_interrupts(check_cli=True)

                try:
                    data, overflow = audio_q.get(timeout=0.05)
                except queue.Empty:
                    self._check_listen_interrupts(check_cli=False)
                    continue

                self._check_listen_interrupts(check_cli=False)

                if overflow:
                    overflow_count += 1
                    if overflow_count > MAX_OVERFLOWS:
                        raise RuntimeError(f"Stream buffer overflows exceeded limit ({overflow_count})")
                    continue
                else:
                    overflow_count = max(0, overflow_count - 1)

                chunk = np.asarray(data, dtype=np.int16)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()

                chunk_f = chunk.astype(np.float32) / 32768.0
                if sos is not None and chunk_f.size > 0:
                    chunk_hp, zi = scipy.signal.sosfilt(sos, chunk_f, zi=zi)
                elif chunk_f.size > 0:
                    chunk_hp = chunk_f - float(np.mean(chunk_f))
                else:
                    chunk_hp = chunk_f
                rms = float(np.sqrt(np.mean(chunk_hp * chunk_hp))) if chunk_hp.size > 0 else 0.0

                if len(calib_rms) < calib_chunks_needed:
                    calib_rms.append(rms)
                    if len(calib_rms) < calib_chunks_needed:
                        preroll.append(chunk)
                        last_meter_time = self._print_wake_meter(chunk, in_speech, last_meter_time)
                        continue
                    noise_floor = min(float(np.percentile(calib_rms, 25)), 0.15)
                    self._ambient_noise_floor = noise_floor
                    speech_threshold, silence_threshold = _compute_thresholds(noise_floor)

                if not in_speech:
                    if rms >= speech_threshold:
                        in_speech = True
                        speech_chunks = list(preroll)
                        speech_chunks.append(chunk)
                        burst_rms_vals = [rms]
                        speech_active_chunks = 1
                        burst_samples = chunk.size
                        silence_samples = 0
                        preroll.clear()
                    else:
                        if rms < noise_floor:
                            noise_floor = noise_floor * 0.92 + rms * 0.08
                            self._ambient_noise_floor = noise_floor
                            speech_threshold, silence_threshold = _compute_thresholds(noise_floor)
                        elif rms < speech_threshold:
                            noise_floor = noise_floor * 0.985 + rms * 0.015
                            self._ambient_noise_floor = noise_floor
                            speech_threshold, silence_threshold = _compute_thresholds(noise_floor)
                        preroll.append(chunk)
                else:
                    speech_chunks.append(chunk)
                    burst_rms_vals.append(rms)
                    burst_samples += chunk.size
                    if rms >= silence_threshold:
                        silence_samples = 0
                        if rms >= speech_threshold:
                            speech_active_chunks += 1
                    else:
                        silence_samples += chunk.size

                    silence_sec = silence_samples / float(input_rate)
                    burst_sec = burst_samples / float(input_rate)
                    if silence_sec >= SILENCE_END_SEC or burst_sec >= MAX_BURST_SEC:
                        chunks_to_process = speech_chunks
                        active_chunks = speech_active_chunks
                        rms_snapshot = burst_rms_vals
                        hit_max_cap = (burst_sec >= MAX_BURST_SEC and silence_sec < SILENCE_END_SEC)

                        in_speech = False
                        speech_chunks = []
                        burst_rms_vals = []
                        speech_active_chunks = 0
                        burst_samples = 0
                        silence_samples = 0
                        preroll.clear()

                        if active_chunks >= min_speech_chunks and chunks_to_process:
                            burst_int16 = np.concatenate(chunks_to_process)
                            burst_f32 = burst_int16.astype(np.float32) / 32768.0
                            burst_f32 = _highpass_filter(burst_f32, input_rate, cutoff_hz=150.0)
                            if input_rate != 16000:
                                from math import gcd
                                g = gcd(int(input_rate), 16000)
                                burst_f32 = scipy.signal.resample_poly(
                                    burst_f32, 16000 // g, int(input_rate) // g
                                ).astype(np.float32)
                            burst_f32 = np.ascontiguousarray(burst_f32, dtype=np.float32)

                            heard_text = self._transcribe_wake_audio(burst_f32)
                            self._drain_audio_queue(audio_q)
                            overflow_count = 0
                            self._check_listen_interrupts(check_cli=False)

                            matched_phrase, trailing = _match_wake_phrase(heard_text, self._wake_patterns)
                            if matched_phrase:
                                print(f"\n[WAKE] Wake Word Triggered ('{matched_phrase}')! Heard: '{heard_text}'", flush=True)
                                self._pending_wake_command = trailing
                                return
                            if heard_text:
                                print(f"\n[WAKE HEAR] '{heard_text}'", flush=True)
                            elif hit_max_cap and rms_snapshot:
                                noise_floor = min(0.15, noise_floor * 0.7 + float(np.median(rms_snapshot)) * 0.3)
                                speech_threshold, silence_threshold = _compute_thresholds(noise_floor)

                last_meter_time = self._print_wake_meter(chunk, in_speech, last_meter_time)
        finally:
            self._close_stream(stream)

        if self.exiting or self.stop_event.is_set():
            raise StopIteration("EXIT")

    def _listen_loop(self, stream_args, input_rate, oww_rate, oww_chunk, use_resampling):
        needed_input_samples = int(oww_chunk * (input_rate / oww_rate)) if use_resampling else oww_chunk
        needed_bytes = needed_input_samples * 2
        pcm_buf = bytearray()
        overflow_count = 0
        MAX_OVERFLOWS = 30
        audio_q = queue.Queue(maxsize=64)

        cb_args = dict(stream_args)
        cb_args["callback"] = self._make_audio_queue_callback(audio_q)
        stream = sd.InputStream(**cb_args)
        stream.start()
        try:
            while not self.exiting and not self.stop_event.is_set():
                self._check_listen_interrupts(check_cli=True)

                try:
                    data, overflow = audio_q.get(timeout=0.05)
                except queue.Empty:
                    self._check_listen_interrupts(check_cli=False)
                    continue

                self._check_listen_interrupts(check_cli=False)

                if overflow:
                    overflow_count += 1
                    if overflow_count > MAX_OVERFLOWS:
                        raise RuntimeError(f"Stream buffer overflows exceeded limit ({overflow_count})")
                    continue
                else:
                    overflow_count = max(0, overflow_count - 1)

                chunk = np.asarray(data, dtype=np.int16)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()
                pcm_buf.extend(chunk.tobytes())

                if len(pcm_buf) < needed_bytes:
                    continue

                window = np.frombuffer(pcm_buf[:needed_bytes], dtype=np.int16).copy()
                del pcm_buf[:needed_bytes]

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

                if IS_TTY:
                    bar_length = 30
                    filled = int((current_max / 32768) * bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    level_label = "LOUD" if current_max > 10000 else ("MED " if current_max > 2000 else "LOW ")
                    print(f"\r[MIC] {level_label} |{bar}| {current_max:5d}{oww_score_str}   ", end="", flush=True)
        finally:
            self._close_stream(stream)

        if self.exiting or self.stop_event.is_set():
            raise StopIteration("EXIT")

    def record_voice_adaptive(self, filename="input.wav"):
        if self.exiting or self.stop_event.is_set():
            return None
        print("Recording (Adaptive)...", flush=True)
        samplerate = self._input_samplerate

        silence_duration = float(CURRENT_CONFIG.get("silence_duration", 0.9))
        no_speech_timeout = float(CURRENT_CONFIG.get("no_speech_timeout", 4.0))
        max_record_time = float(CURRENT_CONFIG.get("max_record_time", 10.0))
        blocksize = 1024

        buffer = []
        audio_q = queue.Queue(maxsize=128)
        noise_floor = float(np.clip(getattr(self, "_ambient_noise_floor", 0.008), 0.002, 0.12))

        def _compute_thresholds(floor):
            return max(0.015, floor * 2.2 + 0.008), max(0.010, floor * 1.5 + 0.004)

        speech_threshold, silence_threshold = _compute_thresholds(noise_floor)
        speech_detected = False
        last_speech_time = None
        peak_rms = 0.0

        try:
            sos = scipy.signal.butter(2, 150.0, btype="highpass", fs=samplerate, output="sos")
            zi = scipy.signal.sosfilt_zi(sos) * 0.0
        except Exception:
            sos = None
            zi = None

        stream = None
        rec_start = time.time()
        try:
            if self.exiting or self.stop_event.is_set() or self.idle_event.is_set():
                return None
            stream = sd.InputStream(
                samplerate=samplerate,
                channels=1,
                dtype="int16",
                callback=self._make_audio_queue_callback(audio_q),
                device=INPUT_DEVICE_NAME,
                blocksize=blocksize,
                latency="high",
            )
            stream.start()
            rec_start = time.time()
            while not self.exiting and not self.stop_event.is_set() and not self.idle_event.is_set() and not self.interrupted.is_set():
                now = time.time()
                elapsed = now - rec_start
                if elapsed >= max_record_time:
                    break
                if speech_detected and last_speech_time is not None and (now - last_speech_time) >= silence_duration:
                    break
                if not speech_detected and elapsed >= no_speech_timeout:
                    break

                try:
                    data, overflow = audio_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if overflow or data is None:
                    continue

                chunk = np.asarray(data, dtype=np.int16)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()
                buffer.append(chunk)

                chunk_f = chunk.astype(np.float32) / 32768.0
                if sos is not None and chunk_f.size > 0:
                    chunk_hp, zi = scipy.signal.sosfilt(sos, chunk_f, zi=zi)
                elif chunk_f.size > 0:
                    chunk_hp = chunk_f - float(np.mean(chunk_f))
                else:
                    chunk_hp = chunk_f

                rms = float(np.sqrt(np.mean(chunk_hp * chunk_hp))) if chunk_hp.size > 0 else 0.0
                if rms > peak_rms:
                    peak_rms = rms

                if rms < noise_floor:
                    noise_floor = noise_floor * 0.9 + rms * 0.1
                    speech_threshold, silence_threshold = _compute_thresholds(noise_floor)

                now = time.time()
                if rms >= speech_threshold:
                    speech_detected = True
                    last_speech_time = now
                elif speech_detected and rms >= silence_threshold:
                    last_speech_time = now
        except Exception as e:
            if not self.exiting and not self.stop_event.is_set():
                print(f"[AUDIO ERROR] Adaptive recording failed: {e}", flush=True)
            return None
        finally:
            self._close_stream(stream)

        self._ambient_noise_floor = noise_floor
        print(
            f"[AUDIO] Adaptive done ({time.time() - rec_start:.1f}s | speech={speech_detected} "
            f"| peak_rms={peak_rms:.4f} | floor={noise_floor:.4f} | thresh={speech_threshold:.4f})",
            flush=True,
        )
        if self.exiting or self.stop_event.is_set() or self.idle_event.is_set() or self.interrupted.is_set():
            return None
        if not speech_detected:
            print("[AUDIO] No speech detected above floor — skipping transcription.", flush=True)
            return None
        return self.process_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        if self.exiting or self.stop_event.is_set():
            return None
        duration = self.ptt_auto_stop_duration
        self.ptt_auto_stop_duration = None
        if duration is not None and not self.recording_active.is_set():
            self.recording_active.set()

        print("Recording (PTT)...", flush=True)
        samplerate = self._input_samplerate
        buffer = []

        def callback(indata, frames, time_info, status):
            buffer.append(indata.copy())

        stream = None
        try:
            if self.exiting or self.stop_event.is_set() or self.idle_event.is_set():
                return None
            stream = sd.InputStream(
                samplerate=samplerate,
                channels=1,
                dtype="int16",
                blocksize=1024,
                callback=callback,
                device=INPUT_DEVICE_NAME,
                latency="high",
            )
            stream.start()
            if duration is not None:
                self._start_auto_stop_timer(duration)
            while self.recording_active.is_set() and not self.exiting and not self.stop_event.is_set() and not self.idle_event.is_set():
                time.sleep(0.05)
        except Exception as e:
            if not self.exiting and not self.stop_event.is_set():
                print(f"[AUDIO ERROR] PTT recording failed: {e}", flush=True)
            return None
        finally:
            self._close_stream(stream)
            self._cancel_auto_stop_timer()
            self.recording_active.clear()

        if self.exiting or self.stop_event.is_set() or self.idle_event.is_set():
            return None
        return self.process_audio_buffer(buffer, filename, samplerate)

    def process_audio_buffer(self, buffer, filename="input.wav", samplerate=16000):
        """High-pass filters and resamples captured audio in RAM for Whisper (zero SD writes)."""
        if not buffer:
            return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        min_samples = int(samplerate * 0.1)
        if len(audio_data) < min_samples:
            print(f"[AUDIO] Recording too short ({len(audio_data)/samplerate:.2f}s) — skipping.", flush=True)
            return None
        if getattr(audio_data, "dtype", None) == np.int16:
            audio_f = audio_data.astype(np.float32) / 32768.0
        else:
            audio_f = np.nan_to_num(audio_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        audio_f = _highpass_filter(audio_f, samplerate, cutoff_hz=150.0)

        WHISPER_RATE = 16000
        if samplerate != WHISPER_RATE:
            from math import gcd
            g = gcd(int(samplerate), WHISPER_RATE)
            up, down = WHISPER_RATE // g, int(samplerate) // g
            audio_f = scipy.signal.resample_poly(audio_f, up, down).astype(np.float32)

        audio_f = np.ascontiguousarray(np.clip(audio_f, -1.0, 1.0), dtype=np.float32)

        if CURRENT_CONFIG.get("debug_save_wav", False) and filename:
            wav_path = _resolve_path(filename)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(WHISPER_RATE)
                wf.writeframes((audio_f * 32767.0).astype(np.int16).tobytes())
            print(f"[AUDIO] Debug saved {len(audio_f)/WHISPER_RATE:.1f}s to {wav_path}", flush=True)

        self.play_sound_async(self.get_random_sound(ack_sounds_dir))
        return audio_f

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        return self.process_audio_buffer(buffer, filename, samplerate)

    def transcribe_audio(self, filename):
        print("Transcribing...", flush=True)
        if self.whisper_model is None:
            print("[STT ERROR] Whisper model not loaded.", flush=True)
            return ""
        try:
            if isinstance(filename, np.ndarray):
                audio_float32 = np.ascontiguousarray(filename.squeeze(), dtype=np.float32)
            else:
                with wave.open(_resolve_path(filename), "rb") as wf:
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

                audio_float32 = _highpass_filter(audio_float32, sr, cutoff_hz=150.0)

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
                beam_size=2,
                temperature=0.0,
                initial_prompt=f"Noah is a cute, helpful robot assistant. {_format_wake_prompt(self.wake_words)}",
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=200,
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
                condition_on_previous_text=False,
                no_speech_threshold=0.55,
                compression_ratio_threshold=2.0,
                log_prob_threshold=-0.8,
            )

            HALLUCINATION_PHRASES = {
                "thank you",
                "thanks for watching",
                "thank you for watching",
                "you",
                "bye",
                "mbc",
                "subtitles by",
            }
            kept_segs = [
                seg for seg in segments
                if getattr(seg, "no_speech_prob", 0.0) < 0.55
                and getattr(seg, "avg_logprob", 0.0) > -0.85
            ]
            transcription = " ".join(seg.text for seg in kept_segs).strip()
            cleaned_lower = re.sub(r'[^a-z\s]', '', transcription.lower()).strip()
            if (len(kept_segs) == 1 and cleaned_lower in HALLUCINATION_PHRASES
                    and getattr(kept_segs[0], "no_speech_prob", 0.0) > 0.30):
                transcription = ""

            if transcription:
                transcription = re.sub(r'\bhey\s+noa\b', 'Hey Noah', transcription, flags=re.IGNORECASE)

            if not transcription:
                print("[STT] Heard: (nothing confident)", flush=True)
                return ""
            print(f"[STT] Heard: '{transcription}'", flush=True)
            return transcription
        except Exception as e:
            print(f"[STT ERROR] Whisper decoding failed: {e}", flush=True)
            return ""

    @staticmethod
    def _extract_ollama_content(resp):
        if resp is None:
            return ""
        if isinstance(resp, dict):
            msg = resp.get("message")
            if msg is not None:
                if isinstance(msg, dict):
                    return msg.get("content", "") or ""
                return getattr(msg, "content", "") or ""
            if resp.get("response"):
                return resp.get("response") or ""
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                if isinstance(c0, dict):
                    delta = c0.get("delta") or c0.get("message") or {}
                    if isinstance(delta, dict):
                        return delta.get("content", "") or ""
            return ""
        msg = getattr(resp, "message", None)
        if msg is not None:
            return getattr(msg, "content", "") or ""
        return getattr(resp, "response", "") or ""

    @staticmethod
    def _norm_text(s):
        return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

    _ABBREV_PAT = re.compile(r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.', re.IGNORECASE)

    @classmethod
    def _split_complete_sentences(cls, text):
        """Splits complete sentences while keeping abbreviations, decimals, and trailing fragments."""
        if not text:
            return [], ""
        protected = cls._ABBREV_PAT.sub(lambda m: m.group(0).replace('.', '\x00'), text)
        parts = re.split(r'(?<=[.!?])\s+|\n+', protected)
        if len(parts) <= 1:
            return [], text
        sentences = [p.replace('\x00', '.').strip() for p in parts[:-1] if p.strip()]
        remainder = parts[-1].replace('\x00', '.')
        return sentences, remainder

    @staticmethod
    def _is_potential_action_prefix(stripped):
        if not stripped:
            return True
        if stripped.startswith('{') or stripped.startswith('```'):
            return True
        for prefix in ("```json", "```"):
            if prefix.startswith(stripped.lower()):
                return True
        return False

    @staticmethod
    def _is_action_payload(stripped):
        return bool(stripped.startswith('{') or re.match(r'^```(?:json)?\s*\{', stripped, re.IGNORECASE))

    def _enqueue_tts(self, text):
        clean = (text or "").strip()
        if not clean:
            return
        with self.tts_cond:
            self.tts_queue.append(clean)
            self.tts_cond.notify_all()

    def _clear_tts_queue(self):
        with self.tts_cond:
            self.tts_queue.clear()
            self.tts_cond.notify_all()

    @staticmethod
    def _strip_speaker_prefix(text):
        return re.sub(r"^\s*(?:you|assistant|bot|noah)\s*:\s*", "", str(text or ""), flags=re.IGNORECASE)

    def build_system_prompt(self, query=""):
        return KNOWLEDGE_BASE.build_system_prompt(
            query=query,
            top_k=int(CURRENT_CONFIG.get("knowledge_top_k", 3)),
            extras=CURRENT_CONFIG.get("system_prompt_extras", ""),
        )

    def _commit_turn(self, user_msg, assistant_text):
        if not self.chat_memory_enabled or not str(assistant_text).strip():
            return
        max_msgs = max(2, int(CURRENT_CONFIG.get("max_history_turns", 6)) * 2)
        self.session_memory.append(user_msg)
        self.session_memory.append({"role": "assistant", "content": str(assistant_text)})
        if len(self.session_memory) > max_msgs:
            self.session_memory = self.session_memory[-max_msgs:]

    def _start_thinking_sound(self):
        self._sfx_abort.clear()
        self.thinking_sound_active.set()
        if self._thinking_thread is None or not self._thinking_thread.is_alive():
            self._thinking_thread = threading.Thread(
                target=self._run_thinking_sound_loop, daemon=True, name="ThinkingSoundThread"
            )
            self._thinking_thread.start()

    def chat_and_respond(self, text):
        if "forget everything" in text.lower() or "reset memory" in text.lower():
            self.session_memory = []
            self.permanent_memory = []
            self._last_assistant_reply_norm = ""
            self._last_user_text_norm = ""
            self.save_chat_history()
            self.append_to_text("BOT: Okay. Memory wiped.")
            self._enqueue_tts("Okay. Memory wiped.")
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return True

        self.set_state(BotStates.THINKING, "Thinking...")

        live_sys = self.build_system_prompt(text)
        user_msg = {"role": "user", "content": text}
        if self.chat_memory_enabled:
            messages = [{"role": "system", "content": live_sys}] + self.permanent_memory + self.session_memory + [user_msg]
        else:
            messages = [{"role": "system", "content": live_sys}, user_msg]

        user_norm = self._norm_text(text)
        self._start_thinking_sound()

        full_response_buffer = ""
        chunk_buffer = ""
        buffered_gui_chunks = []
        hold_for_repeat = bool(
            self._last_assistant_reply_norm and user_norm != self._last_user_text_norm
        )

        def _flush_held_to_gui_and_tts():
            nonlocal chunk_buffer, bot_label_written
            if buffered_gui_chunks:
                joined = "".join(buffered_gui_chunks)
                buffered_gui_chunks.clear()
                if not bot_label_written:
                    joined = self._strip_speaker_prefix(joined)
                    self.append_to_text("BOT: ", newline=False)
                    bot_label_written = True
                self._stream_to_text(joined)
                chunk_buffer += joined
            elif not bot_label_written:
                self.append_to_text("BOT: ", newline=False)
                bot_label_written = True
            sentences, chunk_buffer = self._split_complete_sentences(chunk_buffer)
            for sent in sentences:
                clean_s = sent.encode('ascii', 'ignore').decode('ascii').strip()
                if clean_s and re.search(r'[a-zA-Z0-9]', clean_s):
                    self._enqueue_tts(clean_s)

        try:
            stream = OLLAMA_CLIENT.chat(model=TEXT_MODEL, messages=messages, stream=True, options=OLLAMA_OPTIONS)
            is_action_mode = False
            action_decided = False
            bot_label_written = False

            try:
                for chunk in stream:
                    if self.interrupted.is_set():
                        break
                    raw = self._extract_ollama_content(chunk)
                    if not raw:
                        continue
                    content = raw.encode('ascii', 'ignore').decode('ascii')
                    full_response_buffer += content

                    stripped = full_response_buffer.lstrip()
                    if not action_decided:
                        if self._is_action_payload(stripped):
                            is_action_mode = True
                            action_decided = True
                            self.thinking_sound_active.clear()
                            buffered_gui_chunks.clear()
                            continue
                        if len(stripped) < 16 and self._is_potential_action_prefix(stripped):
                            buffered_gui_chunks.append(content)
                            continue
                        action_decided = True
                        self.thinking_sound_active.clear()

                    if is_action_mode:
                        continue

                    self.thinking_sound_active.clear()
                    buffered_gui_chunks.append(content)

                    if hold_for_repeat:
                        cur_norm = self._norm_text(full_response_buffer)
                        if cur_norm and not self._last_assistant_reply_norm.startswith(cur_norm):
                            hold_for_repeat = False
                        else:
                            continue

                    _flush_held_to_gui_and_tts()
            finally:
                if hasattr(stream, "close"):
                    stream.close()

            if self.interrupted.is_set():
                self.thinking_sound_active.clear()
                return False

            if not full_response_buffer.strip():
                print("[LLM] Stream returned empty content — retrying non-streaming...", flush=True)
                fallback_resp = OLLAMA_CLIENT.chat(
                    model=TEXT_MODEL,
                    messages=messages,
                    stream=False,
                    options=OLLAMA_OPTIONS,
                )
                fallback_raw = self._extract_ollama_content(fallback_resp)
                fallback_clean = fallback_raw.encode('ascii', 'ignore').decode('ascii')
                if not fallback_clean.strip():
                    print(f"[LLM WARNING] Empty non-stream response: {fallback_resp!r}", flush=True)
                    self.thinking_sound_active.clear()
                    self.set_state(BotStates.IDLE, "No response.")
                    return False

                full_response_buffer = fallback_clean
                stripped = full_response_buffer.lstrip()
                if self._is_action_payload(stripped):
                    is_action_mode = True
                    self.thinking_sound_active.clear()
                else:
                    self.thinking_sound_active.clear()
                    buffered_gui_chunks = [full_response_buffer]
                    hold_for_repeat = False

            if full_response_buffer.strip():
                print(f"[LLM] Reply ({len(full_response_buffer)} chars): {full_response_buffer[:200]!r}", flush=True)

            reply_norm = self._norm_text(full_response_buffer)
            if (not is_action_mode and reply_norm
                    and reply_norm == self._last_assistant_reply_norm
                    and user_norm != self._last_user_text_norm):
                print("[LLM] Repetition loop detected — clearing context and retrying fresh...", flush=True)
                self.permanent_memory = []
                self.session_memory = []
                fresh_resp = OLLAMA_CLIENT.chat(
                    model=TEXT_MODEL,
                    messages=[{"role": "system", "content": live_sys}, user_msg],
                    stream=False,
                    options=OLLAMA_OPTIONS,
                )
                fresh_raw = self._extract_ollama_content(fresh_resp)
                fresh_clean = fresh_raw.encode('ascii', 'ignore').decode('ascii').strip()
                if fresh_clean and self._norm_text(fresh_clean) != reply_norm:
                    full_response_buffer = fresh_clean
                    buffered_gui_chunks = [full_response_buffer]
                    chunk_buffer = ""
                    if self._is_action_payload(full_response_buffer.lstrip()):
                        is_action_mode = True
                        buffered_gui_chunks.clear()
                    else:
                        self._clear_tts_queue()

            if not is_action_mode and buffered_gui_chunks:
                _flush_held_to_gui_and_tts()

            if not is_action_mode and chunk_buffer.strip():
                text_sentence = chunk_buffer.encode('ascii', 'ignore').decode('ascii').strip()
                if text_sentence and re.search(r'[a-zA-Z0-9]', text_sentence):
                    self._enqueue_tts(text_sentence)
                chunk_buffer = ""

            if is_action_mode:
                action_data = self.extract_json_from_text(full_response_buffer)
                if not action_data:
                    print(f"[LLM WARNING] Malformed action JSON: {full_response_buffer[:200]!r}", flush=True)
                    err_reply = "Sorry, I got confused on that one."
                    self.thinking_sound_active.clear()
                    self.set_state(BotStates.SPEAKING, "Speaking...")
                    self.append_to_text("BOT: ", newline=False)
                    self.append_to_text(err_reply, newline=True)
                    self._enqueue_tts(err_reply)
                else:
                    tool_result = self.execute_action_and_get_result(action_data)

                    if tool_result and tool_result.startswith("CHAT_FALLBACK::"):
                        chat_text = tool_result.split("::", 1)[1]
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(chat_text, newline=True)
                        self._enqueue_tts(chat_text)
                        self._commit_turn(user_msg, chat_text)
                        self._last_assistant_reply_norm = self._norm_text(chat_text)
                        self._last_user_text_norm = user_norm
                        self.save_chat_history()
                        self.wait_for_tts()
                        self.set_state(BotStates.IDLE, "Ready")
                        return True

                    if tool_result == "INVALID_ACTION":
                        fallback_text = "I am not sure how to do that."
                    elif tool_result == "SEARCH_EMPTY":
                        fallback_text = "I searched online, but couldn't find a clear answer for that."
                    elif tool_result == "SEARCH_ERROR":
                        fallback_text = "I cannot reach the internet right now."
                    else:
                        fallback_text = None

                    if fallback_text:
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(fallback_text, newline=True)
                        self._enqueue_tts(fallback_text)
                        self._last_assistant_reply_norm = self._norm_text(fallback_text)
                        self._last_user_text_norm = user_norm
                    elif tool_result:
                        today_str = datetime.date.today().isoformat()
                        search_val = str(action_data.get("value") or action_data.get("query") or text).strip()
                        summary_prompt = [
                            {
                                "role": "system",
                                "content": (
                                    f"Today's date is {today_str}. "
                                    "Answer the question in 1 or 2 short plain-text sentences using ONLY facts explicitly stated in the SEARCH RESULTS. "
                                    "Do NOT guess, assume, or invent numbers, names, or dates. "
                                    "If the SEARCH RESULTS do not contain the answer, say: I searched online, but could not find the exact answer."
                                ),
                            },
                            {"role": "user", "content": f"{tool_result}\n\nSearch Query: {search_val}\nUser Message: {text}"}
                        ]
                        self.set_state(BotStates.THINKING, "Reading...")
                        self._start_thinking_sound()

                        final_resp = OLLAMA_CLIENT.chat(
                            model=TEXT_MODEL,
                            messages=summary_prompt,
                            stream=False,
                            options=OLLAMA_FACTUAL_OPTIONS,
                        )
                        final_text = self._strip_speaker_prefix(
                            self._extract_ollama_content(final_resp).encode('ascii', 'ignore').decode('ascii')
                        ).strip()
                        if not final_text:
                            final_text = "I searched online, but could not find the exact answer."

                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")

                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(final_text, newline=True)
                        self._enqueue_tts(final_text)
                        if final_text.strip():
                            self._commit_turn(user_msg, final_text)
                            self._last_assistant_reply_norm = self._norm_text(final_text)
                            self._last_user_text_norm = user_norm
            else:
                if full_response_buffer.strip():
                    self.append_to_text("")
                    self._commit_turn(user_msg, full_response_buffer)
                    self._last_assistant_reply_norm = self._norm_text(full_response_buffer)
                    self._last_user_text_norm = user_norm

            self.save_chat_history()
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")
            return bool(full_response_buffer.strip())

        except Exception as e:
            self.thinking_sound_active.clear()
            print(f"LLM Chat Error: {e}", flush=True)
            self.set_state(BotStates.ERROR, "Brain Freeze!")
            return False

    def wait_for_tts(self):
        with self.tts_cond:
            while not (self.interrupted.is_set() or self.exiting or self.stop_event.is_set()):
                if not self.tts_queue and not self.tts_active.is_set():
                    break
                self.tts_cond.wait(timeout=0.1)
        if not (self.interrupted.is_set() or self.exiting or self.stop_event.is_set()):
            # Let ALSA playback tail finish before reopening microphone
            self.interrupted.wait(0.6)

    def _tts_worker(self):
        while not self.exiting and not self.stop_event.is_set():
            with self.tts_cond:
                while not self.tts_queue and not self.exiting and not self.stop_event.is_set():
                    self.tts_cond.wait(timeout=0.2)
                if self.exiting or self.stop_event.is_set():
                    break
                full_text = self.tts_queue.popleft()
                self.tts_active.set()
                self._sfx_abort.set()
            try:
                self.speak(full_text)
            finally:
                with self.tts_cond:
                    if not self.tts_queue:
                        self.tts_active.clear()
                    self.tts_cond.notify_all()

    def _kill_piper(self):
        proc = getattr(self, "_piper_process", None)
        self._piper_process = None
        if proc is None:
            return
        for stream_attr in ("stdin", "stdout"):
            pipe = getattr(proc, stream_attr, None)
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=0.5)
                except Exception:
                    pass

    def _drain_piper_stdout(self, max_sec=0.35):
        """Drains remaining PCM bytes after an interrupt or resets Piper if still streaming."""
        proc = self._piper_process
        if proc is None or proc.poll() is not None:
            return
        deadline = time.time() + max_sec
        quiet_ticks = 0
        while time.time() < deadline and quiet_ticks < 3:
            rlist, _, _ = select.select([proc.stdout], [], [], 0.02)
            if not rlist:
                quiet_ticks += 1
                continue
            quiet_ticks = 0
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
        else:
            if time.time() >= deadline:
                print("[PIPER] Interrupted stream still active; restarting Piper.", flush=True)
                self._kill_piper()

    def _init_piper(self):
        piper_bin = _resolve_path("piper/piper")
        piper_dir = os.path.dirname(piper_bin)
        voice_model = _resolve_path(CURRENT_CONFIG.get("voice_model", "piper/en_GB-semaine-medium.onnx"))
        espeak_data = _resolve_path("piper/espeak-ng-data")
        if not os.path.exists(piper_bin):
            print(f"[PIPER ERROR] {piper_bin} binary not found!", flush=True)
            return
        if not os.path.exists(voice_model):
            print(f"[PIPER ERROR] Voice model not found: {voice_model}", flush=True)
            return
        try:
            self._kill_piper()
            env = os.environ.copy()
            ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{piper_dir}:{ld_path}" if ld_path else piper_dir

            cmd = [piper_bin, "--model", voice_model, "--output-raw", "--json-input"]
            if os.path.isdir(espeak_data):
                cmd.extend(["--espeak_data", espeak_data])

            self._piper_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            print("[INIT] Persistent Piper TTS subprocess started.", flush=True)

            PIPER_RATE = 22050
            model_json = f"{voice_model}.json"
            if os.path.exists(model_json):
                try:
                    with open(model_json, "r") as mf:
                        PIPER_RATE = int(json.load(mf).get("audio", {}).get("sample_rate", PIPER_RATE))
                except Exception:
                    pass

            use_native_rate = False
            native_rate = 48000
            try:
                dev_info = sd.query_devices(self._output_device)
                native_rate = int(dev_info['default_samplerate'])
            except Exception:
                pass
            try:
                sd.check_output_settings(device=self._output_device, samplerate=PIPER_RATE, channels=1)
            except Exception:
                use_native_rate = True

            self._piper_rate = PIPER_RATE
            self._piper_native_rate = native_rate
            self._piper_use_native_rate = use_native_rate
        except Exception as e:
            print(f"[PIPER ERROR] Initialization failed: {e}", flush=True)
            self._kill_piper()

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
        text = text.encode('ascii', 'ignore').decode('ascii')
        clean = re.sub(r'[^a-zA-Z0-9\s,.!?:\'\-]', '', text).strip()
        if not clean:
            return

        print(f"[PIPER] Synthesis text: '{clean}'", flush=True)

        with self._audio_out_lock:
            if self._piper_process is None or self._piper_process.poll() is not None:
                print("[PIPER] Piper process died, restarting...", flush=True)
                self._init_piper()
            if self._piper_process is None:
                return

            was_interrupted = False
            try:
                msg = json.dumps({"text": clean}) + "\n"
                self._piper_process.stdin.write(msg.encode())
                self._piper_process.stdin.flush()

                playback_rate = self._piper_native_rate if self._piper_use_native_rate else self._piper_rate
                step = float(self._piper_rate) / float(playback_rate)
                prev_sample = None
                resample_pos = 0.0
                pending_byte = b""

                with sd.RawOutputStream(samplerate=playback_rate, channels=1, dtype='int16',
                                        device=self._output_device, latency='high', blocksize=2048) as stream:
                    audio_started = False
                    pre_audio_deadline = time.time() + 3.0
                    post_audio_gap = 0
                    MAX_GAP_TICKS = 25  # 25 * 20ms = 500ms silence flags end of phrase

                    while True:
                        if self.interrupted.is_set() or self.exiting or self.stop_event.is_set():
                            was_interrupted = True
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
                        raw_data = self._piper_process.stdout.read1(4096)
                        if not raw_data:
                            if audio_started:
                                break
                            continue

                        data = pending_byte + raw_data
                        even_len = len(data) & ~1
                        pending_byte = data[even_len:]
                        if even_len == 0:
                            continue

                        audio_chunk = np.frombuffer(data[:even_len], dtype=np.int16)
                        if audio_chunk.size > 0:
                            if not audio_started:
                                audio_started = True
                                self.set_state(BotStates.SPEAKING, "Speaking...")
                            self.current_volume = int(np.max(np.abs(audio_chunk)))

                            if self._piper_use_native_rate:
                                src = audio_chunk.astype(np.float32)
                                if prev_sample is not None:
                                    src = np.concatenate(([prev_sample], src))
                                else:
                                    resample_pos = 0.0
                                n_src = src.size
                                if n_src >= 2 and resample_pos < n_src - 1:
                                    out_pos = np.arange(resample_pos, n_src - 1, step, dtype=np.float64)
                                    if out_pos.size > 0:
                                        out_f = np.interp(out_pos, np.arange(n_src, dtype=np.float64), src)
                                        out_pcm = np.clip(np.rint(out_f), -32768, 32767).astype(np.int16)
                                        stream.write(out_pcm.tobytes())
                                        resample_pos = float(out_pos[-1] + step) - (n_src - 1)
                                    else:
                                        resample_pos -= (n_src - 1)
                                else:
                                    resample_pos -= max(0, n_src - 1)
                                prev_sample = float(src[-1])
                            else:
                                stream.write(audio_chunk.tobytes())

                if was_interrupted:
                    self._drain_piper_stdout()
                self.current_volume = 0
            except Exception as e:
                print(f"[PIPER ERROR] Synthesis failure: {e}", flush=True)
                self._kill_piper()
                self.current_volume = 0

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set() and not self._sfx_abort.is_set():
            if not self.tts_active.is_set():
                sound = self.get_random_sound(thinking_sounds_dir)
                if sound:
                    self.play_sound(sound, interruptible=True)
            for _ in range(50):
                if not self.thinking_sound_active.is_set() or self._sfx_abort.is_set() or self.exiting or self.stop_event.is_set():
                    return
                time.sleep(0.1)

    def get_random_sound(self, directory):
        resolved = _resolve_path(directory)
        if os.path.exists(resolved):
            files = [f for f in os.listdir(resolved) if f.lower().endswith(".wav")]
            return os.path.join(resolved, random.choice(files)) if files else None
        return None

    def play_sound_async(self, file_path, interruptible=True):
        if not file_path:
            return
        threading.Thread(
            target=self.play_sound,
            args=(file_path, interruptible),
            daemon=True,
            name="SoundEffectThread",
        ).start()

    def play_sound(self, file_path, interruptible=True):
        if not file_path or not os.path.exists(file_path):
            return
        if self.exiting or self.stop_event.is_set():
            return
        if interruptible and (self._sfx_abort.is_set() or self.tts_active.is_set() or self.interrupted.is_set()):
            return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw_bytes = wf.readframes(wf.getnframes())
            if not raw_bytes:
                return

            if sampwidth == 1:
                audio_f = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 2:
                audio_f = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio_f = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                return

            if nchannels > 1:
                audio_f = audio_f.reshape(-1, nchannels).mean(axis=1)

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=self._output_device, samplerate=file_sr, channels=1)
            except Exception:
                try:
                    dev_info = sd.query_devices(self._output_device)
                    playback_rate = int(dev_info['default_samplerate'])
                except Exception:
                    playback_rate = 48000
                if playback_rate != file_sr:
                    from math import gcd
                    g = gcd(int(file_sr), int(playback_rate))
                    audio_f = scipy.signal.resample_poly(
                        audio_f, int(playback_rate) // g, int(file_sr) // g
                    ).astype(np.float32)

            pcm = np.clip(np.rint(audio_f * 32767.0), -32768, 32767).astype(np.int16)

            with self._audio_out_lock:
                if self.exiting or self.stop_event.is_set():
                    return
                if interruptible and (self._sfx_abort.is_set() or self.tts_active.is_set() or self.interrupted.is_set()):
                    return
                with sd.RawOutputStream(
                    samplerate=playback_rate,
                    channels=1,
                    dtype='int16',
                    device=self._output_device,
                    latency='high',
                    blocksize=2048,
                ) as stream:
                    chunk_sz = 2048
                    for i in range(0, pcm.size, chunk_sz):
                        if self.exiting or self.stop_event.is_set():
                            break
                        if interruptible and (self._sfx_abort.is_set() or self.tts_active.is_set() or self.interrupted.is_set()):
                            break
                        stream.write(pcm[i:i + chunk_sz].tobytes())
        except Exception as e:
            print(f"[AUDIO WARNING] Sound effect playback failed ({file_path}): {e}", flush=True)

    # --- MEMORY STACK LOAD/SAVE ---

    def load_chat_history(self):
        if not bool(CURRENT_CONFIG.get("chat_memory", True)):
            return []
        max_msgs = max(2, int(CURRENT_CONFIG.get("max_history_turns", 6)) * 2)
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    history = json.load(f)
                if isinstance(history, list):
                    cleaned = []
                    for m in history:
                        if not isinstance(m, dict):
                            continue
                        role = str(m.get("role", "")).strip()
                        content = str(m.get("content", "")).strip()
                        if role not in ("user", "assistant") or not content:
                            continue
                        if role == "assistant" and ("### EXAMPLES ###" in content or "Normal Text:" in content):
                            continue
                        cleaned.append({"role": role, "content": content})
                    prev_a = None
                    for m in cleaned:
                        if m["role"] == "assistant":
                            cur_a = self._norm_text(m["content"])
                            if cur_a and cur_a == prev_a:
                                print("[MEM] Cleared repetitive history from memory.json.", flush=True)
                                return []
                            prev_a = cur_a
                    return cleaned[-max_msgs:]
            except Exception:
                pass
        return []

    def save_chat_history(self):
        if not getattr(self, "chat_memory_enabled", bool(CURRENT_CONFIG.get("chat_memory", True))):
            conv = []
        else:
            max_msgs = max(2, int(CURRENT_CONFIG.get("max_history_turns", 6)) * 2)
            conv = [
                m for m in (self.permanent_memory + self.session_memory)
                if isinstance(m, dict) and m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
            ][-max_msgs:]
        tmp_path = f"{MEMORY_FILE}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(conv, f, indent=4)
            os.replace(tmp_path, MEMORY_FILE)
        except Exception as e:
            print(f"[MEM ERROR] Save chat history failed: {e}", flush=True)

if __name__ == "__main__":
    print("--- SOCIAL ROBOT BMO SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
