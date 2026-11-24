#!/bin/bash
set -e

# 1. System & Hardware Dependencies (Raspberry Pi 5 ARM64)
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-tk \
    python3-pil.imagetk \
    python3-lgpio \
    python3-picamera2 \
    rpicam-apps \
    v4l-utils \
    swig \
    liblgpio-dev \
    libportaudio2 \
    portaudio19-dev \
    curl \
    tar

# 2. Piper TTS (ARM64 / aarch64 Binary & Voice Model)
mkdir -p piper
if [ ! -f "./piper/piper" ]; then
    echo "[SETUP] Downloading Piper TTS (linux_aarch64) for Raspberry Pi 5..."
    curl -L https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz | tar -xz -C .
    chmod +x ./piper/piper
fi

if [ ! -f "./piper/en_GB-semaine-medium.onnx" ]; then
    echo "[SETUP] Downloading Piper voice model (en_GB-semaine-medium)..."
    curl -L -o piper/en_GB-semaine-medium.onnx \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx
    curl -L -o piper/en_GB-semaine-medium.onnx.json \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json
fi

# 3. Ollama LLM Server & Default Model (gemma3:1b)
if ! command -v ollama >/dev/null 2>&1; then
    echo "[SETUP] Installing Ollama for ARM64..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

sudo systemctl enable --now ollama
sleep 2
ollama pull gemma3:1b