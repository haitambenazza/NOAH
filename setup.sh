#!/bin/bash
set -e

NEEDS_REBOOT=0
HAILO_GENAI_DEB_URL="${HAILO_GENAI_DEB_URL:-https://dev-public.hailo.ai/2025_12/Hailo10/hailo_gen_ai_model_zoo_5.1.1_arm64.deb}"
HAILO_MODEL="${HAILO_MODEL:-qwen2.5-instruct:1.5b}"

# 1. System & Hardware Dependencies (Raspberry Pi 5 ARM64 + AI HAT+ 2 Hailo-10H)
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
    dkms \
    curl \
    tar

echo "[SETUP] Installing Hailo-10H (AI HAT+ 2, 40 TOPS) driver & runtime..."
sudo apt install -y hailo-h10-all

# 2. PCIe Gen 3 Configuration for Raspberry Pi 5 + AI HAT+ 2
BOOT_CONFIG="/boot/firmware/config.txt"
if [ -f "$BOOT_CONFIG" ]; then
    if ! grep -Eq '^[[:space:]]*dtparam=pciex1(=on)?[[:space:]]*(#.*)?$' "$BOOT_CONFIG"; then
        echo "[SETUP] Enabling PCIe in $BOOT_CONFIG..."
        echo "dtparam=pciex1=on" | sudo tee -a "$BOOT_CONFIG" >/dev/null
        NEEDS_REBOOT=1
    fi
    if ! grep -Eq '^[[:space:]]*dtparam=pciex1_gen=3[[:space:]]*(#.*)?$' "$BOOT_CONFIG"; then
        echo "[SETUP] Enabling PCIe Gen 3 in $BOOT_CONFIG..."
        echo "dtparam=pciex1_gen=3" | sudo tee -a "$BOOT_CONFIG" >/dev/null
        NEEDS_REBOOT=1
    fi
fi

# 3. Piper TTS (ARM64 / aarch64 Binary & Voice Model)
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

# 4. Hailo-10H NPU & hailo-ollama LLM Server Setup (replaces CPU Ollama)
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files ollama.service 2>/dev/null | grep -q '^ollama\.service'; then
    echo "[SETUP] Disabling CPU-only ollama.service to free RAM/CPU..."
    sudo systemctl disable --now ollama || true
fi

HAILO_READY=0
if [ -e /dev/hailo0 ]; then
    echo "[HAILO] /dev/hailo0 detected."
    if command -v hailortcli >/dev/null 2>&1; then
        hailortcli fw-control identify || true
    fi
    HAILO_READY=1
else
    echo "[WARNING] /dev/hailo0 not found yet — a reboot is required for the hailo-h10-all PCIe driver to load."
    NEEDS_REBOOT=1
fi

if ! command -v hailo-ollama >/dev/null 2>&1; then
    DEB_PKG=""
    for candidate in ./hailo_gen_ai_model_zoo_*_arm64.deb "$HOME"/Downloads/hailo_gen_ai_model_zoo_*_arm64.deb; do
        if [ -f "$candidate" ]; then
            DEB_PKG="$candidate"
            break
        fi
    done

    if [ -n "$DEB_PKG" ]; then
        echo "[HAILO] Installing local GenAI Model Zoo package: $DEB_PKG..."
        sudo apt install -y "$DEB_PKG" || (sudo dpkg -i "$DEB_PKG" || sudo apt install -f -y)
    elif apt-cache show hailo-gen-ai-model-zoo >/dev/null 2>&1; then
        echo "[HAILO] Installing hailo-gen-ai-model-zoo via apt..."
        sudo apt install -y hailo-gen-ai-model-zoo
    elif apt-cache show hailo-ollama >/dev/null 2>&1; then
        echo "[HAILO] Installing hailo-ollama via apt..."
        sudo apt install -y hailo-ollama
    else
        TMP_DEB="/tmp/hailo_gen_ai_model_zoo_arm64.deb"
        echo "[HAILO] Downloading Hailo GenAI Model Zoo (hailo-ollama) from $HAILO_GENAI_DEB_URL..."
        if curl -fL -o "$TMP_DEB" "$HAILO_GENAI_DEB_URL"; then
            sudo apt install -y "$TMP_DEB" || (sudo dpkg -i "$TMP_DEB" || sudo apt install -f -y)
        else
            echo "[WARNING] Failed to download $HAILO_GENAI_DEB_URL."
        fi
        rm -f "$TMP_DEB"
    fi
fi

if ! command -v hailo-ollama >/dev/null 2>&1; then
    echo "[WARNING] hailo-ollama is not installed yet."
    echo "          Download hailo_gen_ai_model_zoo_*_arm64.deb from Hailo Developer Zone"
    echo "          (see Raspberry Pi AI HAT+ 2 docs), place it in $(pwd) or ~/Downloads,"
    echo "          install with: sudo dpkg -i hailo_gen_ai_model_zoo_*_arm64.deb"
    echo "          and re-run ./setup.sh."
elif [ "$HAILO_READY" -eq 1 ]; then
    HAILO_OLLAMA_BIN="$(command -v hailo-ollama)"
    if command -v systemctl >/dev/null 2>&1; then
        if [ ! -f /etc/systemd/system/hailo-ollama.service ] && ! systemctl list-unit-files hailo-ollama.service 2>/dev/null | grep -q '^hailo-ollama\.service'; then
            echo "[HAILO] Creating /etc/systemd/system/hailo-ollama.service..."
            sudo tee /etc/systemd/system/hailo-ollama.service >/dev/null <<EOF
[Unit]
Description=Hailo Ollama Local LLM Service for Hailo-10H
After=network.target

[Service]
Type=simple
User=root
Environment="OLLAMA_HOST=127.0.0.1:8000"
ExecStart=${HAILO_OLLAMA_BIN}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
            sudo systemctl daemon-reload
        fi
        sudo systemctl enable --now hailo-ollama
    fi

    HAILO_URL="http://127.0.0.1:8000"
    SERVER_UP=0
    for _ in $(seq 1 10); do
        if curl -sf "${HAILO_URL}/api/tags" >/dev/null 2>&1 || curl -sf "${HAILO_URL}/hailo/v1/list" >/dev/null 2>&1; then
            SERVER_UP=1
            break
        fi
        sleep 1
    done

    if [ "$SERVER_UP" -eq 1 ]; then
        echo "[HAILO] Pulling Hailo NPU model (${HAILO_MODEL})..."
        if ! curl -sf -X POST "${HAILO_URL}/api/pull" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"${HAILO_MODEL}\", \"stream\": false}"; then
            echo "[WARNING] Could not pull '${HAILO_MODEL}'. Check available models with: curl -s ${HAILO_URL}/hailo/v1/list"
        fi
        echo ""
    else
        echo "[WARNING] hailo-ollama did not respond on ${HAILO_URL}. Check: sudo systemctl status hailo-ollama"
    fi
fi

if [ "$NEEDS_REBOOT" -eq 1 ]; then
    echo "===================================================="
    echo " REBOOT REQUIRED FOR AI HAT+ 2 (HAILO-10H) DRIVER"
    echo "===================================================="
    echo " 1. Run: sudo reboot"
    echo " 2. Verify NPU: hailortcli fw-control identify"
    echo " 3. Re-run: ./setup.sh  (to start hailo-ollama and pull the model)"
    echo "===================================================="
fi