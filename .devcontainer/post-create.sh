#!/bin/sh
set -e

# 1. 시스템 패키지 업데이트 및 기본 도구 설치
echo "Updating apt and installing system packages..."
apt-get update
apt-get install -y python3 python3-pip python3-dev git build-essential libgl1

# 2. Pip 업그레이드
echo "Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# 3. PyTorch (CUDA 지원 버전) 설치
echo "Installing PyTorch for CUDA..."
python3 -m pip install --no-cache-dir torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. YOLO 및 필수 라이브러리 설치
echo "Installing Ultralytics YOLO and essential libraries..."
python3 -m pip install --no-cache-dir ultralytics matplotlib opencv-python-headless

echo "✅ All essential packages for YOLO have been installed successfully!"