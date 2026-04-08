# 📹 Video Surveillance System - Complete Implementation

**AI-Powered Detection, Tracking & Event Recognition Pipeline**

A production-ready system for processing security camera footage with real-time person detection, multi-object tracking, zone-based event detection, and comprehensive analytics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🌟 Features

### Core Capabilities

- ✅ **Person Detection**: YOLOv8-based real-time detection with configurable confidence thresholds
- ✅ **Multi-Object Tracking**: ByteTrack algorithm with robust ID persistence across occlusions
- ✅ **Event Detection**:
  - Zone intrusion detection with debouncing
  - Loitering detection with movement analysis
  - Configurable per-zone thresholds
- ✅ **Annotated Video Output**: Visual overlays with bounding boxes, track IDs, zones, and events
- ✅ **Structured Event Logs**: JSON and CSV export with video metadata
- ✅ **Production-Ready**: GPU/CPU support, error handling, comprehensive logging

### Advanced Features (Stretch Goals)

- 🎯 **Real-time FPS Dashboard**: Live matplotlib charts showing processing metrics
- 📦 **Batch Processing**: Process multiple videos sequentially or in parallel
- 🎨 **Interactive Zone Editor**: OpenCV-based GUI for visual zone definition
- 📊 **MOT Metrics Evaluation**: MOTA, MOTP, IDF1 benchmark metrics
- 🌐 **Web Dashboard**: Flask-based web interface for upload and monitoring

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Configuration](#️-configuration)
- [Model Selection](#-model-selection--justification)
- [Stretch Goals](#-stretch-goals-implemented)
- [Sample Results](#-sample-results)
- [Performance](#-performance-benchmarks)
- [Known Limitations](#️-known-limitations)
- [Development](#️-development)
- [Project Structure](#-project-structure)
- [Evaluation Coverage](#-evaluation-criteria-coverage)

---

## 🏗️ Architecture

### High-Level Pipeline

┌─────────────────────────────────────────────────────────────────┐
│ INPUT LAYER │
│ Video File(s) → Frame Extraction → Preprocessing │
└─────────────────────┬───────────────────────────────────────────┘
│
┌─────────────────────▼───────────────────────────────────────────┐
│ DETECTION LAYER │
│ YOLOv8 Detector → Person Bounding Boxes + Confidence │
│ • Pre-trained on COCO (80 classes) │
│ • Multiple model sizes (n/s/m/l/x) │
│ • GPU-accelerated inference │
└─────────────────────┬───────────────────────────────────────────┘
│
┌─────────────────────▼───────────────────────────────────────────┐
│ TRACKING LAYER │
│ ByteTrack Multi-Object Tracker → Unique Track IDs │
│ • Kalman filter motion prediction │
│ • Two-stage association (high + low confidence) │
│ • Handles occlusions and re-identification │
└─────────────────────┬───────────────────────────────────────────┘
│
┌─────────────────────▼───────────────────────────────────────────┐
│ EVENT DETECTION LAYER │
│ Zone Manager → Spatial Reasoning → Event Detectors │
│ • Point-in-polygon intersection (Shapely) │
│ • Intrusion: Zone entry with debouncing │
│ • Loitering: Stationary time + movement threshold │
│ • Event deduplication and aggregation │
└─────────────────────┬───────────────────────────────────────────┘
│
┌─────────────────────▼───────────────────────────────────────────┐
│ OUTPUT LAYER │
│ • Annotated Video (MP4/AVI) with overlays │
│ • Event Logs (JSON + CSV) with timestamps │
│ • Performance Metrics & Statistics │
│ • Optional: Real-time Dashboard, Web Interface │
└─────────────────────────────────────────────────────────────────┘

![Architecture Diagram](architecture_diagram.png)

text

### Component Breakdown

| Layer             | Technology         | Purpose               | Design Rationale                          |
| ----------------- | ------------------ | --------------------- | ----------------------------------------- |
| **Detection**     | YOLOv8             | Person detection      | SOTA speed/accuracy, easy integration     |
| **Tracking**      | ByteTrack          | Multi-object tracking | Better occlusion handling, no ReID needed |
| **Events**        | Shapely            | Spatial reasoning     | Robust polygon operations                 |
| **Visualization** | OpenCV             | Video annotation      | Fast, GPU-accelerated                     |
| **Configuration** | YAML + JSON        | Settings management   | Human-readable, version controllable      |
| **Dashboard**     | Matplotlib + Flask | Monitoring            | Real-time metrics, web accessible         |

---

## 💻 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **GPU** (optional): CUDA-capable GPU for 10x+ speedup
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk**: 2GB for dependencies + video storage

### Option 1: Quick Install (Recommended)

```bash
# Clone repository
git clone https://github.com/joelpaulehealth/surveillance-system.git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python run.py --help
```

### Option 2: Docker

```bash
# Build image
docker build -t surveillance-system .

# Run with GPU
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  surveillance-system \
  python run.py --video /app/data/sample.mp4

# Run CPU-only
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  surveillance-system \
  python run.py --video /app/data/sample.mp4 --device cpu
```

### Option 3: Automated Setup

chmod +x setup.sh
./setup.sh
source venv/bin/activate

# 🚀 Quick Start

1. Prepare Sample Video
   Bash

# Create data directory

mkdir -p data/videos

# Option A: Use your own video

cp /path/to/your/video.mp4 data/videos/sample.mp4

# Option B: Generate test video

python -c "
import cv2
import numpy as np

out = cv2.VideoWriter('data/videos/test.mp4',
cv2.VideoWriter_fourcc(\*'mp4v'),
30, (1280, 720))

for i in range(300):
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
x = (i \* 4) % 1200
cv2.rectangle(frame, (x, 300), (x+80, 420), (0, 255, 0), -1)
out.write(frame)

out.release()
print('✓ Test video created')
" 2. Run Basic Processing
Bash

# Process video with default settings

python run.py --video data/videos/sample.mp4

# Output generated in:

# - output/videos/sample_annotated.mp4

# - output/events/sample\_\*\_events.json

# - output/events/sample\_\*\_events.csv

3. View Results
   Bash

# Open annotated video

open output/videos/sample_annotated.mp4

# View events

cat output/events/\*\_events.json | python -m json.tool

# Summary

cat output/events/\*\_events.json | jq '.summary'
📖 Usage Examples
Basic Usage
Bash

# Process single video

python run.py --video input.mp4

# Specify output directory

python run.py --video input.mp4 --output results/

# Custom configuration

python run.py --video input.mp4 --config configs/custom.yaml

# Custom zones

python run.py --video input.mp4 --zones configs/my_zones.json
Device Selection
Bash

# Force CPU mode

python run.py --video input.mp4 --device cpu

# Use GPU (auto if available)

python run.py --video input.mp4 --device cuda
Adjust Detection
Bash

# Lower threshold (more detections)

python run.py --video input.mp4 --confidence 0.3

# Higher threshold (more precise)

python run.py --video input.mp4 --confidence 0.7
Performance
Bash

# Skip video output (faster)

python run.py --video input.mp4 --no-save

# Benchmark mode

python run.py --video input.mp4 --benchmark

# Quiet mode

python run.py --video input.mp4 --quiet
Advanced Features
Bash

# Real-time dashboard

python run.py --video input.mp4 --dashboard

# Batch processing

python run.py --batch video1.mp4 --batch video2.mp4 --batch video3.mp4
