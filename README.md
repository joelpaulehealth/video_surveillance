# Video Surveillance System

**Detection, Tracking & Event Recognition Pipeline**

A production-ready system for processing security camera footage to detect people, track them across frames, and identify events of interest (zone intrusion, loitering).

---

## 🚀 Features

- **Person Detection**: YOLOv8-based real-time detection
- **Multi-Object Tracking**: ByteTrack algorithm with ID persistence
- **Event Detection**:
  - Zone intrusion detection
  - Loitering detection with configurable thresholds
- **Configurable Zones**: JSON-based zone definitions
- **Annotated Video Output**: Visual overlay of detections, tracks, and events
- **Structured Event Logs**: JSON and CSV export
- **Production-Ready**: GPU/CPU support, error handling, logging

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Model Selection](#model-selection)
- [Sample Results](#sample-results)
- [Known Limitations](#known-limitations)
- [Performance](#performance)
- [Development](#development)

---

## 🏗️ Architecture

┌─────────────────────────────────────────────────────────┐
│ INPUT: Video File │
└────────────────────┬────────────────────────────────────┘
│
┌────────────────────▼────────────────────────────────────┐
│ DETECTION: YOLOv8 Person Detector │
│ → Bounding boxes + confidence scores │
└────────────────────┬────────────────────────────────────┘
│
┌────────────────────▼────────────────────────────────────┐
│ TRACKING: ByteTrack Multi-Object Tracker │
│ → Unique IDs + motion prediction │
└────────────────────┬────────────────────────────────────┘
│
┌────────────────────▼────────────────────────────────────┐
│ EVENTS: Zone-Based Event Detection │
│ → Intrusion + Loitering detection │
└────────────────────┬────────────────────────────────────┘
│
┌────────────────────▼────────────────────────────────────┐
│ OUTPUT: Annotated Video + Event Logs │
│ → MP4 video + JSON/CSV logs │
└─────────────────────────────────────────────────────────┘

### Component Overview

| Component           | Technology             | Purpose                            |
| ------------------- | ---------------------- | ---------------------------------- |
| **Detector**        | YOLOv8 (Ultralytics)   | Fast, accurate person detection    |
| **Tracker**         | ByteTrack              | Robust ID assignment across frames |
| **Event Detection** | Custom logic + Shapely | Zone-based spatial reasoning       |
| **Visualization**   | OpenCV                 | Annotated video output             |
| **Configuration**   | YAML + JSON            | Human-editable settings            |

---

## 💻 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for better performance)
- 4GB+ RAM

### Option 1: pip (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/surveillance-system.git
cd surveillance-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python run.py --help

```

### Option 2: Docker

```
# Build image
docker build -t surveillance-system .

# Run with GPU support
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
  surveillance-system python run.py --video /app/data/sample.mp4

# Run CPU-only
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
  surveillance-system python run.py --video /app/data/sample.mp4 --device cpu
```
