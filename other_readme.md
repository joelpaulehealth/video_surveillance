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
