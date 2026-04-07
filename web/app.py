"""
Web dashboard for surveillance system monitoring.

Provides:
- File upload interface
- Real-time processing status
- Results visualization
- Event log viewing
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from pathlib import Path
import os
import json
import threading
from datetime import datetime
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import VideoProcessor, BatchProcessor
from src.utils import Config

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('web/uploads')
OUTPUT_FOLDER = Path('output/web')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Processing state
processing_jobs = {}
job_counter = 0


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_async(job_id: str, video_path: str, config: Config):
    """Process video in background thread."""
    try:
        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['progress'] = 0
        
        # Process video
        processor = VideoProcessor(config)
        result = processor.process_video(
            input_path=video_path,
            output_path=str(OUTPUT_FOLDER / f"{job_id}_annotated.mp4"),
            save_video=True,
            show_progress=False
        )
        
        processing_jobs[job_id]['status'] = 'completed'
        processing_jobs[job_id]['progress'] = 100
        processing_jobs[job_id]['result'] = result
        processing_jobs[job_id]['completed_at'] = datetime.now().isoformat()
    
    except Exception as e:
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['error'] = str(e)


@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle video file upload."""
    global job_counter
    
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_id = f"job_{timestamp}_{job_counter}"
    job_counter += 1
    
    video_path = UPLOAD_FOLDER / f"{job_id}_{filename}"
    file.save(video_path)
    
    # Initialize job
    processing_jobs[job_id] = {
        'id': job_id,
        'filename': filename,
        'status': 'queued',
        'progress': 0,
        'created_at': datetime.now().isoformat(),
        'video_path': str(video_path)
    }
    
    # Start processing in background
    config = Config(config_path='configs/default_config.yaml')
    thread = threading.Thread(
        target=process_video_async,
        args=(job_id, str(video_path), config)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id, 'status': 'queued'})


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all processing jobs."""
    return jsonify({
        'jobs': list(processing_jobs.values())
    })


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get job status."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_jobs[job_id])


@app.route('/api/jobs/<job_id>/video', methods=['GET'])
def download_video(job_id):
    """Download processed video."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Video not ready'}), 400
    
    video_path = OUTPUT_FOLDER / f"{job_id}_annotated.mp4"
    
    if not video_path.exists():
        return jsonify({'error': 'Video file not found'}), 404
    
    return send_file(video_path, as_attachment=True)


@app.route('/api/jobs/<job_id>/events', methods=['GET'])
def get_events(job_id):
    """Get event log for job."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Processing not complete'}), 400
    
    # Find event JSON file
    event_files = list(OUTPUT_FOLDER.glob(f"*{job_id}*events.json"))
    
    if not event_files:
        return jsonify({'error': 'Event log not found'}), 404
    
    with open(event_files[0], 'r') as f:
        events_data = json.load(f)
    
    return jsonify(events_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)