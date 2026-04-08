#!/usr/bin/env python3
"""
Debug script to identify why events aren't being logged.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import json

from src.detection import YOLODetector
from src.tracking import ByteTracker
from src.events import EventManager
from src.utils import Config


def debug_pipeline(video_path: str, zones_path: str = None, num_frames: int = 100):
    """Debug each stage of the pipeline."""
    
    print("=" * 60)
    print("PIPELINE DEBUG")
    print("=" * 60)
    
    # Load config
    config = Config(
        config_path='configs/default_config.yaml',
        zones_path=zones_path
    )
    
    # Print zone info
    print(f"\n1. ZONES CONFIGURATION")
    print("-" * 40)
    print(f"   Zones file: {zones_path or 'default'}")
    print(f"   Number of zones: {len(config.zones)}")
    
    if len(config.zones) == 0:
        print("   ⚠️  NO ZONES DEFINED - Events will not be detected!")
    else:
        for zone in config.zones:
            print(f"   - {zone.name} ({zone.type}): {zone.polygon[:2]}...")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n2. VIDEO INFO")
    print("-" * 40)
    print(f"   Path: {video_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Testing first {num_frames} frames")
    
    # Initialize components
    print(f"\n3. DETECTION")
    print("-" * 40)
    
    detector = YOLODetector(
        model_path=config.detection.model,
        confidence_threshold=config.detection.confidence_threshold,
        device='cpu'  # Force CPU for debugging
    )
    print(f"   Model: {config.detection.model}")
    print(f"   Confidence threshold: {config.detection.confidence_threshold}")
    print(f"   Device: {detector.device}")
    
    # Initialize tracker
    print(f"\n4. TRACKING")
    print("-" * 40)
    
    tracker = ByteTracker(
        max_age=config.tracking.max_age,
        min_hits=config.tracking.min_hits,
        iou_threshold=config.tracking.iou_threshold
    )
    print(f"   Max age: {config.tracking.max_age}")
    print(f"   Min hits: {config.tracking.min_hits}")
    print(f"   IoU threshold: {config.tracking.iou_threshold}")
    
    # Initialize event manager
    print(f"\n5. EVENT DETECTION")
    print("-" * 40)
    
    event_manager = EventManager(config, fps=fps)
    print(f"   Intrusion zones: {len(config.get_intrusion_zones())}")
    print(f"   Loitering zones: {len(config.get_loitering_zones())}")
    
    # Process frames
    print(f"\n6. PROCESSING FRAMES")
    print("-" * 40)
    
    frame_detections = []
    frame_tracks = []
    all_events = []
    
    for frame_num in range(min(num_frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame)
        det_dicts = [
            {'bbox': list(d.bbox), 'confidence': d.confidence}
            for d in detections
        ]
        
        # Track
        tracks = tracker.update(det_dicts, frame)
        
        # Events
        tracks_data = [
            {
                'track_id': t.track_id,
                'bbox': list(t.bbox),
                'confidence': t.confidence,
                'in_zone': False
            }
            for t in tracks
        ]
        
        timestamp = frame_num / fps
        events = event_manager.process_frame(tracks_data, frame_num, timestamp)
        
        # Log progress
        if frame_num % 10 == 0:
            print(f"   Frame {frame_num}: {len(detections)} detections, {len(tracks)} tracks, {len(events)} events")
        
        frame_detections.append(len(detections))
        frame_tracks.append(len(tracks))
        all_events.extend(events)
    
    cap.release()
    
    # Summary
    print(f"\n7. SUMMARY")
    print("-" * 40)
    print(f"   Total detections: {sum(frame_detections)}")
    print(f"   Avg detections/frame: {sum(frame_detections)/len(frame_detections):.1f}")
    print(f"   Total tracks observed: {sum(frame_tracks)}")
    print(f"   Avg tracks/frame: {sum(frame_tracks)/len(frame_tracks):.1f}")
    print(f"   Unique track IDs: {tracker.track_count}")
    print(f"   Total events: {len(all_events)}")
    
    # Diagnose issues
    print(f"\n8. DIAGNOSIS")
    print("-" * 40)
    
    if sum(frame_detections) == 0:
        print("   ❌ PROBLEM: No detections")
        print("   → Try lowering confidence_threshold to 0.3")
        print("   → Check if video contains people")
    
    elif sum(frame_tracks) == 0:
        print("   ❌ PROBLEM: Detections exist but no tracks")
        print("   → Issue is in ByteTracker")
        print("   → Check if detections are high confidence (>0.5)")
        print("   → Try lowering min_hits to 1")
        
        # Check confidence distribution
        all_confs = []
        cap = cv2.VideoCapture(video_path)
        for i in range(min(10, total_frames)):
            ret, frame = cap.read()
            if ret:
                dets = detector.detect(frame)
                all_confs.extend([d.confidence for d in dets])
        cap.release()
        
        if all_confs:
            print(f"\n   Detection confidence distribution:")
            print(f"   - Min: {min(all_confs):.3f}")
            print(f"   - Max: {max(all_confs):.3f}")
            print(f"   - Avg: {sum(all_confs)/len(all_confs):.3f}")
            
            high_conf = [c for c in all_confs if c >= 0.5]
            print(f"   - High confidence (>=0.5): {len(high_conf)}/{len(all_confs)} ({100*len(high_conf)/len(all_confs):.1f}%)")
    
    elif len(config.zones) == 0:
        print("   ❌ PROBLEM: No zones defined")
        print("   → Create zones using zone editor or JSON file")
        print("   → Run: python tools/zone_editor.py --video <video>")
    
    elif len(all_events) == 0:
        print("   ⚠️  PROBLEM: Tracks exist but no events")
        print("   → Check if zones are positioned correctly")
        print("   → People might not be entering zones")
        print("   → Try adjusting zone positions")
        
        # Check zone occupancy
        print(f"\n   Zone coverage check:")
        for zone in config.zones:
            print(f"   - {zone.name}: polygon={zone.polygon}")
    
    else:
        print("   ✅ Pipeline working correctly!")
        print(f"   Events detected: {len(all_events)}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Video path')
    parser.add_argument('--zones', default=None, help='Zones JSON path')
    parser.add_argument('--frames', type=int, default=100, help='Frames to test')
    
    args = parser.parse_args()
    debug_pipeline(args.video, args.zones, args.frames)