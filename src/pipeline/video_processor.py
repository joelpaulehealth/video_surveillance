"""
Main video processing pipeline orchestrator.

Design Choice:
- Single class coordinates all pipeline components
- Frame-by-frame processing for memory efficiency
- Progress tracking and FPS benchmarking
- Clean separation of detection, tracking, events, and output
"""

import time
from typing import Optional, Dict, Any, Generator, Tuple
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from ..detection import YOLODetector
from ..tracking import ByteTracker, Track
from ..events import EventManager
from ..output import VideoWriter, EventLogger
from ..utils import Config, Visualizer, setup_logger, get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """
    Main video processing pipeline.
    
    Orchestrates detection, tracking, event detection, and output generation.
    Provides progress tracking and performance metrics.
    """
    
    def __init__(self, config: Config):
        """
        Initialize video processor with configuration.
        
        Args:
            config: System configuration
        """
        self._config = config
        
        # Setup logging
        setup_logger(
            log_level=config.logging.level,
            log_file=config.logging.log_file
        )
        
        # Initialize components (lazy loading)
        self._detector: Optional[YOLODetector] = None
        self._tracker: Optional[ByteTracker] = None
        self._event_manager: Optional[EventManager] = None
        self._visualizer: Optional[Visualizer] = None
        
        # Processing state
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._video_writer: Optional[VideoWriter] = None
        self._event_logger: Optional[EventLogger] = None
        
        # Metrics
        self._metrics = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'processing_time': 0.0,
            'fps_history': []
        }
        
        logger.info("VideoProcessor initialized")
    
    def _init_components(self, fps: float) -> None:
        """Initialize processing components."""
        # Detector
        self._detector = YOLODetector(
            model_path=self._config.detection.model,
            confidence_threshold=self._config.detection.confidence_threshold,
            iou_threshold=self._config.detection.iou_threshold,
            device=self._config.detection.device,
            classes=self._config.detection.classes,
            max_detections=self._config.detection.max_detections
        )
        self._detector.warmup()
        
        # Tracker
        self._tracker = ByteTracker(
            max_age=self._config.tracking.max_age,
            min_hits=self._config.tracking.min_hits,
            iou_threshold=self._config.tracking.iou_threshold
        )
        
        # Event manager
        self._event_manager = EventManager(self._config, fps=fps)
        
        # Visualizer
        self._visualizer = Visualizer(self._config.visualization)
        
        logger.info("All components initialized")
    
    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        save_video: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process a video file through the complete pipeline.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video (optional)
            save_video: Whether to save annotated video
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with processing results and metrics
        """
        logger.info(f"Processing video: {input_path}")
        
        # Open video
        self._video_capture = cv2.VideoCapture(input_path)
        if not self._video_capture.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        
        # Get video properties
        fps = self._video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Apply resize if configured
        if self._config.video.resize_width:
            scale = self._config.video.resize_width / width
            width = self._config.video.resize_width
            height = int(height * scale)
        
        # Initialize components
        self._init_components(fps)
        
        # Initialize output
        if save_video:
            if output_path is None:
                output_path = self._config.video.output_path
            
            self._video_writer = VideoWriter(
                output_path=output_path,
                fps=fps,
                frame_size=(width, height)
            )
        
        # Initialize event logger
        video_name = Path(input_path).name
        output_dir = Path(output_path).parent if output_path else Path("output/events")
        
        self._event_logger = EventLogger(
            output_dir=str(output_dir),
            video_name=video_name,
            fps=fps,
            total_frames=total_frames
        )
        
        # Process frames
        self._metrics['total_frames'] = total_frames
        start_time = time.time()
        
        progress_bar = tqdm(
            total=total_frames,
            desc="Processing",
            disable=not show_progress
        )
        
        try:
            for frame_data in self._frame_generator():
                frame_number, frame, timestamp = frame_data
                
                # Process frame
                result = self._process_frame(frame, frame_number, timestamp)
                
                # Draw annotations and write output
                if save_video and self._video_writer:
                    annotated = self._visualizer.draw_frame(
                        frame=result['frame'],
                        detections=result['tracks_data'],
                        zones=self._config.zones,
                        events=result['events'],
                        fps=result['fps'],
                        frame_number=frame_number
                    )
                    self._video_writer.write_frame(annotated)
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'FPS': f"{result['fps']:.1f}",
                    'Tracks': len(result['tracks_data']),
                    'Events': len(result['events'])
                })
        
        finally:
            progress_bar.close()
            # self._cleanup()
        
        # Calculate final metrics
        self._metrics['processing_time'] = time.time() - start_time
        avg_fps = self._metrics['processed_frames'] / self._metrics['processing_time']
        
        # Save events
        all_events = self._event_manager.get_all_events()
        event_files = self._event_logger.save(
            events=all_events,
            metadata={
                'detector': self._detector.model_name,
                'device': self._detector.device,
                'config': self._config.to_dict()
            }
        )
        
        # Compile results
        results = {
            'video_path': input_path,
            'output_video': str(self._video_writer.output_path) if self._video_writer else None,
            'event_files': {k: str(v) for k, v in event_files.items()},
            'metrics': {
                'total_frames': total_frames,
                'processed_frames': self._metrics['processed_frames'],
                'total_detections': self._metrics['total_detections'],
                'unique_tracks': self._tracker.track_count,
                'total_events': len(all_events),
                'processing_time_seconds': round(self._metrics['processing_time'], 2),
                'average_fps': round(avg_fps, 2)
            },
            'events_summary': self._event_manager.get_events_summary()
        }
        
        logger.info(f"Processing complete: {results['metrics']}")
        
        self._cleanup()
        return results
    
    def _frame_generator(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Generate frames from video with frame skipping support.
        
        Yields:
            Tuple of (frame_number, frame, timestamp)
        """
        frame_number = 0
        frame_skip = self._config.video.frame_skip
        fps = self._video_capture.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = self._video_capture.read()
            
            if not ret:
                break
            
            # Apply frame skipping
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue
            
            # Resize if configured
            if self._config.video.resize_width:
                frame = cv2.resize(
                    frame,
                    (self._config.video.resize_width, 
                     int(frame.shape[0] * self._config.video.resize_width / frame.shape[1]))
                )
            
            timestamp = frame_number / fps
            
            yield frame_number, frame, timestamp
            
            frame_number += 1
    
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Process a single frame through detection, tracking, and events.
        
        Args:
            frame: Input frame
            frame_number: Frame number
            timestamp: Video timestamp
            
        Returns:
            Dictionary with processing results
        """
        frame_start = time.time()
        
        # Detection
        detections = self._detector.detect(frame)
        self._metrics['total_detections'] += len(detections)
        
        # Prepare detections for tracker
        det_dicts = [
            {
                'bbox': list(d.bbox),
                'confidence': d.confidence
            }
            for d in detections
        ]
        
        # Tracking
        tracks = self._tracker.update(det_dicts, frame)
        self._metrics['total_tracks'] = self._tracker.track_count
        
        # Convert tracks to dicts for event processing
        tracks_data = [
            {
                'track_id': t.track_id,
                'bbox': list(t.bbox),
                'confidence': t.confidence,
                'in_zone': False  # Will be updated by event manager
            }
            for t in tracks
        ]
        
        # Event detection
        events = self._event_manager.process_frame(
            tracks=tracks_data,
            frame_number=frame_number,
            timestamp=timestamp
        )
        
        # Calculate FPS
        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        self._metrics['fps_history'].append(current_fps)
        
        # Keep only recent FPS values for smoothing
        if len(self._metrics['fps_history']) > 30:
            self._metrics['fps_history'].pop(0)
        
        avg_fps = sum(self._metrics['fps_history']) / len(self._metrics['fps_history'])
        
        self._metrics['processed_frames'] += 1
        
        # Update dashboard if enabled
        if hasattr(self, '_dashboard') and self._dashboard:
            self._dashboard.update({
                'fps': avg_fps,
                'detections': len(detections),
                'tracks': len(tracks_data),
                'events': len(self._event_manager.get_all_events())
            })
            
        logger.debug(f"Frame {frame_number}: {len(events)} events detected")
        return {
            'frame': frame,
            'detections': detections,
            'tracks': tracks,
            'tracks_data': tracks_data,
            'events': self._event_manager.get_all_events(),
            'fps': avg_fps
        }
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._video_capture:
            self._video_capture.release()
            self._video_capture = None
        
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        
        if self._tracker:
            self._tracker.reset()
        
        if self._event_manager:
            self._event_manager.reset()
        
        if self._visualizer:
            self._visualizer.clear_track_history()