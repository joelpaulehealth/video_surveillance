"""
Real-time FPS monitoring dashboard using matplotlib.

Design Choice:
- Non-blocking matplotlib animation for live updates
- Tracks multiple metrics: FPS, detection count, track count
- Can run in separate thread or process
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from collections import deque
from typing import Dict, Optional
import threading
import time

from .logger import get_logger

logger = get_logger(__name__)


class FPSDashboard:
    """
    Real-time FPS and metrics dashboard.
    
    Displays live charts of:
    - Processing FPS over time
    - Detection count per frame
    - Active track count
    - Event count
    """
    
    def __init__(
        self,
        max_points: int = 100,
        update_interval: int = 100,
        enable: bool = True
    ):
        """
        Initialize FPS dashboard.
        
        Args:
            max_points: Maximum data points to display
            update_interval: Update interval in milliseconds
            enable: Whether dashboard is enabled
        """
        self._enable = enable
        if not enable:
            return
        
        self._max_points = max_points
        self._update_interval = update_interval
        
        # Data buffers
        self._fps_data = deque(maxlen=max_points)
        self._detection_data = deque(maxlen=max_points)
        self._track_data = deque(maxlen=max_points)
        self._event_data = deque(maxlen=max_points)
        self._time_data = deque(maxlen=max_points)
        
        # Thread safety
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Matplotlib setup
        self._fig = None
        self._axes = None
        self._lines = None
        self._anim = None
        
        # Running state
        self._is_running = False
        
        logger.info("FPS Dashboard initialized")
    
    def start(self) -> None:
        """Start the dashboard in non-blocking mode."""
        if not self._enable or self._is_running:
            return
        
        # Create figure and subplots
        self._fig, self._axes = plt.subplots(2, 2, figsize=(12, 8))
        self._fig.suptitle('Video Processing Dashboard', fontsize=16)
        
        # Configure subplots
        ax_fps = self._axes[0, 0]
        ax_det = self._axes[0, 1]
        ax_track = self._axes[1, 0]
        ax_event = self._axes[1, 1]
        
        # FPS plot
        ax_fps.set_title('Processing FPS')
        ax_fps.set_xlabel('Time (s)')
        ax_fps.set_ylabel('FPS')
        ax_fps.grid(True, alpha=0.3)
        line_fps, = ax_fps.plot([], [], 'b-', linewidth=2, label='FPS')
        ax_fps.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='30 FPS')
        ax_fps.axhline(y=24, color='orange', linestyle='--', alpha=0.5, label='24 FPS')
        ax_fps.legend()
        
        # Detection count plot
        ax_det.set_title('Detections per Frame')
        ax_det.set_xlabel('Time (s)')
        ax_det.set_ylabel('Count')
        ax_det.grid(True, alpha=0.3)
        line_det, = ax_det.plot([], [], 'r-', linewidth=2, label='Detections')
        ax_det.legend()
        
        # Track count plot
        ax_track.set_title('Active Tracks')
        ax_track.set_xlabel('Time (s)')
        ax_track.set_ylabel('Count')
        ax_track.grid(True, alpha=0.3)
        line_track, = ax_track.plot([], [], 'g-', linewidth=2, label='Tracks')
        ax_track.legend()
        
        # Event count plot
        ax_event.set_title('Cumulative Events')
        ax_event.set_xlabel('Time (s)')
        ax_event.set_ylabel('Count')
        ax_event.grid(True, alpha=0.3)
        line_event, = ax_event.plot([], [], 'm-', linewidth=2, label='Events')
        ax_event.legend()
        
        self._lines = {
            'fps': line_fps,
            'detection': line_det,
            'track': line_track,
            'event': line_event
        }
        
        # Create animation
        self._anim = animation.FuncAnimation(
            self._fig,
            self._update_plots,
            interval=self._update_interval,
            blit=False
        )
        
        self._is_running = True
        
        # Show in non-blocking mode
        plt.ion()
        plt.show(block=False)
        
        logger.info("FPS Dashboard started")
    
    def update(self, metrics: Dict) -> None:
        """
        Update dashboard with new metrics.
        
        Args:
            metrics: Dictionary with keys: fps, detections, tracks, events
        """
        if not self._enable:
            return
        
        with self._lock:
            current_time = time.time() - self._start_time
            
            self._time_data.append(current_time)
            self._fps_data.append(metrics.get('fps', 0))
            self._detection_data.append(metrics.get('detections', 0))
            self._track_data.append(metrics.get('tracks', 0))
            self._event_data.append(metrics.get('events', 0))
    
    def _update_plots(self, frame) -> None:
        """Update plot data (called by animation)."""
        if not self._enable or not self._is_running:
            return
        
        with self._lock:
            if len(self._time_data) == 0:
                return
            
            time_array 