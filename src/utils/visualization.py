"""
Visualization utilities for drawing annotations on video frames.

Design Choice:
- Using OpenCV for drawing (fast, GPU-accelerated when available)
- Color-coding by state (normal, in zone, event active)
- Semi-transparent overlays for zones to maintain visibility
- Track history trails for motion visualization
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import cv2
import numpy as np

from .config import ZoneDefinition, VisualizationConfig
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColorPalette:
    """Color definitions for visualization elements."""
    # BGR format for OpenCV
    DETECTION_NORMAL: Tuple[int, int, int] = (0, 255, 0)      # Green
    DETECTION_IN_ZONE: Tuple[int, int, int] = (0, 165, 255)   # Orange
    DETECTION_EVENT: Tuple[int, int, int] = (0, 0, 255)       # Red
    
    ZONE_INTRUSION: Tuple[int, int, int] = (0, 0, 255)        # Red
    ZONE_LOITERING: Tuple[int, int, int] = (0, 255, 255)      # Yellow
    ZONE_NORMAL: Tuple[int, int, int] = (255, 255, 0)         # Cyan
    
    TRACK_TRAIL: Tuple[int, int, int] = (255, 0, 255)         # Magenta
    TEXT_BG: Tuple[int, int, int] = (0, 0, 0)                 # Black
    TEXT_FG: Tuple[int, int, int] = (255, 255, 255)           # White
    
    FPS_GOOD: Tuple[int, int, int] = (0, 255, 0)              # Green (>24 fps)
    FPS_OK: Tuple[int, int, int] = (0, 255, 255)              # Yellow (12-24 fps)
    FPS_BAD: Tuple[int, int, int] = (0, 0, 255)               # Red (<12 fps)


class Visualizer:
    """
    Handles all visualization/drawing operations on video frames.
    
    Maintains track history for trail visualization and provides
    consistent styling across all annotation elements.
    """
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration settings
        """
        self.config = config
        self.colors = ColorPalette()
        
        # Track history for trails: {track_id: [(x, y), ...]}
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        # Active events for highlighting
        self.active_events: Dict[int, str] = {}  # {track_id: event_type}
        
        logger.debug("Visualizer initialized")
    
    def draw_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        zones: List[ZoneDefinition],
        events: List[Dict],
        fps: Optional[float] = None,
        frame_number: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw all annotations on a frame.
        
        Args:
            frame: Input frame (BGR)
            detections: List of detection dicts with bbox, track_id, confidence
            zones: List of zone definitions
            events: List of active events in this frame
            fps: Current processing FPS
            frame_number: Current frame number
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Update active events
        self._update_active_events(events)
        
        # Draw layers in order (back to front)
        if self.config.draw_zones:
            annotated = self._draw_zones(annotated, zones)
        
        if self.config.draw_tracks:
            annotated = self._draw_track_trails(annotated, detections)
        
        if self.config.draw_boxes:
            annotated = self._draw_detections(annotated, detections)
        
        if self.config.draw_events:
            annotated = self._draw_event_indicators(annotated, events)
        
        if self.config.draw_fps and fps is not None:
            annotated = self._draw_fps(annotated, fps)
        
        if frame_number is not None:
            annotated = self._draw_frame_number(annotated, frame_number)
        
        return annotated
    
    def _update_active_events(self, events: List[Dict]) -> None:
      """Update active events mapping for current frame."""
      # Clear old events and add new ones
      self.active_events.clear()
      
      for event in events:
          # Handle both Event objects and dictionaries
          if isinstance(event, dict):
              # Event is already a dictionary (from .to_dict())
              track_id = event.get('track_id')
              event_type = event.get('type', 'UNKNOWN')
          else:
              # Event is an Event object
              track_id = event.track_id
              # event_type is an enum, need to get .value
              event_type = event.event_type.value if hasattr(event, 'event_type') else 'UNKNOWN'
          
          if track_id is not None:
              self.active_events[track_id] = event_type
    
    def _draw_zones(
        self,
        frame: np.ndarray,
        zones: List[ZoneDefinition]
    ) -> np.ndarray:
        """Draw semi-transparent zone overlays."""
        overlay = frame.copy()
        
        for zone in zones:
            if not zone.enabled:
                continue
            
            points = np.array(zone.polygon, dtype=np.int32)
            
            # Choose color based on zone type
            if zone.type == 'intrusion':
                color = self.colors.ZONE_INTRUSION
            elif zone.type == 'loitering':
                color = self.colors.ZONE_LOITERING
            else:
                color = self.colors.ZONE_NORMAL
            
            # Fill with transparency
            cv2.fillPoly(overlay, [points], color)
            
            # Draw border
            cv2.polylines(frame, [points], True, color, 2)
            
            # Draw zone label
            centroid = np.mean(points, axis=0).astype(int)
            self._draw_label(
                frame,
                f"{zone.name} ({zone.type})",
                (centroid[0], centroid[1] - 10),
                color
            )
        
        # Blend overlay with original
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw bounding boxes and track IDs."""
        for det in detections:
            bbox = det['bbox']
            track_id = det.get('track_id', -1)
            confidence = det.get('confidence', 0.0)
            in_zone = det.get('in_zone', False)
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on state
            if track_id in self.active_events:
                color = self.colors.DETECTION_EVENT
            elif in_zone:
                color = self.colors.DETECTION_IN_ZONE
            else:
                color = self.colors.DETECTION_NORMAL
            
            # Draw bounding box
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                color, self.config.box_thickness
            )
            
            # Draw label with track ID and confidence
            if track_id >= 0:
                label = f"ID:{track_id} ({confidence:.2f})"
            else:
                label = f"({confidence:.2f})"
            
            self._draw_label(frame, label, (x1, y1 - 5), color)
            
            # Update track history
            if track_id >= 0:
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.track_history[track_id].append(centroid)
                
                # Limit history length
                if len(self.track_history[track_id]) > self.config.track_history_length:
                    self.track_history[track_id].pop(0)
        
        return frame
    
    def _draw_track_trails(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw motion trails for tracked objects."""
        active_ids = {d.get('track_id') for d in detections if d.get('track_id', -1) >= 0}
        
        for track_id, history in self.track_history.items():
            if track_id not in active_ids:
                continue
            
            if len(history) < 2:
                continue
            
            # Draw trail with fading effect
            for i in range(1, len(history)):
                alpha = i / len(history)
                thickness = max(1, int(2 * alpha))
                
                pt1 = history[i - 1]
                pt2 = history[i]
                
                # Create color with fading
                color = tuple(int(c * alpha) for c in self.colors.TRACK_TRAIL)
                
                cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame
    
    def _draw_event_indicators(
        self,
        frame: np.ndarray,
        events: List
    ) -> np.ndarray:
        """
        Draw event notification indicators.
        
        Args:
            frame: Input frame
            events: List of Event objects or event dictionaries
            
        Returns:
            Annotated frame
        """
        if not events:
            return frame
        
        # Draw event banner at top
        event_count = len(events)
        banner_height = 40
        cv2.rectangle(
            frame, (0, 0), (frame.shape[1], banner_height),
            self.colors.DETECTION_EVENT, -1
        )
        
        text = f"ALERT: {event_count} active event(s)"
        cv2.putText(
            frame, text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.colors.TEXT_FG,
            2
        )
        
        # Draw individual event labels near bboxes
        for event in events:
            try:
                # Extract bbox and event_type based on event format
                if isinstance(event, dict):
                    # Event is a dictionary
                    bbox = event.get('bbox')
                    event_type = event.get('type', 'EVENT')
                else:
                    # Event is an Event object
                    bbox = event.bbox if hasattr(event, 'bbox') else None
                    
                    # Handle EventType enum
                    if hasattr(event, 'event_type'):
                        if hasattr(event.event_type, 'value'):
                            event_type = event.event_type.value
                        else:
                            event_type = str(event.event_type)
                    else:
                        event_type = 'EVENT'
                
                # Draw label if bbox exists
                if bbox is not None and len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    
                    # Draw event type below bbox
                    self._draw_label(
                        frame,
                        str(event_type),
                        (x1, y2 + 20),
                        self.colors.DETECTION_EVENT
                    )
            
            except Exception as e:
                logger.warning(f"Error drawing event indicator: {e}")
                continue
        
        return frame
    
    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter in corner."""
        # Choose color based on FPS
        if fps >= 24:
            color = self.colors.FPS_GOOD
        elif fps >= 12:
            color = self.colors.FPS_OK
        else:
            color = self.colors.FPS_BAD
        
        text = f"FPS: {fps:.1f}"
        pos = (frame.shape[1] - 120, 30)
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (pos[0] - 5, pos[1] - 25),
            (pos[0] + 110, pos[1] + 5),
            self.colors.TEXT_BG,
            -1
        )
        
        cv2.putText(
            frame, text, pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )
        
        return frame
    
    def _draw_frame_number(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> np.ndarray:
        """Draw frame number in corner."""
        text = f"Frame: {frame_number}"
        pos = (10, frame.shape[0] - 10)
        
        cv2.putText(
            frame, text, pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, self.colors.TEXT_FG, 1
        )
        
        return frame
    
    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int]
    ) -> None:
        """Draw text label with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Draw background
        cv2.rectangle(
            frame,
            (x, y - text_height - 5),
            (x + text_width + 5, y + 5),
            self.colors.TEXT_BG,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, text,
            (x + 2, y),
            font, font_scale, color, thickness
        )
    
    def clear_track_history(self, track_ids: Optional[List[int]] = None) -> None:
        """
        Clear track history for specified IDs or all tracks.
        
        Args:
            track_ids: List of track IDs to clear, or None for all
        """
        if track_ids is None:
            self.track_history.clear()
        else:
            for tid in track_ids:
                self.track_history.pop(tid, None)