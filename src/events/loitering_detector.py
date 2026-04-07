"""
Loitering event detector.

Design Choice:
- Detects when a person remains stationary in a zone beyond a threshold
- Uses movement tracking to determine if person is stationary
- Per-zone configurable thresholds
- Continuous events while loitering persists
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import numpy as np

from .base_event_detector import BaseEventDetector, Event, EventType
from .zone_manager import ZoneManager
from ..utils.geometry import GeometryUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LoiteringDetector(BaseEventDetector):
    """
    Detects loitering events in designated zones.
    
    Triggers when a tracked person:
    1. Remains in a loitering zone
    2. Has minimal movement (below threshold)
    3. Exceeds the time threshold
    """
    
    def __init__(
        self,
        zone_manager: ZoneManager,
        fps: float = 30.0,
        default_time_threshold: float = 10.0,
        default_movement_threshold: float = 25.0,
        event_interval_seconds: float = 5.0
    ):
        """
        Initialize loitering detector.
        
        Args:
            zone_manager: ZoneManager instance
            fps: Video FPS for time calculations
            default_time_threshold: Default loitering time in seconds
            default_movement_threshold: Max pixels to be considered stationary
            event_interval_seconds: Minimum interval between repeated events
        """
        self._zone_manager = zone_manager
        self._fps = fps
        self._default_time_threshold = default_time_threshold
        self._default_movement_threshold = default_movement_threshold
        self._event_interval_seconds = event_interval_seconds
        
        self._geometry = GeometryUtils()
        
        # Track state
        # {track_id: {zone_id: {'entry_frame', 'positions', 'last_event_time'}}}
        self._track_state: Dict[int, Dict[str, dict]] = defaultdict(dict)
        
        # Loitering zones
        self._loitering_zones = zone_manager.get_zones_by_type('loitering')
        
        logger.info(
            f"LoiteringDetector initialized: "
            f"{len(self._loitering_zones)} loitering zones, "
            f"default_threshold={default_time_threshold}s"
        )
    
    def process_frame(
        self,
        tracks: List[dict],
        frame_number: int,
        timestamp: float,
        zones_info: Dict[int, List[str]]
    ) -> List[Event]:
        """
        Process frame and detect loitering events.
        
        Args:
            tracks: List of track dictionaries
            frame_number: Current frame number
            timestamp: Current video timestamp
            zones_info: Mapping of track_id to zones they're in
            
        Returns:
            List of loitering events detected this frame
        """
        events = []
        
        loitering_zone_ids = {z.id for z in self._loitering_zones}
        active_track_zones = set()
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            confidence = track['confidence']
            
            centroid = self._geometry.bbox_centroid(tuple(bbox))
            track_zones = zones_info.get(track_id, [])
            
            for zone_id in track_zones:
                if zone_id not in loitering_zone_ids:
                    continue
                
                active_track_zones.add((track_id, zone_id))
                
                # Initialize or update track state for this zone
                if zone_id not in self._track_state[track_id]:
                    self._track_state[track_id][zone_id] = {
                        'entry_frame': frame_number,
                        'positions': [centroid],
                        'last_event_time': -float('inf')
                    }
                else:
                    self._track_state[track_id][zone_id]['positions'].append(centroid)
                    # Keep only recent positions (last 30 frames)
                    positions = self._track_state[track_id][zone_id]['positions']
                    if len(positions) > 30:
                        self._track_state[track_id][zone_id]['positions'] = positions[-30:]
                
                state = self._track_state[track_id][zone_id]
                zone = self._zone_manager.get_zone(zone_id)
                
                # Get zone-specific thresholds
                time_threshold = zone.loitering_threshold_seconds if zone else self._default_time_threshold
                movement_threshold = zone.movement_threshold_pixels if zone else self._default_movement_threshold
                
                # Calculate time in zone
                time_in_zone = (frame_number - state['entry_frame']) / self._fps
                
                # Check if stationary
                if self._is_stationary(state['positions'], movement_threshold):
                    # Check if enough time has passed
                    if time_in_zone >= time_threshold:
                        # Check event interval
                        time_since_last = timestamp - state['last_event_time']
                        
                        if time_since_last >= self._event_interval_seconds:
                            event = Event(
                                event_id=Event.generate_id(),
                                event_type=EventType.LOITERING,
                                track_id=track_id,
                                zone_id=zone_id,
                                zone_name=zone.name if zone else zone_id,
                                frame_number=frame_number,
                                timestamp_seconds=timestamp,
                                bbox=tuple(bbox),
                                confidence=confidence,
                                duration_seconds=time_in_zone,
                                metadata={
                                    'entry_frame': state['entry_frame'],
                                    'movement_pixels': self._calculate_movement(state['positions'])
                                }
                            )
                            events.append(event)
                            state['last_event_time'] = timestamp
                            
                            logger.debug(
                                f"Loitering detected: track {track_id} in zone {zone_id}, "
                                f"duration={time_in_zone:.1f}s"
                            )
        
        # Clean up tracks that left zones
        tracks_to_clean = []
        for track_id in self._track_state:
            zones_to_remove = []
            for zone_id in self._track_state[track_id]:
                if (track_id, zone_id) not in active_track_zones:
                    zones_to_remove.append(zone_id)
            
            for zone_id in zones_to_remove:
                del self._track_state[track_id][zone_id]
            
            if not self._track_state[track_id]:
                tracks_to_clean.append(track_id)
        
        for track_id in tracks_to_clean:
            del self._track_state[track_id]
        
        return events
    
    def _is_stationary(
        self,
        positions: List[Tuple[float, float]],
        threshold: float
    ) -> bool:
        """
        Check if positions indicate stationary behavior.
        
        Uses the bounding box of all positions to determine movement.
        """
        if len(positions) < 2:
            return False
        
        movement = self._calculate_movement(positions)
        return movement < threshold
    
    def _calculate_movement(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate total bounding box span of positions."""
        if len(positions) < 2:
            return 0.0
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        # Use max span as movement indicator
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        
        return max(x_span, y_span)
    
    def reset(self) -> None:
        """Reset detector state."""
        self._track_state.clear()
        logger.debug("LoiteringDetector reset")
    
    def set_fps(self, fps: float) -> None:
        """Update FPS for time calculations."""
        self._fps = fps
        logger.debug(f"LoiteringDetector FPS updated to {fps}")
    
    @property
    def event_type(self) -> EventType:
        """Get event type."""
        return EventType.LOITERING