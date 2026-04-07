"""
Central event management and coordination.

Design Choice:
- Single point of coordination for all event detectors
- Deduplication logic to prevent redundant events
- Event aggregation and filtering
"""

from typing import List, Dict, Optional, Set
from collections import defaultdict

from .zone_manager import ZoneManager
from .base_event_detector import Event, EventType
from .intrusion_detector import IntrusionDetector
from .loitering_detector import LoiteringDetector
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EventManager:
    """
    Manages all event detection and coordination.
    
    Coordinates multiple event detectors, handles deduplication,
    and provides a unified interface for event processing.
    """
    
    def __init__(
        self,
        config: Config,
        fps: float = 30.0
    ):
        """
        Initialize event manager with configuration.
        
        Args:
            config: System configuration
            fps: Video FPS for temporal calculations
        """
        self._config = config
        self._fps = fps
        
        # Initialize zone manager
        self._zone_manager = ZoneManager(config.zones)
        
        # Initialize event detectors
        self._intrusion_detector = IntrusionDetector(
            self._zone_manager,
            debounce_frames=int(fps * 0.2)  # 0.2 second debounce
        )
        
        self._loitering_detector = LoiteringDetector(
            self._zone_manager,
            fps=fps,
            default_time_threshold=config.events.default_loitering_threshold,
            default_movement_threshold=config.events.default_movement_threshold,
            event_interval_seconds=config.events.deduplicate_window_seconds
        )
        
        # All events collected
        self._all_events: List[Event] = []
        
        # Deduplication
        self._dedup_window = config.events.deduplicate_window_seconds
        self._recent_events: Dict[str, float] = {}  # {event_key: timestamp}
        
        logger.info("EventManager initialized")
    
    def process_frame(
        self,
        tracks: List[dict],
        frame_number: int,
        timestamp: float
    ) -> List[Event]:
        """
        Process a frame through all event detectors.
        
        Args:
            tracks: List of track dictionaries
            frame_number: Current frame number
            timestamp: Current video timestamp
            
        Returns:
            List of events detected in this frame
        """
        # First, check zone occupancy for all tracks
        zones_info: Dict[int, List[str]] = {}
        
        for track in tracks:
            track_id = track['track_id']
            bbox = tuple(track['bbox'])
            
            # Check which zones this track is in
            zones = self._zone_manager.check_zones(track_id, bbox)
            zones_info[track_id] = zones
            
            # Update zone manager's occupancy tracking
            self._zone_manager.update_occupancy(track_id, zones)
            
            # Mark track as in_zone for visualization
            track['in_zone'] = len(zones) > 0
        
        # Collect events from all detectors
        frame_events = []
        
        # Intrusion detection
        intrusion_events = self._intrusion_detector.process_frame(
            tracks, frame_number, timestamp, zones_info
        )
        frame_events.extend(intrusion_events)
        
        # Loitering detection
        loitering_events = self._loitering_detector.process_frame(
            tracks, frame_number, timestamp, zones_info
        )
        frame_events.extend(loitering_events)
        
        # Deduplicate events
        deduplicated = self._deduplicate_events(frame_events, timestamp)
        
        # Store all events
        self._all_events.extend(deduplicated)
        
        return deduplicated
    
    def _deduplicate_events(
        self,
        events: List[Event],
        current_time: float
    ) -> List[Event]:
        """
        Remove duplicate events within the deduplication window.
        
        Args:
            events: List of events to deduplicate
            current_time: Current video timestamp
            
        Returns:
            Deduplicated event list
        """
        # Clean old entries
        self._recent_events = {
            k: v for k, v in self._recent_events.items()
            if current_time - v < self._dedup_window
        }
        
        deduplicated = []
        for event in events:
            # Create unique key for event
            key = f"{event.event_type.value}_{event.track_id}_{event.zone_id}"
            
            if key not in self._recent_events:
                deduplicated.append(event)
                self._recent_events[key] = current_time
        
        return deduplicated
    
    def get_all_events(self) -> List[Event]:
        """Get all events collected so far."""
        return self._all_events.copy()
    
    def get_events_summary(self) -> Dict:
        """Get summary statistics of all events."""
        summary = {
            'total_events': len(self._all_events),
            'by_type': defaultdict(int),
            'by_zone': defaultdict(int),
            'unique_tracks': set()
        }
        
        for event in self._all_events:
            summary['by_type'][event.event_type.value] += 1
            summary['by_zone'][event.zone_id] += 1
            summary['unique_tracks'].add(event.track_id)
        
        summary['unique_tracks'] = len(summary['unique_tracks'])
        summary['by_type'] = dict(summary['by_type'])
        summary['by_zone'] = dict(summary['by_zone'])
        
        return summary
    
    def reset(self) -> None:
        """Reset all event detectors and state."""
        self._intrusion_detector.reset()
        self._loitering_detector.reset()
        self._zone_manager.clear_occupancy()
        self._all_events.clear()
        self._recent_events.clear()
        logger.debug("EventManager reset")
    
    def set_fps(self, fps: float) -> None:
        """Update FPS for temporal calculations."""
        self._fps = fps
        self._loitering_detector.set_fps(fps)
    
    @property
    def zone_manager(self) -> ZoneManager:
        """Get the zone manager instance."""
        return self._zone_manager