"""
Zone intrusion event detector.

Design Choice:
- Detects when a person enters a restricted zone
- Debouncing to prevent duplicate events from brief exits/entries
- Tracks entry time for event metadata
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict

from .base_event_detector import BaseEventDetector, Event, EventType
from .zone_manager import ZoneManager
from ..utils.config import ZoneDefinition
from ..utils.logger import get_logger

logger = get_logger(__name__)


class IntrusionDetector(BaseEventDetector):
    """
    Detects zone intrusion events.
    
    Triggers when a tracked person enters a zone marked as 'intrusion' type.
    Includes debouncing to prevent duplicate events from brief boundary crossings.
    """
    
    def __init__(
        self,
        zone_manager: ZoneManager,
        debounce_frames: int = 5
    ):
        """
        Initialize intrusion detector.
        
        Args:
            zone_manager: ZoneManager instance for zone queries
            debounce_frames: Minimum frames in zone before triggering event
        """
        self._zone_manager = zone_manager
        self._debounce_frames = debounce_frames
        
        # Track state: {(track_id, zone_id): frames_in_zone}
        self._entry_frames: Dict[tuple, int] = defaultdict(int)
        
        # Already triggered events: {(track_id, zone_id)}
        self._triggered: Set[tuple] = set()
        
        # Intrusion zones
        self._intrusion_zones = zone_manager.get_zones_by_type('intrusion')
        
        logger.info(
            f"IntrusionDetector initialized: "
            f"{len(self._intrusion_zones)} intrusion zones, "
            f"debounce={debounce_frames} frames"
        )
    
    def process_frame(
        self,
        tracks: List[dict],
        frame_number: int,
        timestamp: float,
        zones_info: Dict[int, List[str]]
    ) -> List[Event]:
        """
        Process frame and detect intrusion events.
        
        Args:
            tracks: List of track dictionaries
            frame_number: Current frame number
            timestamp: Current video timestamp
            zones_info: Mapping of track_id to zones they're in
            
        Returns:
            List of intrusion events detected this frame
        """
        events = []
        
        # Get intrusion zone IDs
        intrusion_zone_ids = {z.id for z in self._intrusion_zones}
        
        # Track which (track_id, zone_id) pairs are active this frame
        active_pairs = set()
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            confidence = track['confidence']
            
            # Get zones this track is in
            track_zones = zones_info.get(track_id, [])
            
            for zone_id in track_zones:
                if zone_id not in intrusion_zone_ids:
                    continue
                
                key = (track_id, zone_id)
                active_pairs.add(key)
                
                # Increment frame counter
                self._entry_frames[key] += 1
                
                # Check if we should trigger event
                if key not in self._triggered:
                    if self._entry_frames[key] >= self._debounce_frames:
                        # Trigger intrusion event
                        zone = self._zone_manager.get_zone(zone_id)
                        
                        event = Event(
                            event_id=Event.generate_id(),
                            event_type=EventType.INTRUSION,
                            track_id=track_id,
                            zone_id=zone_id,
                            zone_name=zone.name if zone else zone_id,
                            frame_number=frame_number,
                            timestamp_seconds=timestamp,
                            bbox=tuple(bbox),
                            confidence=confidence,
                            metadata={
                                'entry_frame': frame_number - self._entry_frames[key] + 1
                            }
                        )
                        events.append(event)
                        self._triggered.add(key)
                        
                        logger.debug(
                            f"Intrusion detected: track {track_id} in zone {zone_id}"
                        )
        
        # Reset counters for tracks that left zones
        keys_to_remove = []
        for key in self._entry_frames:
            if key not in active_pairs:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._entry_frames[key]
            self._triggered.discard(key)
        
        return events
    
    def reset(self) -> None:
        """Reset detector state."""
        self._entry_frames.clear()
        self._triggered.clear()
        logger.debug("IntrusionDetector reset")
    
    @property
    def event_type(self) -> EventType:
        """Get event type."""
        return EventType.INTRUSION