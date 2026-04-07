"""
Base class for event detectors.

Design Choice:
- Abstract interface for different event types
- Event dataclass provides consistent output format
- State management for temporal reasoning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import uuid


class EventType(Enum):
    """Types of detectable events."""
    INTRUSION = "INTRUSION"
    LOITERING = "LOITERING"
    ZONE_EXIT = "ZONE_EXIT"
    CROWD_FORMATION = "CROWD_FORMATION"


@dataclass
class Event:
    """
    Represents a detected event.
    
    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        track_id: Associated track ID
        zone_id: Zone where event occurred
        zone_name: Human-readable zone name
        frame_number: Frame when event started/detected
        timestamp_seconds: Video timestamp
        bbox: Bounding box at event time
        confidence: Detection confidence
        duration_seconds: For temporal events (e.g., loitering)
        metadata: Additional event-specific data
    """
    event_id: str
    event_type: EventType
    track_id: int
    zone_id: str
    zone_name: str
    frame_number: int
    timestamp_seconds: float
    bbox: Tuple[float, float, float, float]
    confidence: float
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'type': self.event_type.value,
            'track_id': self.track_id,
            'zone_id': self.zone_id,
            'zone_name': self.zone_name,
            'frame_number': self.frame_number,
            'timestamp_seconds': round(self.timestamp_seconds, 3),
            'bbox': [round(x, 2) for x in self.bbox],
            'confidence': round(self.confidence, 3),
            'duration_seconds': round(self.duration_seconds, 2) if self.duration_seconds else None,
            'metadata': self.metadata
        }
    
    @staticmethod
    def generate_id() -> str:
        """Generate unique event ID."""
        return f"evt_{uuid.uuid4().hex[:8]}"


class BaseEventDetector(ABC):
    """
    Abstract base class for event detectors.
    
    All event detector implementations must inherit from this class
    and implement the process_frame method.
    """
    
    @abstractmethod
    def process_frame(
        self,
        tracks: List[dict],
        frame_number: int,
        timestamp: float,
        zones_info: Dict[int, List[str]]
    ) -> List[Event]:
        """
        Process a frame and detect events.
        
        Args:
            tracks: List of track dictionaries with bbox, track_id, confidence
            frame_number: Current frame number
            timestamp: Current video timestamp in seconds
            zones_info: Mapping of track_id to list of zone_ids they're in
            
        Returns:
            List of detected events
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state for new video."""
        pass
    
    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Get the type of events this detector produces."""
        pass