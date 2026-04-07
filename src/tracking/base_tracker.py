"""
Abstract base class for multi-object trackers.

Design Choice:
- Abstract interface allows swapping tracking algorithms
- Track dataclass maintains state across frames
- Support for track lifecycle management (confirmed, lost, deleted)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = 1   # New track, not yet confirmed
    CONFIRMED = 2   # Confirmed track with stable ID
    LOST = 3        # Temporarily lost, may recover
    DELETED = 4     # Permanently deleted


@dataclass
class Track:
    """
    Represents a tracked object across frames.
    
    Attributes:
        track_id: Unique identifier for this track
        bbox: Current bounding box (x1, y1, x2, y2)
        confidence: Latest detection confidence
        state: Current track state
        age: Total frames since track creation
        hits: Total successful detections
        time_since_update: Frames since last detection
        history: List of past bounding boxes
    """
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    state: TrackState = TrackState.TENTATIVE
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert track to dictionary."""
        return {
            'track_id': self.track_id,
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'state': self.state.name,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update
        }
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of current bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.CONFIRMED


class BaseTracker(ABC):
    """
    Abstract base class for multi-object trackers.
    
    All tracker implementations must inherit from this class
    and implement the update method.
    """
    
    @abstractmethod
    def update(
        self,
        detections: List[dict],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'
            frame: Optional frame for appearance-based tracking
            
        Returns:
            List of active Track objects
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state for new video."""
        pass
    
    @property
    @abstractmethod
    def active_tracks(self) -> List[Track]:
        """Get currently active tracks."""
        pass
    
    @property
    @abstractmethod
    def track_count(self) -> int:
        """Get total number of tracks created."""
        pass