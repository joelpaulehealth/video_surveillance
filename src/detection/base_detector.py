"""
Abstract base class for person detectors.

Design Choice:
- Abstract interface allows swapping detection models easily
- Detection dataclass provides consistent output format
- Supports batch processing for GPU efficiency
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Detection:
    """
    Represents a single detection result.
    
    Attributes:
        bbox: Bounding box (x1, y1, x2, y2)
        confidence: Detection confidence score (0-1)
        class_id: Class ID (0 for person in COCO)
        class_name: Human-readable class name
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = "person"
    
    def to_dict(self) -> dict:
        """Convert detection to dictionary."""
        return {
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name
        }
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.
    
    All detector implementations must inherit from this class
    and implement the detect method.
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame (BGR, numpy array)
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect objects in a batch of frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists, one per frame
        """
        pass
    
    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the model with a dummy inference.
        
        Useful for accurate FPS benchmarking.
        """
        pass
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Get the device (cuda/cpu) the model is running on."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier."""
        pass