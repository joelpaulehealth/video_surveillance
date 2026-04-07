"""Tests for detection module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.detection import Detection, YOLODetector


class TestDetection:
    """Test cases for Detection dataclass."""
    
    def test_detection_creation(self):
        """Test creating a Detection object."""
        det = Detection(
            bbox=(10.0, 20.0, 50.0, 80.0),
            confidence=0.95,
            class_id=0,
            class_name="person"
        )
        
        assert det.bbox == (10.0, 20.0, 50.0, 80.0)
        assert det.confidence == 0.95
        assert det.class_id == 0
    
    def test_detection_area(self):
        """Test bounding box area calculation."""
        det = Detection(
            bbox=(0.0, 0.0, 100.0, 50.0),
            confidence=0.9,
            class_id=0
        )
        assert det.area == 5000.0
    
    def test_detection_center(self):
        """Test center point calculation."""
        det = Detection(
            bbox=(0.0, 0.0, 100.0, 100.0),
            confidence=0.9,
            class_id=0
        )
        assert det.center == (50.0, 50.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        det = Detection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.85,
            class_id=0
        )
        
        d = det.to_dict()
        assert d['bbox'] == [10.0, 20.0, 30.0, 40.0]
        assert d['confidence'] == 0.85
        assert d['class_id'] == 0


class TestYOLODetector:
    """Test cases for YOLODetector."""
    
    @pytest.fixture
    def mock_yolo_model(self):
        """Create a mock YOLO model."""
        with patch('src.detection.yolo_detector.YOLO') as mock:
            # Setup mock return values
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_detector_initialization(self, mock_yolo_model):
        """Test detector initialization."""
        detector = YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5,
            device="cpu"
        )
        
        assert detector.model_name == "yolov8n.pt"
        assert detector.device == "cpu"
        assert detector.confidence_threshold == 0.5
    
    def test_warmup(self, mock_yolo_model):
        """Test model warmup."""
        detector = YOLODetector(model_path="yolov8n.pt", device="cpu")
        
        # Mock the detect method
        detector.detect = MagicMock(return_value=[])
        
        detector.warmup()
        
        # Should have called detect at least once
        assert detector.detect.call_count >= 3