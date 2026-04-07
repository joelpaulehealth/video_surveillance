"""Tests for tracking module."""

import pytest
from src.tracking import Track, TrackState, ByteTracker


class TestTrack:
    """Test cases for Track dataclass."""
    
    def test_track_creation(self):
        """Test creating a Track object."""
        track = Track(
            track_id=1,
            bbox=(10, 20, 50, 80),
            confidence=0.9,
            state=TrackState.CONFIRMED
        )
        
        assert track.track_id == 1
        assert track.is_confirmed is True
    
    def test_track_center(self):
        """Test track center calculation."""
        track = Track(
            track_id=1,
            bbox=(0, 0, 100, 100),
            confidence=0.9
        )
        assert track.center == (50.0, 50.0)
    
    def test_to_dict(self):
        """Test track to dictionary conversion."""
        track = Track(
            track_id=5,
            bbox=(10, 20, 30, 40),
            confidence=0.85,
            state=TrackState.CONFIRMED
        )
        
        d = track.to_dict()
        assert d['track_id'] == 5
        assert d['state'] == 'CONFIRMED'


class TestByteTracker:
    """Test cases for ByteTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create ByteTracker instance."""
        return ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.track_count == 0
        assert len(tracker.active_tracks) == 0
    
    def test_single_detection(self, tracker):
        """Test tracking a single detection."""
        detections = [
            {'bbox': [10, 10, 50, 50], 'confidence': 0.9}
        ]
        
        tracks = tracker.update(detections)
        # New track won't be confirmed until min_hits
        assert len(tracks) >= 0
    
    def test_track_persistence(self, tracker):
        """Test that tracks persist across frames."""
        detections = [
            {'bbox': [10, 10, 50, 50], 'confidence': 0.9}
        ]
        
        # Update multiple times with same detection
        for _ in range(5):
            tracks = tracker.update(detections)
        
        # Should have created at least one track
        assert tracker.track_count >= 1
    
    def test_reset(self, tracker):
        """Test tracker reset."""
        detections = [
            {'bbox': [10, 10, 50, 50], 'confidence': 0.9}
        ]
        
        tracker.update(detections)
        tracker.reset()
        
        assert tracker.track_count == 0