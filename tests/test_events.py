"""Tests for event detection."""

import pytest
from unittest.mock import MagicMock, patch

from src.utils.config import ZoneDefinition
from src.events.zone_manager import ZoneManager
from src.events.intrusion_detector import IntrusionDetector
from src.events.loitering_detector import LoiteringDetector
from src.events.base_event_detector import EventType


class TestZoneManager:
    """Test cases for ZoneManager."""
    
    @pytest.fixture
    def zones(self):
        """Create test zone definitions."""
        return [
            ZoneDefinition(
                id="zone1",
                name="Test Zone 1",
                type="intrusion",
                polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
                enabled=True
            ),
            ZoneDefinition(
                id="zone2",
                name="Test Zone 2",
                type="loitering",
                polygon=[[200, 200], [300, 200], [300, 300], [200, 300]],
                enabled=True,
                loitering_threshold_seconds=5.0
            ),
            ZoneDefinition(
                id="zone3",
                name="Disabled Zone",
                type="intrusion",
                polygon=[[400, 400], [500, 400], [500, 500], [400, 500]],
                enabled=False
            )
        ]
    
    @pytest.fixture
    def zone_manager(self, zones):
        """Create ZoneManager instance."""
        return ZoneManager(zones)
    
    def test_zone_count(self, zone_manager):
        """Test that only enabled zones are counted."""
        assert zone_manager.zone_count == 2
    
    def test_check_zones_inside(self, zone_manager):
        """Test checking zones with point inside."""
        # Point in zone1
        bbox = (40, 40, 60, 60)  # Centroid at (50, 50)
        zones = zone_manager.check_zones(1, bbox)
        assert "zone1" in zones
    
    def test_check_zones_outside(self, zone_manager):
        """Test checking zones with point outside all zones."""
        bbox = (500, 500, 520, 520)  # Centroid at (510, 510)
        zones = zone_manager.check_zones(1, bbox)
        assert len(zones) == 0
    
    def test_update_occupancy_enter(self, zone_manager):
        """Test occupancy update when entering zone."""
        entered, exited = zone_manager.update_occupancy(1, ["zone1"])
        assert "zone1" in entered
        assert len(exited) == 0
    
    def test_update_occupancy_exit(self, zone_manager):
        """Test occupancy update when exiting zone."""
        # First enter
        zone_manager.update_occupancy(1, ["zone1"])
        # Then exit
        entered, exited = zone_manager.update_occupancy(1, [])
        assert len(entered) == 0
        assert "zone1" in exited
    
    def test_get_zones_by_type(self, zone_manager):
        """Test filtering zones by type."""
        intrusion_zones = zone_manager.get_zones_by_type("intrusion")
        loitering_zones = zone_manager.get_zones_by_type("loitering")
        
        assert len(intrusion_zones) == 1
        assert len(loitering_zones) == 1


class TestIntrusionDetector:
    """Test cases for IntrusionDetector."""
    
    @pytest.fixture
    def zone_manager(self):
        """Create ZoneManager with intrusion zone."""
        zones = [
            ZoneDefinition(
                id="intrusion_zone",
                name="Test Intrusion Zone",
                type="intrusion",
                polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
                enabled=True
            )
        ]
        return ZoneManager(zones)
    
    @pytest.fixture
    def detector(self, zone_manager):
        """Create IntrusionDetector instance."""
        return IntrusionDetector(zone_manager, debounce_frames=2)
    
    def test_intrusion_detection(self, detector):
        """Test intrusion event is detected."""
        tracks = [
            {'track_id': 1, 'bbox': [40, 40, 60, 60], 'confidence': 0.9}
        ]
        zones_info = {1: ['intrusion_zone']}
        
        # Frame 1 - enters zone
        events = detector.process_frame(tracks, 1, 0.033, zones_info)
        assert len(events) == 0  # Not triggered yet (debounce)
        
        # Frame 2 - still in zone
        events = detector.process_frame(tracks, 2, 0.066, zones_info)
        assert len(events) == 1  # Triggered after debounce
        assert events[0].event_type == EventType.INTRUSION
    
    def test_no_double_trigger(self, detector):
        """Test that same intrusion isn't triggered twice."""
        tracks = [
            {'track_id': 1, 'bbox': [40, 40, 60, 60], 'confidence': 0.9}
        ]
        zones_info = {1: ['intrusion_zone']}
        
        # Trigger first event
        detector.process_frame(tracks, 1, 0.033, zones_info)
        events = detector.process_frame(tracks, 2, 0.066, zones_info)
        assert len(events) == 1
        
        # Should not trigger again
        events = detector.process_frame(tracks, 3, 0.1, zones_info)
        assert len(events) == 0


# class TestLoiteringDetector:
#     """Test cases for LoiteringDetector."""
    
#     @pytest.fixture
#     def zone_manager(self):
#         """Create ZoneManager with loitering zone."""
#         zones = [
#             ZoneDefinition(
#                 id="loiter_zone",
#                 name="Test Loiter Zone",
#                 type="loitering",
#                 polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
#                 enabled=True,