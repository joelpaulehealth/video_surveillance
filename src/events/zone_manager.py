"""
Zone management for spatial event detection.

Design Choice:
- Zones defined in JSON for easy editing without code changes
- Support for multiple zone types (intrusion, loitering)
- Efficient point-in-polygon testing with Shapely
- Per-zone configuration for thresholds
"""

from typing import List, Dict, Optional, Set, Tuple
from shapely.geometry import Polygon

from ..utils.config import ZoneDefinition
from ..utils.geometry import GeometryUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ZoneManager:
    """
    Manages zone definitions and provides zone intersection testing.
    
    Maintains zone state and provides efficient spatial queries
    for detecting when tracks enter or exit zones.
    """
    
    def __init__(self, zones: List[ZoneDefinition]):
        """
        Initialize zone manager with zone definitions.
        
        Args:
            zones: List of ZoneDefinition objects
        """
        self._zones = {z.id: z for z in zones if z.enabled}
        self._geometry = GeometryUtils()
        
        # Pre-create polygon objects for efficiency
        self._polygons: Dict[str, Polygon] = {}
        for zone_id, zone in self._zones.items():
            self._polygons[zone_id] = self._geometry.create_polygon(
                zone.polygon, zone_id
            )
        
        # Track zone occupancy: {zone_id: set(track_ids)}
        self._zone_occupancy: Dict[str, Set[int]] = {
            zone_id: set() for zone_id in self._zones
        }
        
        logger.info(f"ZoneManager initialized with {len(self._zones)} zones")
    
    def check_zones(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        use_centroid: bool = True
    ) -> List[str]:
        """
        Check which zones a track is currently in.
        
        Args:
            track_id: Track identifier
            bbox: Bounding box (x1, y1, x2, y2)
            use_centroid: If True, use bbox centroid; else use bottom center
            
        Returns:
            List of zone IDs the track is currently in
        """
        # Get test point
        if use_centroid:
            point = self._geometry.bbox_centroid(bbox)
        else:
            point = self._geometry.bbox_bottom_center(bbox)
        
        zones_in = []
        
        for zone_id, polygon in self._polygons.items():
            if self._geometry.point_in_polygon(point, polygon):
                zones_in.append(zone_id)
        
        return zones_in
    
    def update_occupancy(
        self,
        track_id: int,
        current_zones: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Update zone occupancy and return enter/exit events.
        
        Args:
            track_id: Track identifier
            current_zones: List of zones currently occupied
            
        Returns:
            Tuple of (entered_zones, exited_zones)
        """
        current_set = set(current_zones)
        entered = []
        exited = []
        
        for zone_id in self._zones:
            was_in = track_id in self._zone_occupancy[zone_id]
            is_in = zone_id in current_set
            
            if is_in and not was_in:
                # Just entered zone
                self._zone_occupancy[zone_id].add(track_id)
                entered.append(zone_id)
            elif not is_in and was_in:
                # Just exited zone
                self._zone_occupancy[zone_id].discard(track_id)
                exited.append(zone_id)
        
        return entered, exited
    
    def get_zone(self, zone_id: str) -> Optional[ZoneDefinition]:
        """Get zone definition by ID."""
        return self._zones.get(zone_id)
    
    def get_polygon(self, zone_id: str) -> Optional[Polygon]:
        """Get zone polygon by ID."""
        return self._polygons.get(zone_id)
    
    def get_zones_by_type(self, zone_type: str) -> List[ZoneDefinition]:
        """Get all zones of a specific type."""
        return [z for z in self._zones.values() if z.type == zone_type]
    
    def get_occupants(self, zone_id: str) -> Set[int]:
        """Get set of track IDs currently in a zone."""
        return self._zone_occupancy.get(zone_id, set()).copy()
    
    def remove_track(self, track_id: int) -> None:
        """Remove a track from all zone occupancy records."""
        for zone_id in self._zones:
            self._zone_occupancy[zone_id].discard(track_id)
    
    def clear_occupancy(self) -> None:
        """Clear all occupancy records."""
        for zone_id in self._zones:
            self._zone_occupancy[zone_id].clear()
    
    @property
    def all_zones(self) -> List[ZoneDefinition]:
        """Get all zone definitions."""
        return list(self._zones.values())
    
    @property
    def zone_count(self) -> int:
        """Get number of active zones."""
        return len(self._zones)