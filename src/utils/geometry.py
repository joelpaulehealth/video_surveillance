"""
Geometry utilities for zone intersection and spatial calculations.

Design Choice:
- Using Shapely for robust polygon operations (handles edge cases)
- Caching polygon objects for performance
- Supporting both centroid and bbox intersection modes
"""

from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep

from .logger import get_logger

logger = get_logger(__name__)


class GeometryUtils:
    """
    Utility class for geometric operations.
    
    Provides efficient point-in-polygon testing and distance calculations
    with caching for repeated polygon operations.
    """
    
    def __init__(self):
        """Initialize geometry utilities with polygon cache."""
        self._polygon_cache: dict = {}
    
    def create_polygon(
        self,
        points: List[List[int]],
        zone_id: Optional[str] = None
    ) -> Polygon:
        """
        Create a Shapely polygon from coordinate list.
        
        Args:
            points: List of [x, y] coordinates
            zone_id: Optional ID for caching
            
        Returns:
            Shapely Polygon object
        """
        if zone_id and zone_id in self._polygon_cache:
            return self._polygon_cache[zone_id]
        
        polygon = Polygon(points)
        
        if not polygon.is_valid:
            logger.warning(f"Invalid polygon for zone {zone_id}, attempting fix")
            polygon = polygon.buffer(0)
        
        if zone_id:
            self._polygon_cache[zone_id] = polygon
        
        return polygon
    
    def point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: Polygon
    ) -> bool:
        """
        Check if a point is inside a polygon.
        
        Uses prepared polygon for faster repeated checks.
        
        Args:
            point: (x, y) tuple
            polygon: Shapely Polygon
            
        Returns:
            True if point is inside polygon
        """
        p = Point(point)
        prepared = prep(polygon)
        return prepared.contains(p)
    
    def bbox_centroid(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Calculate centroid of bounding box.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            (cx, cy) centroid coordinates
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def bbox_bottom_center(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Calculate bottom-center point of bounding box.
        
        Useful for ground-plane intersection with zones.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            (x, y) bottom center coordinates
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)
    
    def bbox_intersects_polygon(
        self,
        bbox: Tuple[float, float, float, float],
        polygon: Polygon,
        min_overlap: float = 0.0
    ) -> bool:
        """
        Check if bounding box intersects with polygon.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            polygon: Shapely Polygon
            min_overlap: Minimum intersection area ratio (0-1)
            
        Returns:
            True if intersection exists (and meets overlap threshold)
        """
        x1, y1, x2, y2 = bbox
        bbox_poly = box(x1, y1, x2, y2)
        
        if min_overlap <= 0:
            return bbox_poly.intersects(polygon)
        
        intersection = bbox_poly.intersection(polygon)
        overlap_ratio = intersection.area / bbox_poly.area
        return overlap_ratio >= min_overlap
    
    def euclidean_distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            Distance in pixels
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First box (x1, y1, x2, y2)
            bbox2: Second box (x1, y1, x2, y2)
            
        Returns:
            IoU value (0-1)
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_polygon_bounds(self, polygon: Polygon) -> Tuple[float, float, float, float]:
        """
        Get bounding box of a polygon.
        
        Args:
            polygon: Shapely Polygon
            
        Returns:
            (minx, miny, maxx, maxy) bounds
        """
        return polygon.bounds
    
    def clear_cache(self) -> None:
        """Clear the polygon cache."""
        self._polygon_cache.clear()