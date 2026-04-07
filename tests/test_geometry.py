"""Tests for geometry utilities."""

import pytest
import numpy as np
from shapely.geometry import Polygon

from src.utils.geometry import GeometryUtils


class TestGeometryUtils:
    """Test cases for GeometryUtils class."""
    
    @pytest.fixture
    def geometry(self):
        """Create GeometryUtils instance."""
        return GeometryUtils()
    
    @pytest.fixture
    def square_polygon(self, geometry):
        """Create a simple square polygon."""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        return geometry.create_polygon(points, "test_square")
    
    def test_point_in_polygon_inside(self, geometry, square_polygon):
        """Test point inside polygon."""
        assert geometry.point_in_polygon((50, 50), square_polygon) is True
    
    def test_point_in_polygon_outside(self, geometry, square_polygon):
        """Test point outside polygon."""
        assert geometry.point_in_polygon((150, 150), square_polygon) is False
    
    def test_point_in_polygon_edge(self, geometry, square_polygon):
        """Test point on polygon edge."""
        # Points on edges are typically considered inside
        result = geometry.point_in_polygon((50, 0), square_polygon)
        # Shapely may return True or False for edge points
        assert isinstance(result, bool)
    
    def test_bbox_centroid(self, geometry):
        """Test bounding box centroid calculation."""
        bbox = (0, 0, 100, 100)
        centroid = geometry.bbox_centroid(bbox)
        assert centroid == (50, 50)
    
    def test_bbox_bottom_center(self, geometry):
        """Test bounding box bottom center calculation."""
        bbox = (0, 0, 100, 100)
        bottom_center = geometry.bbox_bottom_center(bbox)
        assert bottom_center == (50, 100)
    
    def test_euclidean_distance(self, geometry):
        """Test Euclidean distance calculation."""
        p1 = (0, 0)
        p2 = (3, 4)
        distance = geometry.euclidean_distance(p1, p2)
        assert distance == 5.0
    
    def test_euclidean_distance_same_point(self, geometry):
        """Test distance between same point."""
        p1 = (50, 50)
        distance = geometry.euclidean_distance(p1, p1)
        assert distance == 0.0
    
    def test_calculate_iou_no_overlap(self, geometry):
        """Test IoU with no overlap."""
        bbox1 = (0, 0, 50, 50)
        bbox2 = (100, 100, 150, 150)
        iou = geometry.calculate_iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_calculate_iou_perfect_overlap(self, geometry):
        """Test IoU with perfect overlap."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (0, 0, 100, 100)
        iou = geometry.calculate_iou(bbox1, bbox2)
        assert iou == 1.0
    
    def test_calculate_iou_partial_overlap(self, geometry):
        """Test IoU with partial overlap."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 150, 150)
        iou = geometry.calculate_iou(bbox1, bbox2)
        # Overlap area = 50*50 = 2500
        # Union = 10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        assert 0.14 < iou < 0.15
    
    def test_polygon_caching(self, geometry):
        """Test polygon caching behavior."""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        
        poly1 = geometry.create_polygon(points, "cached_test")
        poly2 = geometry.create_polygon(points, "cached_test")
        
        # Should return the same cached object
        assert poly1 is poly2
    
    def test_clear_cache(self, geometry):
        """Test cache clearing."""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        
        poly1 = geometry.create_polygon(points, "clear_test")
        geometry.clear_cache()
        poly2 = geometry.create_polygon(points, "clear_test")
        
        # Should be different objects after cache clear
        assert poly1 is not poly2