"""
ByteTrack implementation for multi-object tracking.

Design Choice:
- ByteTrack over DeepSORT: Better handling of low-confidence detections
- No separate ReID model needed (lighter, faster)
- SOTA performance on MOT benchmarks
- Handles occlusions well with two-stage association

Trade-offs:
- Less robust for long-term re-identification across cameras
- May struggle with very similar-looking targets
- Works best with consistent detection quality

Implementation approach:
- Using Kalman filter for motion prediction
- Hungarian algorithm for detection-track association
- Two-stage matching: high confidence first, then low confidence
"""

from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from .base_tracker import BaseTracker, Track, TrackState
from ..utils.logger import get_logger
from ..utils.geometry import GeometryUtils

logger = get_logger(__name__)


class KalmanBoxTracker:
    """
    Kalman filter for tracking a single bounding box.
    
    State: [x, y, s, r, vx, vy, vs]
    where (x, y) is center, s is scale (area), r is aspect ratio
    """
    
    count = 0
    
    def __init__(self, bbox: Tuple[float, float, float, float]):
        """
        Initialize tracker with initial bounding box.
        
        Args:
            bbox: Initial bounding box (x1, y1, x2, y2)
        """
        # Initialize Kalman filter
        # State: [x, y, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.
        
        # Covariance matrix
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        # Track metadata
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history = []
        self.confidence = 1.0
    
    def _bbox_to_z(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert bbox to measurement vector [x, y, s, r]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        s = w * h  # scale (area)
        r = w / (h + 1e-6)  # aspect ratio
        return np.array([[x], [y], [s], [r]])
    
    def _x_to_bbox(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert state vector to bbox."""
        cx, cy, s, r = x[:4].flatten()
        w = np.sqrt(max(0, s * r))
        h = s / (w + 1e-6)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return (float(x1), float(y1), float(x2), float(y2))
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float) -> None:
        """Update state with new detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        self.kf.update(self._bbox_to_z(bbox))
    
    def predict(self) -> Tuple[float, float, float, float]:
        """Predict next state."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._x_to_bbox(self.kf.x))
        
        return self.get_state()
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """Get current bounding box state."""
        return self._x_to_bbox(self.kf.x)


class ByteTracker(BaseTracker):
    """
    ByteTrack multi-object tracker implementation.
    
    Uses two-stage association:
    1. Match high-confidence detections to tracks
    2. Match remaining low-confidence detections to unmatched tracks
    
    This approach recovers occluded targets better than single-stage matching.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1
    ):
        """
        Initialize ByteTrack tracker.
        
        Args:
            max_age: Maximum frames to keep lost track
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
            high_threshold: Confidence threshold for first-stage matching
            low_threshold: Confidence threshold for second-stage matching
        """
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        
        self._trackers: List[KalmanBoxTracker] = []
        self._frame_count = 0
        self._geometry = GeometryUtils()
        
        # Reset track ID counter
        KalmanBoxTracker.count = 0
        
        logger.info(
            f"ByteTracker initialized: max_age={max_age}, min_hits={min_hits}, "
            f"iou_threshold={iou_threshold}"
        )
    
    def update(
    self,
    detections: List[dict],
    frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Implements ByteTrack's two-stage association algorithm.
        
        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'
            frame: Optional frame (not used in IoU-based tracking)
            
        Returns:
            List of confirmed Track objects
        """
        self._frame_count += 1
        
        # Handle empty trackers case
        if len(self._trackers) == 0 and len(detections) == 0:
            return []
        
        # Predict new locations of existing trackers
        for tracker in self._trackers:
            try:
                tracker.predict()
            except Exception as e:
                logger.warning(f"Error predicting tracker {tracker.id}: {e}")
                continue
        
        # Separate high and low confidence detections
        high_dets = []
        low_dets = []
        
        for det in detections:
            try:
                conf = det.get('confidence', 0)
                if conf >= self._high_threshold:
                    high_dets.append(det)
                elif conf >= self._low_threshold:
                    low_dets.append(det)
            except Exception as e:
                logger.warning(f"Error processing detection: {e}")
                continue
        
        # Track which trackers are matched
        matched_tracker_ids = set()
        
        # First association: high confidence detections
        if len(self._trackers) > 0 and len(high_dets) > 0:
            try:
                tracker_boxes = [t.get_state() for t in self._trackers]
                matched, unmatched_dets, unmatched_trackers = self._associate(
                    high_dets,
                    tracker_boxes,
                    self._iou_threshold
                )
                
                # Update matched trackers
                for det_idx, trk_idx in matched:
                    if trk_idx < len(self._trackers) and det_idx < len(high_dets):
                        self._trackers[trk_idx].update(
                            tuple(high_dets[det_idx]['bbox']),
                            high_dets[det_idx]['confidence']
                        )
                        matched_tracker_ids.add(trk_idx)
            except Exception as e:
                logger.error(f"Error in first association: {e}")
                unmatched_dets = list(range(len(high_dets)))
                unmatched_trackers = list(range(len(self._trackers)))
        else:
            unmatched_dets = list(range(len(high_dets)))
            unmatched_trackers = list(range(len(self._trackers)))
        
        # Second association: low confidence detections with unmatched trackers
        if len(unmatched_trackers) > 0 and len(low_dets) > 0:
            try:
                # Get unmatched tracker boxes
                unmatched_tracker_boxes = []
                valid_unmatched_indices = []
                
                for idx in unmatched_trackers:
                    if idx < len(self._trackers):
                        unmatched_tracker_boxes.append(self._trackers[idx].get_state())
                        valid_unmatched_indices.append(idx)
                
                if len(unmatched_tracker_boxes) > 0:
                    matched_low, unmatched_low_dets, unmatched_low_trks = self._associate(
                        low_dets,
                        unmatched_tracker_boxes,
                        self._iou_threshold
                    )
                    
                    # Update matched trackers
                    for det_idx, trk_idx in matched_low:
                        if trk_idx < len(valid_unmatched_indices) and det_idx < len(low_dets):
                            actual_trk_idx = valid_unmatched_indices[trk_idx]
                            self._trackers[actual_trk_idx].update(
                                tuple(low_dets[det_idx]['bbox']),
                                low_dets[det_idx]['confidence']
                            )
                            matched_tracker_ids.add(actual_trk_idx)
            except Exception as e:
                logger.error(f"Error in second association: {e}")
        
        # Create new trackers for unmatched high-confidence detections
        try:
            for det_idx in unmatched_dets:
                if det_idx < len(high_dets):
                    det = high_dets[det_idx]
                    tracker = KalmanBoxTracker(tuple(det['bbox']))
                    tracker.confidence = det['confidence']
                    self._trackers.append(tracker)
        except Exception as e:
            logger.error(f"Error creating new trackers: {e}")
        
        # Remove dead tracks
        try:
            self._trackers = [
                t for t in self._trackers
                if t.time_since_update <= self._max_age
            ]
        except Exception as e:
            logger.error(f"Error removing dead tracks: {e}")
        
        # Return confirmed tracks
        return self._get_tracks()
    
    def _associate(
    self,
    detections: List[dict],
    tracker_boxes: List[Tuple[float, float, float, float]],
    iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to trackers using Hungarian algorithm.
        
        Args:
            detections: List of detections
            tracker_boxes: List of tracker bounding boxes
            iou_threshold: Minimum IoU for valid match
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_trackers)
        """
        # Handle empty cases
        if len(tracker_boxes) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracker_boxes)))
        
        # Compute IoU cost matrix
        cost_matrix = np.zeros((len(detections), len(tracker_boxes)))
        
        try:
            for d, det in enumerate(detections):
                for t, trk_box in enumerate(tracker_boxes):
                    # Safety check for bbox
                    if 'bbox' not in det:
                        continue
                    
                    bbox = det['bbox']
                    if len(bbox) != 4:
                        continue
                    
                    iou = self._geometry.calculate_iou(tuple(bbox), trk_box)
                    cost_matrix[d, t] = 1 - iou  # Cost = 1 - IoU
        except Exception as e:
            logger.error(f"Error computing cost matrix: {e}")
            return [], list(range(len(detections))), list(range(len(tracker_boxes)))
        
        # Hungarian algorithm
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            logger.error(f"Error in Hungarian algorithm: {e}")
            return [], list(range(len(detections))), list(range(len(tracker_boxes)))
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(tracker_boxes)))
        
        for row, col in zip(row_indices, col_indices):
            # Bounds checking
            if row >= len(detections) or col >= len(tracker_boxes):
                continue
            
            if cost_matrix[row, col] > 1 - iou_threshold:
                continue
            
            matches.append((row, col))
            
            # Safe removal
            if row in unmatched_detections:
                unmatched_detections.remove(row)
            if col in unmatched_trackers:
                unmatched_trackers.remove(col)
        
        return matches, unmatched_detections, unmatched_trackers
    
    def _get_tracks(self) -> List[Track]:
        """Get currently active confirmed tracks."""
        tracks = []
        
        for tracker in self._trackers:
            # Only return confirmed tracks
            if tracker.hit_streak >= self._min_hits or self._frame_count <= self._min_hits:
                if tracker.time_since_update == 0:  # Active in current frame
                    state = TrackState.CONFIRMED
                elif tracker.time_since_update <= self._max_age // 2:
                    state = TrackState.LOST
                else:
                    continue
                
                track = Track(
                    track_id=tracker.id,
                    bbox=tracker.get_state(),
                    confidence=tracker.confidence,
                    state=state,
                    age=tracker.age,
                    hits=tracker.hits,
                    time_since_update=tracker.time_since_update,
                    history=list(tracker.history[-30:])  # Keep last 30 positions
                )
                tracks.append(track)
        
        return tracks
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._trackers.clear()
        self._frame_count = 0
        KalmanBoxTracker.count = 0
        logger.debug("ByteTracker reset")
    
    @property
    def active_tracks(self) -> List[Track]:
        """Get currently active tracks."""
        return self._get_tracks()
    
    @property
    def track_count(self) -> int:
        """Get total number of tracks created."""
        return KalmanBoxTracker.count