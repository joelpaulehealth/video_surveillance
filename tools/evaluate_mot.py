#!/usr/bin/env python3
"""
MOT (Multi-Object Tracking) Metrics Evaluation.

Compares tracking results against ground truth annotations
and computes standard MOT metrics (MOTA, MOTP, IDF1, etc.).

Usage:
    python tools/evaluate_mot.py --video video.mp4 --gt ground_truth.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import csv

from src.pipeline import VideoProcessor
from src.utils import Config, get_logger

logger = get_logger(__name__)


class MOTMetrics:
    """
    Calculate MOT benchmark metrics.
    
    Metrics:
    - MOTA (Multi-Object Tracking Accuracy)
    - MOTP (Multi-Object Tracking Precision)
    - IDF1 (ID F1 Score)
    - MT (Mostly Tracked)
    - ML (Mostly Lost)
    - FP (False Positives)
    - FN (False Negatives)
    - ID Switches
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            iou_threshold: IoU threshold for matching
        """
        self.iou_threshold = iou_threshold
        
        # Accumulators
        self.num_frames = 0
        self.num_misses = 0
        self.num_false_positives = 0
        self.num_mismatches = 0
        self.num_matches = 0
        
        self.sum_iou = 0.0
        
        # Per-GT track statistics
        self.gt_track_lengths = defaultdict(int)
        self.tracked_frames = defaultdict(int)  # frames where GT was matched
        
        # For ID switches
        self.last_matched_id = {}  # {gt_id: pred_id}
    
    def update(
        self,
        gt_boxes: List[Tuple[int, float, float, float, float]],
        pred_boxes: List[Tuple[int, float, float, float, float]]
    ) -> None:
        """
        Update metrics for one frame.
        
        Args:
            gt_boxes: List of (id, x1, y1, x2, y2) for ground truth
            pred_boxes: List of (id, x1, y1, x2, y2) for predictions
        """
        self.num_frames += 1
        
        # Update GT track lengths
        for gt_id, *_ in gt_boxes:
            self.gt_track_lengths[gt_id] += 1
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(gt_boxes, pred_boxes)
        
        # Greedy matching based on IoU
        matched_gt = set()
        matched_pred = set()
        matches = []
        
        # Sort by IoU (descending)
        iou_pairs = []
        for i, (gt_id, *gt_box) in enumerate(gt_boxes):
            for j, (pred_id, *pred_box) in enumerate(pred_boxes):
                iou = iou_matrix[i, j]
                if iou >= self.iou_threshold:
                    iou_pairs.append((iou, i, j, gt_id, pred_id))
        
        iou_pairs.sort(reverse=True)
        
        for iou, i, j, gt_id, pred_id in iou_pairs:
            if i not in matched_gt and j not in matched_pred:
                matches.append((i, j, gt_id, pred_id, iou))
                matched_gt.add(i)
                matched_pred.add(j)
        
        # Count matches
        self.num_matches += len(matches)
        
        # Count misses (false negatives)
        self.num_misses += len(gt_boxes) - len(matched_gt)
        
        # Count false positives
        self.num_false_positives += len(pred_boxes) - len(matched_pred)
        
        # Check for ID switches
        for i, j, gt_id, pred_id, iou in matches:
            self.sum_iou += iou
            self.tracked_frames[gt_id] += 1
            
            if gt_id in self.last_matched_id:
                if self.last_matched_id[gt_id] != pred_id:
                    self.num_mismatches += 1
            
            self.last_matched_id[gt_id] = pred_id
    
    def _compute_iou_matrix(
        self,
        gt_boxes: List[Tuple],
        pred_boxes: List[Tuple]
    ) -> np.ndarray:
        """Compute IoU matrix between GT and predictions."""
        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        
        iou_matrix = np.zeros((n_gt, n_pred))
        
        for i, (_, *gt_box) in enumerate(gt_boxes):
            for j, (_, *pred_box) in enumerate(pred_boxes):
                iou_matrix[i, j] = self._calculate_iou(gt_box, pred_box)
        
        return iou_matrix
    
    def _calculate_iou(
        self,
        box1: List[float],
        box2: List[float]
    ) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics."""
        # MOTA = 1 - (FN + FP + IDSW) / GT
        total_gt = sum(self.gt_track_lengths.values())
        
        if total_gt == 0:
            return {
                'MOTA': 0.0,
                'MOTP': 0.0,
                'IDF1': 0.0,
                'MT': 0.0,
                'ML': 0.0,
                'FP': 0,
                'FN': 0,
                'ID_switches': 0
            }
        
        mota = 1.0 - (self.num_misses + self.num_false_positives + self.num_mismatches) / total_gt
        mota = max(0.0, min(1.0, mota))  # Clamp to [0, 1]
        
        # MOTP = average IoU of matched boxes
        motp = self.sum_iou / self.num_matches if self.num_matches > 0 else 0.0
        
        # MT (Mostly Tracked): tracks covered >= 80% of their length
        mt_count = sum(
            1 for gt_id, length in self.gt_track_lengths.items()
            if self.tracked_frames[gt_id] / length >= 0.8
        )
        mt_ratio = mt_count / len(self.gt_track_lengths) if self.gt_track_lengths else 0.0
        
        # ML (Mostly Lost): tracks covered <= 20% of their length
        ml_count = sum(
            1 for gt_id, length in self.gt_track_lengths.items()
            if self.tracked_frames[gt_id] / length <= 0.2
        )
        ml_ratio = ml_count / len(self.gt_track_lengths) if self.gt_track_lengths else 0.0
        
        # IDF1 (simplified - would need full ID mapping for exact calculation)
        idf1 = self.num_matches / (self.num_matches + 0.5 * (self.num_false_positives + self.num_misses))
        
        return {
            'MOTA': round(mota * 100, 2),
            'MOTP': round(motp * 100, 2),
            'IDF1': round(idf1 * 100, 2),
            'MT': round(mt_ratio * 100, 2),
            'ML': round(ml_ratio * 100, 2),
            'FP': self.num_false_positives,
            'FN': self.num_misses,
            'ID_switches': self.num_mismatches,
            'num_frames': self.num_frames,
            'num_gt_tracks': len(self.gt_track_lengths)
        }


def load_mot_ground_truth(gt_file: str) -> Dict[int, List[Tuple]]:
    """
    Load MOT Challenge format ground truth.
    
    Format: frame,id,x,y,w,h,conf,class,visibility
    
    Returns:
        Dictionary mapping frame_number to list of (id, x1, y1, x2, y2)
    """
    gt_data = defaultdict(list)
    
    with open(gt_file, 'r') as f:
        reader = csv.reader(f)
        
        for row in reader:
            if len(row) < 6:
                continue
            
            frame = int(row[0])
            track_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            
            # Convert to x1, y1, x2, y2
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            gt_data[frame].append((track_id, x1, y1, x2, y2))
    
    return dict(gt_data)


def run_evaluation(video_path: str, gt_file: str, config: Config) -> Dict:
    """Run full evaluation pipeline."""
    logger.info(f"Evaluating: {video_path}")
    
    # Load ground truth
    gt_data = load_mot_ground_truth(gt_file)
    logger.info(f"Loaded ground truth: {len(gt_data)} frames")
    
    # Initialize metrics
    metrics_calculator = MOTMetrics(iou_threshold=0.5)
    
    # Run tracking (we need to capture per-frame results)
    # For now, this is a placeholder - would need to modify VideoProcessor
    # to return per-frame tracking results
    
    # TODO: Integrate with VideoProcessor to get per-frame tracks
    # For demonstration, showing the structure
    
    logger.warning("Full integration with VideoProcessor needed for live evaluation")
    logger.info("Placeholder metrics returned")
    
    return {
        'status': 'placeholder',
        'message': 'Full integration pending',
        'expected_metrics': ['MOTA', 'MOTP', 'IDF1', 'MT', 'ML', 'FP', 'FN', 'ID_switches']
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MOT Metrics Evaluation')
    parser.add_argument('--video', '-v', required=True, help='Video file')
    parser.add_argument('--gt', '-g', required=True, help='Ground truth file (MOT format)')
    parser.add_argument('--config', '-c', default='configs/default_config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    try:
        config = Config(config_path=args.config)
        results = run_evaluation(args.video, args.gt, config)
        
        print("\n" + "="*60)
        print("MOT EVALUATION RESULTS")
        print("="*60)
        
        for key, value in results.items():
            print(f"{key}: {value}")
        
        print("="*60 + "\n")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())