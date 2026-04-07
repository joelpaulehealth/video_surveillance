"""
YOLOv8 person detector implementation.

Design Choice:
- Using Ultralytics YOLOv8 for state-of-the-art speed/accuracy
- Pre-trained on COCO dataset (person class well-represented)
- Supports multiple model sizes (nano to extra-large)
- Built-in GPU acceleration and optimizations

Trade-offs:
- YOLOv8n (nano): Fastest, slight accuracy reduction
- YOLOv8m (medium): Good balance
- YOLOv8x (extra-large): Most accurate, slowest

Alternative considered: Faster R-CNN (more accurate but 5-10x slower)
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from .base_detector import BaseDetector, Detection
from ..utils.logger import get_logger

logger = get_logger(__name__)


class YOLODetector(BaseDetector):
    """
    YOLOv8-based person detector.
    
    Provides efficient person detection with configurable
    confidence thresholds and model sizes.
    """
    
    # COCO class names mapping
    COCO_PERSON_CLASS = 0
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
        max_detections: int = 100
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to YOLO model weights or model name
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cuda', 'cpu', or None for auto)
            classes: List of class IDs to detect (default: [0] for person)
            max_detections: Maximum detections per frame
        """
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._classes = classes or [self.COCO_PERSON_CLASS]
        self._max_detections = max_detections
        
        # Auto-select device if not specified
        if device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        
        # Load model
        logger.info(f"Loading YOLO model: {model_path} on {self._device}")
        self._model = YOLO(model_path)
        self._model.to(self._device)
        
        # Get class names from model
        self._class_names = self._model.names
        
        logger.info(
            f"YOLO detector initialized: "
            f"conf={confidence_threshold}, "
            f"iou={iou_threshold}, "
            f"classes={self._classes}"
        )
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in a single frame.
        
        Args:
            frame: Input frame (BGR, numpy array)
            
        Returns:
            List of Detection objects for persons
        """
        # Run inference
        results = self._model(
            frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            classes=self._classes,
            max_det=self._max_detections,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bounding box
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox
                
                # Get confidence
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Get class
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self._class_names.get(class_id, "unknown")
                
                detection = Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect persons in a batch of frames.
        
        More efficient for GPU processing as it processes
        multiple frames in a single forward pass.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists, one per frame
        """
        if not frames:
            return []
        
        # Run batch inference
        results = self._model(
            frames,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            classes=self._classes,
            max_det=self._max_detections,
            verbose=False
        )
        
        batch_detections = []
        
        for result in results:
            frame_detections = []
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self._class_names.get(class_id, "unknown")
                    
                    detection = Detection(
                        bbox=(float(bbox[0]), float(bbox[1]), 
                              float(bbox[2]), float(bbox[3])),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    )
                    frame_detections.append(detection)
            
            batch_detections.append(frame_detections)
        
        return batch_detections
    
    def warmup(self, input_size: Tuple[int, int] = (640, 480)) -> None:
        """
        Warm up the model with dummy inference.
        
        Args:
            input_size: (width, height) of dummy input
        """
        logger.debug("Warming up YOLO model...")
        dummy_frame = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        
        # Run a few warmup iterations
        for _ in range(3):
            self.detect(dummy_frame)
        
        logger.debug("YOLO warmup complete")
    
    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        return self._device
    
    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_path
    
    @property
    def confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return self._confidence_threshold
    
    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        """Set confidence threshold."""
        self._confidence_threshold = max(0.0, min(1.0, value))
    
    def get_model_info(self) -> dict:
        """Get model information for logging."""
        return {
            'model': self._model_path,
            'device': self._device,
            'confidence_threshold': self._confidence_threshold,
            'iou_threshold': self._iou_threshold,
            'classes': self._classes,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }