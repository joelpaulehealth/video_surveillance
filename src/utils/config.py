"""
Configuration management for the surveillance system.

Design Choice:
- YAML for human-readable configuration
- JSON for zone definitions (easier polygon coordinate editing)
- Dataclass-style access for type safety
- Environment variable overrides for deployment flexibility
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import yaml

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoConfig:
    """Video input/output configuration."""
    input_path: str = ""
    output_path: str = "output/videos/annotated.mp4"
    frame_skip: int = 1
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None


@dataclass
class DetectionConfig:
    """Person detection configuration."""
    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cuda"
    classes: List[int] = field(default_factory=lambda: [0])  # 0 = person
    max_detections: int = 100


@dataclass
class TrackingConfig:
    """Multi-object tracking configuration."""
    tracker: str = "bytetrack"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class EventConfig:
    """Event detection configuration."""
    zones_config: str = "video_surveillance/configs/zone_examples.json" # update to configs/zone_examples.json if not working
    event_log_path: str = "output/events/events.json"
    deduplicate_window_seconds: float = 2.0
    default_loitering_threshold: float = 10.0
    default_movement_threshold: float = 25.0


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    draw_boxes: bool = True
    draw_tracks: bool = True
    draw_zones: bool = True
    draw_events: bool = True
    draw_fps: bool = True
    font_scale: float = 0.6
    box_thickness: int = 2
    track_history_length: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_file: Optional[str] = "output/surveillance.log"


@dataclass
class ZoneDefinition:
    """Single zone definition."""
    id: str
    name: str
    type: str  # "intrusion" or "loitering"
    polygon: List[List[int]]
    enabled: bool = True
    loitering_threshold_seconds: float = 10.0
    movement_threshold_pixels: float = 25.0


class Config:
    """
    Main configuration class that loads and validates all settings.
    
    Supports loading from YAML file with environment variable overrides.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        zones_path: Optional[str] = None
    ):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file
            zones_path: Path to zones JSON file (overrides config)
        """
        self.video = VideoConfig()
        self.detection = DetectionConfig()
        self.tracking = TrackingConfig()
        self.events = EventConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        self.zones: List[ZoneDefinition] = []
        
        if config_path:
            self._load_yaml(config_path)
        
        # Override zones path if provided
        if zones_path:
            self.events.zones_config = zones_path
        
        # Load zones
        self._load_zones()
        
        # Apply environment overrides
        self._apply_env_overrides()
        
        logger.info(f"Configuration loaded: {len(self.zones)} zones defined")
    
    def _load_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data:
            return
        
        # Map YAML sections to dataclasses
        if 'video' in data:
            self.video = VideoConfig(**data['video'])
        if 'detection' in data:
            self.detection = DetectionConfig(**data['detection'])
        if 'tracking' in data:
            self.tracking = TrackingConfig(**data['tracking'])
        if 'events' in data:
            self.events = EventConfig(**data['events'])
        if 'visualization' in data:
            self.visualization = VisualizationConfig(**data['visualization'])
        if 'logging' in data:
            self.logging = LoggingConfig(**data['logging'])
        
        logger.debug(f"Loaded config from {config_path}")
    
    def _load_zones(self) -> None:
        """Load zone definitions from JSON file."""
        zones_path = Path(self.events.zones_config)
        
        if not zones_path.exists():
            logger.warning(f"Zones file not found: {zones_path}")
            return
        
        with open(zones_path, 'r') as f:
            data = json.load(f)
        
        self.zones = []
        for zone_data in data.get('zones', []):
            zone = ZoneDefinition(
                id=zone_data['id'],
                name=zone_data['name'],
                type=zone_data['type'],
                polygon=zone_data['polygon'],
                enabled=zone_data.get('enabled', True),
                loitering_threshold_seconds=zone_data.get(
                    'loitering_threshold_seconds',
                    self.events.default_loitering_threshold
                ),
                movement_threshold_pixels=zone_data.get(
                    'movement_threshold_pixels',
                    self.events.default_movement_threshold
                )
            )
            self.zones.append(zone)
        
        logger.debug(f"Loaded {len(self.zones)} zones")
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Device override
        if os.environ.get('SURVEILLANCE_DEVICE'):
            self.detection.device = os.environ['SURVEILLANCE_DEVICE']
        
        # Log level override
        if os.environ.get('SURVEILLANCE_LOG_LEVEL'):
            self.logging.level = os.environ['SURVEILLANCE_LOG_LEVEL']
        
        # Confidence threshold override
        if os.environ.get('SURVEILLANCE_CONFIDENCE'):
            self.detection.confidence_threshold = float(
                os.environ['SURVEILLANCE_CONFIDENCE']
            )
    
    def get_intrusion_zones(self) -> List[ZoneDefinition]:
        """Get all enabled intrusion zones."""
        return [z for z in self.zones if z.type == 'intrusion' and z.enabled]
    
    def get_loitering_zones(self) -> List[ZoneDefinition]:
        """Get all enabled loitering zones."""
        return [z for z in self.zones if z.type == 'loitering' and z.enabled]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'video': self.video.__dict__,
            'detection': self.detection.__dict__,
            'tracking': self.tracking.__dict__,
            'events': self.events.__dict__,
            'visualization': self.visualization.__dict__,
            'logging': self.logging.__dict__,
            'zones_count': len(self.zones)
        }