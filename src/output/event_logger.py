"""
Event logging and export.

Design Choice:
- Support for both JSON and CSV output formats
- Structured event logs with video metadata
- Incremental writing for long videos
"""

import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..events.base_event_detector import Event
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EventLogger:
    """Handles logging events to JSON and CSV files."""

    def __init__(
        self,
        output_dir: str,
        video_name: str,
        fps: float,
        total_frames: Optional[int] = None
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._video_name = video_name
        self._fps = fps
        self._total_frames = total_frames

        # Generate output filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(video_name).stem

        self._json_path = self._output_dir / f"{base_name}_{timestamp}_events.json"
        self._csv_path = self._output_dir / f"{base_name}_{timestamp}_events.csv"

        logger.info(f"EventLogger initialized: {self._output_dir}")

    def save(
        self,
        events: List[Event],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """Save events to JSON and CSV files."""
        if not events:
            logger.warning("No events to save!")
            # Still create empty files
            self._save_json([], {}, metadata)
            self._save_csv([])
            return {
                'json': self._json_path,
                'csv': self._csv_path
            }

        # Prepare event data
        event_dicts = [e.to_dict() for e in events]

        # Calculate summary
        summary = self._calculate_summary(events)

        # Save JSON
        self._save_json(event_dicts, summary, metadata)

        # Save CSV
        self._save_csv(event_dicts)

        logger.info(
            f"Events saved: {len(events)} events to "
            f"{self._json_path.name} and {self._csv_path.name}"
        )

        return {
            'json': self._json_path,
            'csv': self._csv_path
        }

    def _save_json(
        self,
        events: List[Dict],
        summary: Dict,
        metadata: Optional[Dict] = None
    ) -> None:
        """Save events to JSON file."""
        output = {
            'video': self._video_name,
            'processed_at': datetime.now().isoformat(),
            'fps': self._fps,
            'total_frames': self._total_frames,
            'events': events,
            'summary': summary
        }

        if metadata:
            output['metadata'] = metadata

        with open(self._json_path, 'w') as f:
            json.dump(output, f, indent=2)

    def _save_csv(self, events: List[Dict]) -> None:
        """Save events to CSV file."""
        if not events:
            # Write empty CSV with headers
            headers = [
                'event_id', 'type', 'track_id', 'zone_id', 'zone_name',
                'frame_number', 'timestamp_seconds', 'duration_seconds',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence'
            ]
            with open(self._csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            return

        # Flatten events for CSV
        rows = []
        for event in events:
            bbox = event.get('bbox', [0, 0, 0, 0])
            row = {
                'event_id': event.get('event_id', ''),
                'type': event.get('type', ''),
                'track_id': event.get('track_id', ''),
                'zone_id': event.get('zone_id', ''),
                'zone_name': event.get('zone_name', ''),
                'frame_number': event.get('frame_number', ''),
                'timestamp_seconds': event.get('timestamp_seconds', ''),
                'duration_seconds': event.get('duration_seconds', ''),
                'bbox_x1': bbox[0] if len(bbox) > 0 else '',
                'bbox_y1': bbox[1] if len(bbox) > 1 else '',
                'bbox_x2': bbox[2] if len(bbox) > 2 else '',
                'bbox_y2': bbox[3] if len(bbox) > 3 else '',
                'confidence': event.get('confidence', '')
            }
            rows.append(row)

        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _calculate_summary(self, events: List[Event]) -> Dict:
        """Calculate event summary statistics."""
        summary = {
            'total_events': len(events),
            'intrusion_count': 0,
            'loitering_count': 0,
            'unique_tracks': set()
        }

        for event in events:
            if event.event_type.value == 'INTRUSION':
                summary['intrusion_count'] += 1
            elif event.event_type.value == 'LOITERING':
                summary['loitering_count'] += 1

            summary['unique_tracks'].add(event.track_id)

        summary['unique_tracks'] = len(summary['unique_tracks'])

        return summary
    
    @property
    def json_path(self) -> Path:
        """Get JSON output path."""
        return self._json_path
    
    @property
    def csv_path(self) -> Path:
        """Get CSV output path."""
        return self._csv_path