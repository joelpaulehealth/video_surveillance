#!/usr/bin/env python3
"""
Interactive Zone Editor for defining surveillance zones.

Usage:
    python tools/zone_editor.py --video sample.mp4 --output zones.json

Controls:
    - Left click: Add polygon point
    - Right click: Complete current polygon
    - 'i': Set current polygon as intrusion zone
    - 'l': Set current polygon as loitering zone
    - 'u': Undo last point
    - 'd': Delete last polygon
    - 's': Save zones to file
    - 'q': Quit
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import json
import argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict


@dataclass
class ZoneData:
    """Zone data for editor."""
    points: List[List[int]]
    zone_type: str  # 'intrusion' or 'loitering'
    name: str
    completed: bool = False
    loitering_threshold: float = 10.0
    movement_threshold: float = 25.0


class ZoneEditor:
    """
    Interactive zone editor with OpenCV.
    
    Allows users to draw polygons on a video frame
    and configure zone properties.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize zone editor.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        
        # Load first frame
        cap = cv2.VideoCapture(video_path)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to read video: {video_path}")
        
        # Make a copy for drawing
        self.display_frame = self.frame.copy()
        
        # Zone data
        self.zones: List[ZoneData] = []
        self.current_points: List[Tuple[int, int]] = []
        self.current_type: str = 'intrusion'
        
        # UI state
        self.zone_counter = {'intrusion': 1, 'loitering': 1}
        
        # Colors
        self.colors = {
            'intrusion': (0, 0, 255),      # Red
            'loitering': (0, 255, 255),    # Yellow
            'current': (0, 255, 0),        # Green
            'point': (255, 255, 255)       # White
        }
        
        print("\n" + "="*60)
        print("ZONE EDITOR")
        print("="*60)
        print("\nControls:")
        print("  Left Click    - Add polygon point")
        print("  Right Click   - Complete current polygon")
        print("  'i'           - Set as intrusion zone")
        print("  'l'           - Set as loitering zone")
        print("  'u'           - Undo last point")
        print("  'd'           - Delete last polygon")
        print("  's'           - Save zones")
        print("  'q' or ESC    - Quit")
        print("\nCurrent mode: INTRUSION")
        print("="*60 + "\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.current_points.append((x, y))
            self.redraw()
            print(f"Point added: ({x}, {y}) - Total: {len(self.current_points)}")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete polygon
            if len(self.current_points) >= 3:
                self._complete_polygon()
            else:
                print("Need at least 3 points to complete polygon")
    
    def _complete_polygon(self):
        """Complete current polygon and add to zones."""
        zone_type = self.current_type
        zone_num = self.zone_counter[zone_type]
        
        zone = ZoneData(
            points=[[int(p[0]), int(p[1])] for p in self.current_points],
            zone_type=zone_type,
            name=f"{zone_type.capitalize()} Zone {zone_num}",
            completed=True
        )
        
        self.zones.append(zone)
        self.zone_counter[zone_type] += 1
        
        print(f"\n✓ {zone.name} completed with {len(self.current_points)} points")
        
        # Reset current points
        self.current_points = []
        self.redraw()
    
    def redraw(self):
        """Redraw the display frame."""
        # Start with original frame
        self.display_frame = self.frame.copy()
        
        # Draw completed zones
        for zone in self.zones:
            color = self.colors[zone.zone_type]
            points = np.array(zone.points, dtype=np.int32)
            
            # Draw filled polygon with transparency
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
            
            # Draw border
            cv2.polylines(self.display_frame, [points], True, color, 2)
            
            # Draw label
            centroid = np.mean(points, axis=0).astype(int)
            cv2.putText(
                self.display_frame,
                zone.name,
                (centroid[0] - 50, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Draw current polygon being drawn
        if len(self.current_points) > 0:
            # Draw points
            for point in self.current_points:
                cv2.circle(self.display_frame, point, 5, self.colors['point'], -1)
            
            # Draw lines
            if len(self.current_points) > 1:
                points = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [points], False, self.colors['current'], 2)
        
        # Draw status bar
        status = f"Mode: {self.current_type.upper()} | Zones: {len(self.zones)} | Points: {len(self.current_points)}"
        cv2.rectangle(self.display_frame, (0, 0), (self.display_frame.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            self.display_frame,
            status,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        cv2.imshow('Zone Editor', self.display_frame)
    
    def run(self) -> List[ZoneData]:
        """Run the interactive editor."""
        cv2.namedWindow('Zone Editor')
        cv2.setMouseCallback('Zone Editor', self.mouse_callback)
        
        self.redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
            elif key == ord('i'):
                # Switch to intrusion mode
                self.current_type = 'intrusion'
                print("\nMode: INTRUSION")
                self.redraw()
            
            elif key == ord('l'):
                # Switch to loitering mode
                self.current_type = 'loitering'
                print("\nMode: LOITERING")
                self.redraw()
            
            elif key == ord('u'):
                # Undo last point
                if len(self.current_points) > 0:
                    removed = self.current_points.pop()
                    print(f"Removed point: {removed}")
                    self.redraw()
            
            elif key == ord('d'):
                # Delete last zone
                if len(self.zones) > 0:
                    removed = self.zones.pop()
                    print(f"\nDeleted: {removed.name}")
                    self.redraw()
            
            elif key == ord('s'):
                # Save
                if len(self.zones) > 0:
                    return self.zones
                else:
                    print("\nNo zones to save!")
        
        cv2.destroyAllWindows()
        return self.zones
    
    def save_zones(self, output_path: str) -> None:
        """
        Save zones to JSON file.
        
        Args:
            output_path: Output JSON file path
        """
        zones_data = {
            "zones": []
        }
        
        for i, zone in enumerate(self.zones):
            zone_dict = {
                "id": f"{zone.zone_type}_{i+1}",
                "name": zone.name,
                "type": zone.zone_type,
                "polygon": zone.points,
                "enabled": True
            }
            
            if zone.zone_type == 'loitering':
                zone_dict["loitering_threshold_seconds"] = zone.loitering_threshold
                zone_dict["movement_threshold_pixels"] = zone.movement_threshold
            
            zones_data["zones"].append(zone_dict)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(zones_data, f, indent=2)
        
        print(f"\n✓ Zones saved to: {output_path}")
        print(f"  Total zones: {len(self.zones)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Interactive Zone Editor')
    parser.add_argument(
        '--video', '-v',
        required=True,
        help='Path to video file'
    )
    parser.add_argument(
        '--output', '-o',
        default='configs/zones_custom.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    try:
        editor = ZoneEditor(args.video)
        zones = editor.run()
        
        if zones:
            editor.save_zones(args.output)
            print("\nZone editor completed successfully!")
        else:
            print("\nNo zones created.")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())