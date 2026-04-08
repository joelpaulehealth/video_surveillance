#!/usr/bin/env python3
"""
Simple test script for loitering detection.
Simulates a person standing still in a zone to verify loitering works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from src.events import EventManager, ZoneManager
from src.events.base_event_detector import EventType
from src.utils.config import Config, ZoneDefinition


def test_loitering_simple():
    """
    Test loitering detection with simulated tracks.
    
    Simulates a person standing still in a zone for several seconds.
    """
    print("=" * 60)
    print("LOITERING DETECTION TEST")
    print("=" * 60)
    
    # Create a simple loitering zone
    zones = [
        ZoneDefinition(
            id="test_loiter_zone",
            name="Test Loitering Zone",
            type="loitering",
            polygon=[[0, 0], [500, 0], [500, 500], [0, 500]],
            enabled=True,
            loitering_threshold_seconds=3.0,  # 3 seconds for quick testing
            movement_threshold_pixels=30.0
        )
    ]
    
    # Create config with test zone
    config = Config()
    config.zones = zones
    
    # Create event manager
    fps = 10.0  # 10 frames per second for easy calculation
    event_manager = EventManager(config, fps=fps)
    
    print(f"\nTest Configuration:")
    print(f"  - Zone: {zones[0].name}")
    print(f"  - Zone type: {zones[0].type}")
    print(f"  - Zone polygon: {zones[0].polygon}")
    print(f"  - Loitering threshold: {zones[0].loitering_threshold_seconds} seconds")
    print(f"  - Movement threshold: {zones[0].movement_threshold_pixels} pixels")
    print(f"  - FPS: {fps}")
    
    print("\n" + "-" * 60)
    print("SCENARIO 1: Person standing still (should trigger loitering)")
    print("-" * 60)
    
    # Simulate a person standing still at position (250, 250)
    # This is inside the zone [0,0] to [500,500]
    all_events = []
    
    # Simulate 50 frames (5 seconds at 10 FPS)
    for frame in range(50):
        # Person stays at same position (small random movement)
        track_data = [
            {
                'track_id': 1,
                'bbox': [230, 230, 270, 270],  # Centered at (250, 250)
                'confidence': 0.9,
                'in_zone': False
            }
        ]
        
        timestamp = frame / fps
        events = event_manager.process_frame(track_data, frame, timestamp)
        
        if events:
            all_events.extend(events)
            for event in events:
                print(f"  Frame {frame} ({timestamp:.1f}s): EVENT DETECTED!")
                print(f"    Type: {event.event_type.value}")
                print(f"    Track ID: {event.track_id}")
                print(f"    Zone: {event.zone_name}")
                print(f"    Duration: {event.duration_seconds:.1f}s")
        elif frame % 10 == 0:
            print(f"  Frame {frame} ({timestamp:.1f}s): No event yet...")
    
    loitering_events = [e for e in all_events if e.event_type == EventType.LOITERING]
    
    print(f"\n  Result: {len(loitering_events)} loitering event(s) detected")
    
    if len(loitering_events) > 0:
        print("  STATUS: PASS - Loitering detection working!")
    else:
        print("  STATUS: FAIL - No loitering detected")
    
    # Reset for next test
    event_manager.reset()
    
    print("\n" + "-" * 60)
    print("SCENARIO 2: Person walking through (should NOT trigger loitering)")
    print("-" * 60)
    
    all_events = []
    
    # Simulate person walking across zone
    for frame in range(50):
        # Person moves from left to right
        x_pos = 50 + frame * 8  # Moves 8 pixels per frame
        
        track_data = [
            {
                'track_id': 2,
                'bbox': [x_pos, 230, x_pos + 40, 270],
                'confidence': 0.9,
                'in_zone': False
            }
        ]
        
        timestamp = frame / fps
        events = event_manager.process_frame(track_data, frame, timestamp)
        
        if events:
            all_events.extend(events)
            for event in events:
                if event.event_type == EventType.LOITERING:
                    print(f"  Frame {frame}: Loitering detected (unexpected)")
    
    loitering_events = [e for e in all_events if e.event_type == EventType.LOITERING]
    
    print(f"\n  Result: {len(loitering_events)} loitering event(s) detected")
    
    if len(loitering_events) == 0:
        print("  STATUS: PASS - No false loitering for moving person!")
    else:
        print("  STATUS: FAIL - False loitering detected for moving person")
    
    # Reset for next test
    event_manager.reset()
    
    print("\n" + "-" * 60)
    print("SCENARIO 3: Person outside zone (should NOT trigger)")
    print("-" * 60)
    
    all_events = []
    
    # Simulate person standing still but OUTSIDE the zone
    for frame in range(50):
        # Person at position (600, 600) - outside zone [0,0] to [500,500]
        track_data = [
            {
                'track_id': 3,
                'bbox': [580, 580, 620, 620],  # Outside zone
                'confidence': 0.9,
                'in_zone': False
            }
        ]
        
        timestamp = frame / fps
        events = event_manager.process_frame(track_data, frame, timestamp)
        
        if events:
            all_events.extend(events)
    
    loitering_events = [e for e in all_events if e.event_type == EventType.LOITERING]
    
    print(f"\n  Result: {len(loitering_events)} loitering event(s) detected")
    
    if len(loitering_events) == 0:
        print("  STATUS: PASS - No loitering for person outside zone!")
    else:
        print("  STATUS: FAIL - Loitering detected outside zone")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    return loitering_events


def test_loitering_with_video_simulation():
    """
    More detailed test simulating realistic video processing.
    """
    print("\n" + "=" * 60)
    print("DETAILED LOITERING TEST (Video Simulation)")
    print("=" * 60)
    
    # Create zones
    zones = [
        ZoneDefinition(
            id="lobby_zone",
            name="Lobby Area",
            type="loitering",
            polygon=[[100, 100], [400, 100], [400, 400], [100, 400]],
            enabled=True,
            loitering_threshold_seconds=5.0,
            movement_threshold_pixels=25.0
        ),
        ZoneDefinition(
            id="entrance_zone",
            name="Entrance",
            type="intrusion",
            polygon=[[0, 0], [100, 0], [100, 500], [0, 500]],
            enabled=True
        )
    ]
    
    config = Config()
    config.zones = zones
    
    fps = 30.0  # Realistic 30 FPS
    event_manager = EventManager(config, fps=fps)
    
    print(f"\nConfiguration:")
    print(f"  - FPS: {fps}")
    print(f"  - Loitering zone: Lobby Area (threshold: 5 seconds)")
    print(f"  - Intrusion zone: Entrance")
    
    print("\nSimulating 10 seconds of video with 2 people...")
    print("-" * 60)
    
    all_events = []
    
    # Simulate 300 frames (10 seconds at 30 FPS)
    for frame in range(300):
        timestamp = frame / fps
        
        tracks = []
        
        # Person 1: Stands in lobby (should trigger loitering after 5 seconds)
        # Small random movement to simulate real standing
        import random
        jitter = random.randint(-5, 5)
        
        tracks.append({
            'track_id': 1,
            'bbox': [200 + jitter, 200 + jitter, 250 + jitter, 300 + jitter],
            'confidence': 0.9,
            'in_zone': False
        })
        
        # Person 2: Walks through lobby (should NOT trigger loitering)
        x_pos = 100 + (frame % 300)  # Walks across
        if x_pos < 400:  # Only while in frame
            tracks.append({
                'track_id': 2,
                'bbox': [x_pos, 250, x_pos + 40, 350],
                'confidence': 0.85,
                'in_zone': False
            })
        
        # Process frame
        events = event_manager.process_frame(tracks, frame, timestamp)
        
        if events:
            all_events.extend(events)
            for event in events:
                print(f"  {timestamp:.1f}s: {event.event_type.value} - "
                      f"Track {event.track_id} in {event.zone_name}")
                if event.event_type == EventType.LOITERING:
                    print(f"         Duration: {event.duration_seconds:.1f} seconds")
    
    # Summary
    print("\n" + "-" * 60)
    print("Results:")
    
    loitering = [e for e in all_events if e.event_type == EventType.LOITERING]
    intrusion = [e for e in all_events if e.event_type == EventType.INTRUSION]
    
    print(f"  Loitering events: {len(loitering)}")
    print(f"  Intrusion events: {len(intrusion)}")
    print(f"  Total events: {len(all_events)}")
    
    if loitering:
        print("\n  Loitering details:")
        for e in loitering:
            print(f"    - Track {e.track_id}: {e.duration_seconds:.1f}s in {e.zone_name}")
    
    # Verify
    print("\n" + "-" * 60)
    
    # Person 1 should have triggered loitering (stood still for 10 seconds > 5 second threshold)
    person1_loitering = [e for e in loitering if e.track_id == 1]
    
    # Person 2 should NOT have triggered loitering (was walking)
    person2_loitering = [e for e in loitering if e.track_id == 2]
    
    if len(person1_loitering) > 0 and len(person2_loitering) == 0:
        print("OVERALL: PASS")
        print("  - Standing person triggered loitering: YES")
        print("  - Walking person triggered loitering: NO")
    else:
        print("OVERALL: ISSUES DETECTED")
        print(f"  - Standing person loitering events: {len(person1_loitering)}")
        print(f"  - Walking person loitering events: {len(person2_loitering)}")


def test_loitering_threshold_variations():
    """
    Test different loitering thresholds.
    """
    print("\n" + "=" * 60)
    print("THRESHOLD VARIATION TEST")
    print("=" * 60)
    
    thresholds = [2.0, 5.0, 10.0]
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold} seconds ---")
        
        zones = [
            ZoneDefinition(
                id="test_zone",
                name="Test Zone",
                type="loitering",
                polygon=[[0, 0], [500, 0], [500, 500], [0, 500]],
                enabled=True,
                loitering_threshold_seconds=threshold,
                movement_threshold_pixels=30.0
            )
        ]
        
        config = Config()
        config.zones = zones
        
        fps = 10.0
        event_manager = EventManager(config, fps=fps)
        
        # Simulate person standing for 8 seconds
        duration_frames = int(8 * fps)  # 8 seconds
        events_detected = []
        
        for frame in range(duration_frames):
            track_data = [{
                'track_id': 1,
                'bbox': [200, 200, 240, 280],
                'confidence': 0.9,
                'in_zone': False
            }]
            
            timestamp = frame / fps
            events = event_manager.process_frame(track_data, frame, timestamp)
            
            if events:
                for e in events:
                    if e.event_type == EventType.LOITERING:
                        events_detected.append(e)
        
        if events_detected:
            first_event = events_detected[0]
            print(f"  Loitering detected at: {first_event.timestamp_seconds:.1f}s")
            print(f"  Expected around: {threshold}s")
            
            if first_event.timestamp_seconds >= threshold:
                print(f"  PASS: Event triggered after threshold")
            else:
                print(f"  FAIL: Event triggered too early")
        else:
            print(f"  No loitering detected in 8 seconds")
            if threshold > 8:
                print(f"  PASS: Threshold ({threshold}s) > test duration (8s)")
            else:
                print(f"  FAIL: Should have detected loitering")


if __name__ == "__main__":
    print("\n")
    
    # Run all tests
    test_loitering_simple()
    test_loitering_with_video_simulation()
    test_loitering_threshold_variations()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)