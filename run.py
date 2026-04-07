#!/usr/bin/env python3
"""
Video Surveillance System - Main CLI Entry Point

Usage:
    python run.py --video input.mp4 --zones zones.json --output results/
    python run.py --video input.mp4 --config configs/custom.yaml
    python run.py --video input.mp4 --benchmark --no-save
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import click
import json
from typing import Optional

from src.pipeline import VideoProcessor
from src.utils import Config, setup_logger, get_logger


@click.command()
@click.option(
    '--video', '-v',
    required=True,
    type=click.Path(exists=True),
    help='Path to input video file'
)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    default='configs/default_config.yaml',
    help='Path to configuration YAML file'
)
@click.option(
    '--zones', '-z',
    type=click.Path(exists=True),
    default=None,
    help='Path to zones JSON file (overrides config)'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    default='output/',
    help='Output directory for results'
)
@click.option(
    '--device', '-d',
    type=click.Choice(['cuda', 'cpu']),
    default=None,
    help='Device to run on (overrides config)'
)
@click.option(
    '--confidence',
    type=float,
    default=None,
    help='Detection confidence threshold (overrides config)'
)
@click.option(
    '--no-save',
    is_flag=True,
    default=False,
    help='Skip saving annotated video (faster for testing)'
)
@click.option(
    '--benchmark',
    is_flag=True,
    default=False,
    help='Show detailed FPS benchmark'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    default=False,
    help='Suppress progress bar'
)

@click.option(
    '--dashboard',
    is_flag=True,
    default=False,
    help='Show real-time FPS dashboard'
)
@click.option(
    '--batch',
    multiple=True,
    type=click.Path(exists=True),
    help='Process multiple videos in batch mode'
)
def main(
    video: str,
    config: str,
    zones: Optional[str],
    output: str,
    device: Optional[str],
    confidence: Optional[float],
    no_save: bool,
    benchmark: bool,
    quiet: bool,
    dashboard, batch
):
    """
    Video Surveillance System - Detection, Tracking & Event Recognition
    
    Process security camera footage to detect people, track them across
    frames, and identify events of interest (zone intrusion, loitering).
    
    Examples:
    
        # Basic usage
        python run.py --video sample.mp4
        
        # With custom zones and output
        python run.py --video sample.mp4 --zones my_zones.json --output results/
        
        # CPU mode with lower confidence
        python run.py --video sample.mp4 --device cpu --confidence 0.4
        
        # Benchmark mode (no video output)
        python run.py --video sample.mp4 --benchmark --no-save
    """
    try:
        # Load configuration
        cfg = Config(config_path=config, zones_path=zones)
        
        # Apply CLI overrides
        if device:
            cfg.detection.device = device
        
        if confidence:
            cfg.detection.confidence_threshold = confidence
        
        # Setup output paths
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video).stem
        output_video = output_dir / "videos" / f"{video_name}_annotated.mp4"
        
        cfg.video.output_path = str(output_video)
        cfg.events.event_log_path = str(output_dir / "events")
        
        # Initialize and run pipeline
        processor = VideoProcessor(cfg)
        
        results = processor.process_video(
            input_path=video,
            output_path=str(output_video) if not no_save else None,
            save_video=not no_save,
            show_progress=not quiet
        )
        
        # Print results summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"\nInput Video: {video}")
        
        if not no_save:
            print(f"Output Video: {results['output_video']}")
        
        print(f"\nEvent Logs:")
        for log_type, log_path in results['event_files'].items():
            print(f"  - {log_type.upper()}: {log_path}")
        
        print(f"\nMetrics:")
        for key, value in results['metrics'].items():
            print(f"  - {key}: {value}")
        
        print(f"\nEvents Summary:")
        for key, value in results['events_summary'].items():
            print(f"  - {key}: {value}")
        
        if benchmark:
            print(f"\nBenchmark:")
            print(f"  - Average FPS: {results['metrics']['average_fps']:.2f}")
            print(f"  - Total Time: {results['metrics']['processing_time_seconds']:.2f}s")
            print(f"  - Frames/Second: {results['metrics']['processed_frames'] / results['metrics']['processing_time_seconds']:.2f}")
        
        print("\n" + "=" * 60)
        
        # FPS Dashboard
        if dashboard:
            from src.utils.fps_dashboard import FPSDashboard
            fps_dash = FPSDashboard(enable=True)
            processor._dashboard = fps_dash
            fps_dash.start()
        
        # Batch processing
        if batch:
            from src.pipeline.batch_processor import BatchProcessor
            batch_proc = BatchProcessor(cfg, parallel=True)
            results = batch_proc.process_videos(list(batch), output_dir=output)
            
        return 0
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())