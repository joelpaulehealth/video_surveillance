"""
Batch video processing with parallel execution support.

Design Choice:
- Process multiple videos in sequence or parallel
- Aggregate results and statistics
- Error handling per video (don't stop batch on single failure)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
import json

from .video_processor import VideoProcessor
from ..utils import Config, get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """
    Process multiple videos in batch mode.
    
    Supports sequential and parallel processing with
    aggregated results and error handling.
    """
    
    def __init__(
        self,
        config: Config,
        parallel: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            config: System configuration
            parallel: Enable parallel processing
            max_workers: Number of parallel workers (None = CPU count)
        """
        self._config = config
        self._parallel = parallel
        
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)
        
        self._max_workers = max_workers
        
        logger.info(
            f"BatchProcessor initialized: "
            f"parallel={parallel}, workers={max_workers}"
        )
    
    def process_videos(
        self,
        video_paths: List[str],
        output_dir: str = "output/batch",
        save_videos: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple videos.
        
        Args:
            video_paths: List of video file paths
            output_dir: Base output directory
            save_videos: Whether to save annotated videos
            show_progress: Show progress for each video
            
        Returns:
            Dictionary with aggregated results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(video_paths)} videos...")
        
        if self._parallel:
            results = self._process_parallel(
                video_paths, output_path, save_videos, show_progress
            )
        else:
            results = self._process_sequential(
                video_paths, output_path, save_videos, show_progress
            )
        
        # Aggregate results
        summary = self._aggregate_results(results)
        
        # Save batch summary
        summary_path = output_path / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch processing complete. Summary saved to {summary_path}")
        
        return summary
    
    def _process_sequential(
        self,
        video_paths: List[str],
        output_path: Path,
        save_videos: bool,
        show_progress: bool
    ) -> Dict[str, Any]:
        """Process videos sequentially."""
        results = {}
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"Processing video {i}/{len(video_paths)}: {video_path}")
            
            try:
                result = self._process_single_video(
                    video_path, output_path, save_videos, show_progress
                )
                results[video_path] = {
                    'status': 'success',
                    'result': result
                }
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results[video_path] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def _process_parallel(
        self,
        video_paths: List[str],
        output_path: Path,
        save_videos: bool,
        show_progress: bool
    ) -> Dict[str, Any]:
        """Process videos in parallel."""
        results = {}
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        # Note: Each process needs its own GPU if using CUDA
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all jobs
            future_to_video = {
                executor.submit(
                    self._process_single_video_static,
                    video_path,
                    str(output_path),
                    save_videos,
                    False,  # Disable progress in parallel mode
                    self._config
                ): video_path
                for video_path in video_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_video):
                video_path = 