"""
Annotated video output writer.

Design Choice:
- Using OpenCV VideoWriter for compatibility
- Support for multiple codecs with fallbacks
- Frame buffering for consistent FPS
"""

from typing import Optional, Tuple
from pathlib import Path
import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VideoWriter:
    """
    Handles writing annotated frames to video file.
    
    Provides codec fallbacks and automatic directory creation.
    """
    
    # Codec fallback order
    CODECS = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
    ]
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: Optional[str] = None
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            fps: Output video FPS
            frame_size: (width, height) of output frames
            codec: Optional codec (e.g., 'mp4v'), auto-selects if None
        """
        self._output_path = Path(output_path)
        self._fps = fps
        self._frame_size = frame_size
        
        # Create output directory
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize writer
        self._writer = None
        self._frame_count = 0
        
        # Try to initialize with specified or fallback codecs
        if codec:
            self._init_writer(codec, self._output_path.suffix)
        else:
            self._init_with_fallback()
        
        if self._writer is None or not self._writer.isOpened():
            raise RuntimeError(f"Failed to initialize video writer for {output_path}")
        
        logger.info(
            f"VideoWriter initialized: {output_path}, "
            f"fps={fps}, size={frame_size}"
        )
    
    def _init_writer(self, codec: str, extension: str) -> bool:
        """Try to initialize writer with specific codec."""
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_file = self._output_path.with_suffix(extension)
        
        self._writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            self._fps,
            self._frame_size
        )
        
        if self._writer.isOpened():
            self._output_path = output_file
            return True
        
        return False
    
    def _init_with_fallback(self) -> None:
        """Try multiple codecs until one works."""
        for codec, extension in self.CODECS:
            if self._init_writer(codec, extension):
                logger.debug(f"Using codec: {codec}")
                return
        
        logger.warning("All codec fallbacks failed")
    
    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.
        
        Args:
            frame: Frame to write (BGR, numpy array)
        """
        if self._writer is None:
            return
        
        # Ensure frame matches expected size
        if frame.shape[1] != self._frame_size[0] or frame.shape[0] != self._frame_size[1]:
            frame = cv2.resize(frame, self._frame_size)
        
        self._writer.write(frame)
        self._frame_count += 1
    
    def release(self) -> None:
        """Release the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            
            logger.info(
                f"Video saved: {self._output_path}, "
                f"{self._frame_count} frames"
            )
    
    @property
    def output_path(self) -> Path:
        """Get the actual output path (may differ if codec fallback changed extension)."""
        return self._output_path
    
    @property
    def frame_count(self) -> int:
        """Get number of frames written."""
        return self._frame_count
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False