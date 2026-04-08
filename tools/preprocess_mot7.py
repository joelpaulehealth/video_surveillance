#!/usr/bin/env python3
"""
Preprocess MOT17 dataset for surveillance system.

Converts MOT17 image sequences to video files and optionally
extracts ground truth for evaluation.

Usage:
    python tools/preprocess_mot17.py --mot-dir data/MOT17 --output data/videos
    python tools/preprocess_mot17.py --mot-dir data/MOT17 --output data/videos --extract-gt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
import shutil
import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np


class MOT17Preprocessor:
    """
    Preprocessor for MOT17 dataset.
    
    Converts image sequences to video files and extracts
    ground truth annotations if needed.
    """
    
    def __init__(self, mot_dir: str, output_dir: str):
        """
        Initialize preprocessor.
        
        Args:
            mot_dir: Path to MOT17 directory (contains train/test folders)
            output_dir: Output directory for videos
        """
        self.mot_dir = Path(mot_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate MOT17 structure
        if not self.mot_dir.exists():
            raise ValueError(f"MOT17 directory not found: {mot_dir}")
        
        self.train_dir = self.mot_dir / "train"
        self.test_dir = self.mot_dir / "test"
        
        print(f"MOT17 directory: {self.mot_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def find_sequences(self, split: str = "train") -> List[Path]:
        """
        Find all sequences in train or test split.
        
        Args:
            split: "train" or "test"
            
        Returns:
            List of sequence directories
        """
        split_dir = self.train_dir if split == "train" else self.test_dir
        
        if not split_dir.exists():
            print(f"Warning: {split} directory not found: {split_dir}")
            return []
        
        sequences = [d for d in split_dir.iterdir() if d.is_dir()]
        sequences.sort()
        
        print(f"Found {len(sequences)} sequences in {split} split")
        return sequences
    
    def read_seqinfo(self, sequence_dir: Path) -> Dict:
        """
        Read seqinfo.ini file from sequence directory.
        
        Args:
            sequence_dir: Path to sequence directory
            
        Returns:
            Dictionary with sequence info
        """
        seqinfo_path = sequence_dir / "seqinfo.ini"
        
        if not seqinfo_path.exists():
            print(f"Warning: seqinfo.ini not found in {sequence_dir}")
            return {}
        
        info = {}
        with open(seqinfo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('['):
                    key, value = line.split('=', 1)
                    info[key.strip()] = value.strip()
        
        return info
    
    def images_to_video(
        self,
        sequence_dir: Path,
        output_path: Path,
        fps: Optional[float] = None,
        codec: str = 'mp4v'
    ) -> bool:
        """
        Convert image sequence to video file.
        
        Args:
            sequence_dir: Path to sequence directory
            output_path: Output video file path
            fps: Frame rate (reads from seqinfo.ini if None)
            codec: Video codec (mp4v, XVID, etc.)
            
        Returns:
            True if successful
        """
        img_dir = sequence_dir / "img1"
        
        if not img_dir.exists():
            print(f"Error: img1 directory not found in {sequence_dir}")
            return False
        
        # Get image files
        img_files = sorted(img_dir.glob("*.jpg"))
        if not img_files:
            img_files = sorted(img_dir.glob("*.png"))
        
        if not img_files:
            print(f"Error: No images found in {img_dir}")
            return False
        
        print(f"\nProcessing: {sequence_dir.name}")
        print(f"  Images found: {len(img_files)}")
        
        # Read seqinfo for FPS and resolution
        seqinfo = self.read_seqinfo(sequence_dir)
        
        if fps is None:
            fps = float(seqinfo.get('frameRate', 30))
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(img_files[0]))
        if first_img is None:
            print(f"Error: Cannot read first image: {img_files[0]}")
            return False
        
        height, width = first_img.shape[:2]
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Output: {output_path}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Failed to initialize video writer")
            return False
        
        # Write frames
        for img_path in tqdm(img_files, desc="  Converting"):
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"Warning: Cannot read {img_path}, skipping")
                continue
            
            # Resize if needed
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            out.write(img)
        
        out.release()
        
        print(f"  ✓ Video created successfully")
        return True
    
    def extract_ground_truth(
        self,
        sequence_dir: Path,
        output_dir: Path
    ) -> Optional[Path]:
        """
        Extract ground truth annotations.
        
        Args:
            sequence_dir: Path to sequence directory
            output_dir: Output directory for ground truth
            
        Returns:
            Path to ground truth file or None
        """
        gt_path = sequence_dir / "gt" / "gt.txt"
        
        if not gt_path.exists():
            print(f"  Warning: Ground truth not found: {gt_path}")
            return None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{sequence_dir.name}_gt.txt"
        
        shutil.copy(gt_path, output_file)
        print(f"  ✓ Ground truth extracted: {output_file}")
        
        return output_file
    
    def process_sequence(
        self,
        sequence_dir: Path,
        extract_gt: bool = False,
        fps: Optional[float] = None
    ) -> Dict:
        """
        Process a single sequence.
        
        Args:
            sequence_dir: Path to sequence directory
            extract_gt: Whether to extract ground truth
            fps: Frame rate (None = auto from seqinfo)
            
        Returns:
            Dictionary with processing results
        """
        sequence_name = sequence_dir.name
        
        # Output paths
        video_output = self.output_dir / f"{sequence_name}.mp4"
        gt_output_dir = self.output_dir / "ground_truth"
        
        result = {
            'sequence': sequence_name,
            'success': False,
            'video_path': None,
            'gt_path': None
        }
        
        # Convert to video
        success = self.images_to_video(sequence_dir, video_output, fps)
        
        if success:
            result['success'] = True
            result['video_path'] = str(video_output)
        
        # Extract ground truth if requested
        if extract_gt and success:
            gt_path = self.extract_ground_truth(sequence_dir, gt_output_dir)
            if gt_path:
                result['gt_path'] = str(gt_path)
        
        return result
    
    def process_all(
        self,
        split: str = "train",
        extract_gt: bool = False,
        sequences: Optional[List[str]] = None,
        fps: Optional[float] = None
    ) -> List[Dict]:
        """
        Process all sequences in a split.
        
        Args:
            split: "train" or "test"
            extract_gt: Whether to extract ground truth
            sequences: List of specific sequences to process (None = all)
            fps: Frame rate (None = auto from seqinfo)
            
        Returns:
            List of results for each sequence
        """
        all_sequences = self.find_sequences(split)
        
        # Filter sequences if specified
        if sequences:
            sequence_names = set(sequences)
            all_sequences = [
                s for s in all_sequences 
                if s.name in sequence_names
            ]
        
        if not all_sequences:
            print("No sequences to process!")
            return []
        
        print(f"\nProcessing {len(all_sequences)} sequences...\n")
        
        results = []
        for seq_dir in all_sequences:
            result = self.process_sequence(seq_dir, extract_gt, fps)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total sequences: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"{'='*60}\n")
        
        return results
    
    def create_summary(self, results: List[Dict], output_file: str) -> None:
        """
        Create JSON summary of processed sequences.
        
        Args:
            results: Processing results
            output_file: Output JSON file path
        """
        summary = {
            'total_sequences': len(results),
            'successful': sum(1 for r in results if r['success']),
            'sequences': results
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Preprocess MOT17 dataset for surveillance system'
    )
    
    parser.add_argument(
        '--mot-dir',
        required=True,
        help='Path to MOT17 directory'
    )
    
    parser.add_argument(
        '--output',
        default='data/videos',
        help='Output directory for videos'
    )
    
    parser.add_argument(
        '--split',
        choices=['train', 'test', 'both'],
        default='train',
        help='Which split to process'
    )
    
    parser.add_argument(
        '--extract-gt',
        action='store_true',
        help='Extract ground truth annotations'
    )
    
    parser.add_argument(
        '--sequences',
        nargs='+',
        help='Specific sequences to process (e.g., MOT17-02-SDP MOT17-04-SDP)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Override frame rate (default: read from seqinfo.ini)'
    )
    
    parser.add_argument(
        '--summary',
        default='output/mot17_summary.json',
        help='Path to save processing summary'
    )
    
    args = parser.parse_args()
    
    try:
        preprocessor = MOT17Preprocessor(args.mot_dir, args.output)
        
        all_results = []
        
        # Process requested split(s)
        if args.split in ['train', 'both']:
            print("\n" + "="*60)
            print("PROCESSING TRAIN SPLIT")
            print("="*60)
            results = preprocessor.process_all(
                split='train',
                extract_gt=args.extract_gt,
                sequences=args.sequences,
                fps=args.fps
            )
            all_results.extend(results)
        
        if args.split in ['test', 'both']:
            print("\n" + "="*60)
            print("PROCESSING TEST SPLIT")
            print("="*60)
            results = preprocessor.process_all(
                split='test',
                extract_gt=args.extract_gt,
                sequences=args.sequences,
                fps=args.fps
            )
            all_results.extend(results)
        
        # Save summary
        preprocessor.create_summary(all_results, args.summary)
        
        # Print next steps
        print("\nNext Steps:")
        print("-" * 60)
        print("1. Run surveillance system on processed videos:")
        print(f"   python run.py --video {args.output}/MOT17-02-SDP.mp4")
        print()
        print("2. Or process all videos in batch:")
        print(f"   python run.py --batch {args.output}/*.mp4")
        print()
        if args.extract_gt:
            print("3. Evaluate tracking performance:")
            print(f"   python tools/evaluate_mot.py \\")
            print(f"     --video {args.output}/MOT17-02-SDP.mp4 \\")
            print(f"     --gt {args.output}/ground_truth/MOT17-02-SDP_gt.txt")
        print("-" * 60)
        
        return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())