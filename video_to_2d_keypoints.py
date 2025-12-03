#!/usr/bin/env python3
"""
Video to 2D Keypoints CSV Extractor
Processes a video and generates a CSV file with 2D keypoints for all frames.
"""
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import cv2

from mmpose.apis.inferencers import MMPoseInferencer

# COCO body keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


def extract_keypoints_from_video(video_path, output_csv, model_config, model_weights, 
                                 det_model=None, det_weights=None, device='cuda'):
    """
    Extract 2D keypoints from a video and save to CSV.
    
    Args:
        video_path: Path to input video file
        output_csv: Path to output CSV file
        model_config: Path to model config file
        model_weights: Path to model weights file
        det_model: Path to detector config file (optional)
        det_weights: Path to detector weights file (optional)
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"\n{'='*60}")
    print(f"Video to 2D Keypoints CSV Extractor")
    print(f"{'='*60}")
    print(f"Input Video: {video_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Video Info:")
    print(f"  Total Frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f}s\n")
    
    # Initialize the inferencer
    print("Initializing MMPoseInferencer...")
    
    # If detector model is not specified, use RTMDet as default
    if det_model is None:
        det_model = 'rtmdet-m'
    
    inferencer = MMPoseInferencer(
        pose2d=model_config,
        pose2d_weights=model_weights,
        det_model=det_model,
        det_weights=det_weights,
        device=device,
        show_progress=True
    )
    print("Inferencer initialized!\n")
    
    # Prepare CSV headers
    csv_headers = ['frame']
    for keypoint_name in COCO_KEYPOINT_NAMES:
        csv_headers.extend([f'{keypoint_name}_x', f'{keypoint_name}_y', f'{keypoint_name}_conf'])
    
    # Process video and extract keypoints
    print("Processing video...")
    all_keypoints = []
    
    # Run inference
    frame_idx = 0
    for result in inferencer(
        video_path,
        show=False,
        draw_bbox=False,
        draw_heatmap=False,
        pred_out_dir=''
    ):
        # Extract predictions from result
        predictions = result.get('predictions', [[]])
        
        row = [frame_idx]
        
        if predictions and len(predictions[0]) > 0:
            # Get the first person's keypoints
            person_data = predictions[0][0]
            keypoints = person_data.get('keypoints', [])
            keypoint_scores = person_data.get('keypoint_scores', [])
            
            # Add keypoints to row (x, y, confidence)
            for i in range(len(COCO_KEYPOINT_NAMES)):
                if i < len(keypoints):
                    x, y = keypoints[i]
                    conf = keypoint_scores[i] if i < len(keypoint_scores) else 0.0
                    row.extend([x, y, conf])
                else:
                    # Missing keypoint
                    row.extend([0.0, 0.0, 0.0])
        else:
            # No person detected in this frame
            for _ in range(len(COCO_KEYPOINT_NAMES)):
                row.extend([0.0, 0.0, 0.0])
        
        all_keypoints.append(row)
        frame_idx += 1
    
    # Write to CSV
    print(f"\nWriting results to CSV...")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(all_keypoints)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Processed {frame_idx} frames")
    print(f"CSV saved to: {output_csv}")
    print(f"{'='*60}\n")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description='Extract 2D keypoints from video and save to CSV'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Path to output CSV file (default: same as video with .csv extension)'
    )
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        default='configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py',
        help='Path to model config file'
    )
    parser.add_argument(
        '--weights',
        '-w',
        type=str,
        default='ProbPose-s.pth',
        help='Path to model weights file'
    )
    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Detector model name or config path (default: rtmdet-m)'
    )
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to detector weights file (optional)'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        video_path = Path(args.video_path)
        args.output = str(video_path.with_suffix('.csv'))
    
    # Extract keypoints
    extract_keypoints_from_video(
        video_path=args.video_path,
        output_csv=args.output,
        model_config=args.config,
        model_weights=args.weights,
        det_model=args.det_model,
        det_weights=args.det_weights,
        device=args.device
    )


if __name__ == '__main__':
    main()
