#!/usr/bin/env python3
"""
Video to 3D Keypoints CSV Extractor
Processes a video and generates a CSV file with 3D keypoints for all frames.
Uses 2D pose detection + 3D pose lifting.
"""
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import cv2

from mmpose.apis.inferencers import MMPoseInferencer

# H36M body keypoint names (17 keypoints for 3D)
H36M_KEYPOINT_NAMES = [
    'pelvis',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'spine',
    'thorax',
    'nose',
    'head',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'right_shoulder',
    'right_elbow',
    'right_wrist'
]


def extract_3d_keypoints_from_video(video_path, output_csv, pose2d_config, pose2d_weights,
                                     pose3d_config, pose3d_weights,
                                     det_model=None, det_weights=None, device='cuda'):
    """
    Extract 3D keypoints from a video and save to CSV.
    
    Args:
        video_path: Path to input video file
        output_csv: Path to output CSV file
        pose2d_config: Path to 2D pose model config file
        pose2d_weights: Path to 2D pose model weights file
        pose3d_config: Path to 3D pose lifter config file
        pose3d_weights: Path to 3D pose lifter weights file
        det_model: Path to detector config file (optional)
        det_weights: Path to detector weights file (optional)
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"\n{'='*60}")
    print(f"Video to 3D Keypoints CSV Extractor")
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
    
    # Initialize the inferencer with 3D pose lifting
    print("Initializing MMPoseInferencer with 3D pose lifting...")
    
    # If detector model is not specified, use RTMDet as default
    if det_model is None:
        det_model = 'rtmdet-m'
    
    inferencer = MMPoseInferencer(
        pose2d=pose2d_config,
        pose2d_weights=pose2d_weights,
        pose3d=pose3d_config,
        pose3d_weights=pose3d_weights,
        det_model=det_model,
        det_weights=det_weights,
        device=device,
        show_progress=True
    )
    print("Inferencer initialized!\n")
    
    # Prepare CSV headers
    csv_headers = ['frame']
    for keypoint_name in H36M_KEYPOINT_NAMES:
        csv_headers.extend([f'{keypoint_name}_x', f'{keypoint_name}_y', f'{keypoint_name}_z', f'{keypoint_name}_conf'])
    
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
        pred_out_dir='',
        disable_rebase_keypoint=False  # Keep floor-level normalization
    ):
        # Extract predictions from result
        predictions = result.get('predictions', [[]])
        
        row = [frame_idx]
        
        if predictions and len(predictions[0]) > 0:
            # Get the first person's 3D keypoints
            person_data = predictions[0][0]
            
            # Check if 3D keypoints are available
            if 'keypoints_3d' in person_data:
                keypoints_3d = person_data.get('keypoints_3d', [])
                keypoint_scores = person_data.get('keypoint_scores_3d', [])
                
                # Add 3D keypoints to row (x, y, z, confidence)
                for i in range(len(H36M_KEYPOINT_NAMES)):
                    if i < len(keypoints_3d):
                        x, y, z = keypoints_3d[i]
                        conf = keypoint_scores[i] if (keypoint_scores is not None and i < len(keypoint_scores)) else 1.0
                        row.extend([x, y, z, conf])
                    else:
                        # Missing keypoint
                        row.extend([0.0, 0.0, 0.0, 0.0])
            else:
                # No 3D keypoints available (fallback)
                for _ in range(len(H36M_KEYPOINT_NAMES)):
                    row.extend([0.0, 0.0, 0.0, 0.0])
        else:
            # No person detected in this frame
            for _ in range(len(H36M_KEYPOINT_NAMES)):
                row.extend([0.0, 0.0, 0.0, 0.0])
        
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
        description='Extract 3D keypoints from video and save to CSV'
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
        help='Path to output CSV file (default: same as video with _3d.csv extension)'
    )
    parser.add_argument(
        '--pose2d-config',
        type=str,
        default='configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py',
        help='Path to 2D pose model config file'
    )
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default='ProbPose-s.pth',
        help='Path to 2D pose model weights file'
    )
    parser.add_argument(
        '--pose3d-config',
        type=str,
        default='configs/body_3d_keypoint/pose_lift/h36m/pose-lift_simplebaseline3d_8xb64-200e_h36m.py',
        help='Path to 3D pose lifter config file'
    )
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to 3D pose lifter weights file (will auto-download if not specified)'
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
        args.output = str(video_path.with_suffix('')) + '_3d.csv'
    
    # Extract 3D keypoints
    extract_3d_keypoints_from_video(
        video_path=args.video_path,
        output_csv=args.output,
        pose2d_config=args.pose2d_config,
        pose2d_weights=args.pose2d_weights,
        pose3d_config=args.pose3d_config,
        pose3d_weights=args.pose3d_weights,
        det_model=args.det_model,
        det_weights=args.det_weights,
        device=args.device
    )


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Video to 3D Keypoints CSV Extractor
Processes a video and generates a CSV file with 3D keypoints for all frames.
Uses 2D pose detection + 3D pose lifting.
"""
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import cv2

from mmpose.apis.inferencers import MMPoseInferencer

# H36M body keypoint names (17 keypoints for 3D)
H36M_KEYPOINT_NAMES = [
    'pelvis',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'spine',
    'thorax',
    'nose',
    'head',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'right_shoulder',
    'right_elbow',
    'right_wrist'
]


def extract_3d_keypoints_from_video(video_path, output_csv, pose2d_config, pose2d_weights,
                                     pose3d_config, pose3d_weights,
                                     det_model=None, det_weights=None, device='cuda'):
    """
    Extract 3D keypoints from a video and save to CSV.
    
    Args:
        video_path: Path to input video file
        output_csv: Path to output CSV file
        pose2d_config: Path to 2D pose model config file
        pose2d_weights: Path to 2D pose model weights file
        pose3d_config: Path to 3D pose lifter config file
        pose3d_weights: Path to 3D pose lifter weights file
        det_model: Path to detector config file (optional)
        det_weights: Path to detector weights file (optional)
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"\n{'='*60}")
    print(f"Video to 3D Keypoints CSV Extractor")
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
    
    # Initialize the inferencer with 3D pose lifting
    print("Initializing MMPoseInferencer with 3D pose lifting...")
    
    # If detector model is not specified, use RTMDet as default
    if det_model is None:
        det_model = 'rtmdet-m'
    
    inferencer = MMPoseInferencer(
        pose2d=pose2d_config,
        pose2d_weights=pose2d_weights,
        pose3d=pose3d_config,
        pose3d_weights=pose3d_weights,
        det_model=det_model,
        det_weights=det_weights,
        device=device,
        show_progress=True
    )
    print("Inferencer initialized!\n")
    
    # Prepare CSV headers
    csv_headers = ['frame']
    for keypoint_name in H36M_KEYPOINT_NAMES:
        csv_headers.extend([f'{keypoint_name}_x', f'{keypoint_name}_y', f'{keypoint_name}_z', f'{keypoint_name}_conf'])
    
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
        pred_out_dir='',
        disable_rebase_keypoint=False  # Keep floor-level normalization
    ):
        # Extract predictions from result
        predictions = result.get('predictions', [[]])
        
        row = [frame_idx]
        
        if predictions and len(predictions[0]) > 0:
            # Get the first person's 3D keypoints
            person_data = predictions[0][0]
            
            # Check if 3D keypoints are available
            if 'keypoints_3d' in person_data:
                keypoints_3d = person_data.get('keypoints_3d', [])
                keypoint_scores = person_data.get('keypoint_scores_3d', [])
                
                # Add 3D keypoints to row (x, y, z, confidence)
                for i in range(len(H36M_KEYPOINT_NAMES)):
                    if i < len(keypoints_3d):
                        x, y, z = keypoints_3d[i]
                        conf = keypoint_scores[i] if (keypoint_scores is not None and i < len(keypoint_scores)) else 1.0
                        row.extend([x, y, z, conf])
                    else:
                        # Missing keypoint
                        row.extend([0.0, 0.0, 0.0, 0.0])
            else:
                # No 3D keypoints available (fallback)
                for _ in range(len(H36M_KEYPOINT_NAMES)):
                    row.extend([0.0, 0.0, 0.0, 0.0])
        else:
            # No person detected in this frame
            for _ in range(len(H36M_KEYPOINT_NAMES)):
                row.extend([0.0, 0.0, 0.0, 0.0])
        
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
        description='Extract 3D keypoints from video and save to CSV'
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
        help='Path to output CSV file (default: same as video with _3d.csv extension)'
    )
    parser.add_argument(
        '--pose2d-config',
        type=str,
        default='configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py',
        help='Path to 2D pose model config file'
    )
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default='ProbPose-s.pth',
        help='Path to 2D pose model weights file'
    )
    parser.add_argument(
        '--pose3d-config',
        type=str,
        default='configs/body_3d_keypoint/pose_lift/h36m/pose-lift_simplebaseline3d_8xb64-200e_h36m.py',
        help='Path to 3D pose lifter config file'
    )
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to 3D pose lifter weights file (will auto-download if not specified)'
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
        args.output = str(video_path.with_suffix('')) + '_3d.csv'
    
    # Extract 3D keypoints
    extract_3d_keypoints_from_video(
        video_path=args.video_path,
        output_csv=args.output,
        pose2d_config=args.pose2d_config,
        pose2d_weights=args.pose2d_weights,
        pose3d_config=args.pose3d_config,
        pose3d_weights=args.pose3d_weights,
        det_model=args.det_model,
        det_weights=args.det_weights,
        device=args.device
    )


if __name__ == '__main__':
    main()
